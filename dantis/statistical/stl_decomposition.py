"""
STL Decomposition Outlier Detector for DANTIS.

Este algoritmo utiliza la descomposición STL (Seasonal-Trend decomposition using LOESS)
para separar una serie temporal en tres componentes:
1. Tendencia (Trend)
2. Estacionalidad (Seasonal)
3. Residuo (Residual)

La hipótesis es que las anomalías se encuentran en el componente Residual.
El 'anomaly score' es la magnitud absoluta del residuo.

Referencias:
    Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990).
    STL: A seasonal-trend decomposition procedure based on loess.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

try:
    from statsmodels.tsa.seasonal import STL
except ImportError:
    STL = None
try:
    import joblib
except Exception:
    joblib = None

from .. import algorithmbase
from .. import utils

class STLDetector(algorithmbase.AlgorithmBase):
    """
    Detector de anomalías basado en descomposición STL.
    Requiere 'statsmodels'.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        """
        Retorna los parámetros por defecto.
        
        - period: (int) Periodicidad de la serie (ej. 12 para mensual, 24 para horaria, 7 diaria).
          Si es None, intentará inferirlo (riesgoso), mejor especificarlo.
        - robust: (bool) Si True, usa ponderación robusta en LOESS (menos sensible a outliers).
        - contamination: (float) Proporción de anomalías esperada.
        - seasonal_smoother: (int) Longitud de la ventana del suavizador estacional (debe ser impar).
        """
        return {
            "period": None,
            "robust": True,
            "contamination": 0.1,
            "seasonal_smoother": 7,
            "verbose": False
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        if STL is None:
            raise ImportError("STLDetector requiere statsmodels. Instala con: pip install statsmodels")
            
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        self.resid_std_ = None  # Desviación estándar de los residuos (calculada en fit)
        self._model = None
        self._last_scores = None
        self._last_series_len = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            self._hyperparameter = defaults.copy()
            return

        def _hp_template(period=12, robust=True, contamination=0.1, seasonal_smoother=7, verbose=False):
            return None

        pm = utils.ParameterManagement(_hp_template)
        try:
            coerced = pm.check_hyperparameter_type(hyperparameter)
        except Exception as e:
            raise ValueError(f"STLDetector Hyperparameter Error: {e}")

        merged = pm.complete_parameters(coerced)
        
        # Copiar extras
        for k, v in hyperparameter.items():
            if k not in merged:
                merged[k] = v

        self._hyperparameter = merged

    def _infer_period(self, series: pd.Series) -> int:
        """
        Inferir un periodo estacional en número de muestras.

        Estrategia:
        - Si el índice es `DatetimeIndex`, intentar `pd.infer_freq` (siempre que sea útil).
        - Intentar localizar la frecuencia dominante mediante FFT (transformada rápida).
        - Si FFT falla o es ruidosa, usar autocorrelación simple y elegir el lag con mayor autocorrelación.

        Devuelve al menos 1. Este valor se usa como `period` para STL.
        """
        n = len(series)
        if n < 6:
            return 1

        # Intento rápido con DatetimeIndex
        try:
            idx = series.index
            if isinstance(idx, pd.DatetimeIndex):
                freq = pd.infer_freq(idx)
                if freq is not None:
                    # Si infer_freq proporciona una frecuencia (e.g. 'D','H'),
                    # no podemos deducir automáticamente el periodo estacional
                    # con seguridad, así que dejamos que FFT lo determine.
                    pass
        except Exception:
            pass

        x = series.values.astype(float)
        x = x - np.nanmean(x)

        # FFT-based dominant period detection
        try:
            # rfft: frequencies from 0 .. Nyquist
            freqs = np.fft.rfftfreq(n, d=1.0)
            fft = np.fft.rfft(x)
            power = np.abs(fft)
            # ignorar componente DC
            if power.shape[0] <= 1:
                raise RuntimeError("FFT too short")
            power[0] = 0.0
            idx = int(np.argmax(power))
            dominant_freq = freqs[idx]
            if dominant_freq <= 0 or np.isclose(dominant_freq, 0.0):
                raise RuntimeError("No dominant frequency")
            period = int(max(1, round(1.0 / dominant_freq)))
            # defensiva: no mayor que n/2
            if period > max(2, n // 2):
                period = max(1, n // 4)
            return period
        except Exception:
            # Fallback: autocorrelación simple
            try:
                max_lag = min(n // 2, 100)
                x = x - np.mean(x)
                acfs = []
                for lag in range(1, max_lag + 1):
                    v = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                    acfs.append(0.0 if np.isnan(v) else v)
                best = int(np.argmax(np.abs(acfs))) + 1
                return max(1, best)
            except Exception:
                return 1

    def _validate_input(self, x) -> pd.Series:
        """
        STL requiere una Serie de Pandas, idealmente con índice temporal o numérico.
        Si hay NaNs, STL fallará, así que hacemos una interpolación lineal simple.
        """
        if isinstance(x, pd.DataFrame):
            if x.shape[1] >= 2:
                # Asumimos formato (timestamp, value)
                series = x.set_index(x.columns[0]).iloc[:, 0]
            else:
                series = x.iloc[:, 0]
        elif isinstance(x, pd.Series):
            series = x.copy()
        else:
            # Array numpy
            x = np.array(x).flatten()
            series = pd.Series(x)

        # Manejo de NaNs (STL no soporta NaNs)
        if series.isnull().any():
            series = series.interpolate(method='linear').bfill().ffill()
            
        return series

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        Ajusta el modelo. En STL para detección de anomalías, 'ajustar' significa
        calcular las estadísticas de los residuos sobre el conjunto de entrenamiento
        para establecer una línea base de "ruido normal".
        """
        series = self._validate_input(x_train)
        params = self._hyperparameter
        
        period_param = params.get("period", None)
        if period_param is None or (isinstance(period_param, str) and period_param.lower() == 'auto'):
            period = int(self._infer_period(series))
        else:
            period = int(period_param)
        robust = bool(params.get("robust", True))
        seasonal = int(params.get("seasonal_smoother", 7))

        stl_res = STL(series, period=period, robust=robust, seasonal=seasonal).fit()

        # Guardamos la desviación estándar robusta (MAD) de los residuos
        # para tener una referencia de escala.
        resid = stl_res.resid
        median_resid = np.median(resid)
        # Median Absolute Deviation
        mad = np.median(np.abs(resid - median_resid))

        # Estimador robusto de sigma: k * MAD (k approx 1.4826 para distribuciones normales)
        self.resid_std_ = 1.4826 * mad

        # Guardar modelo y últimos scores para posible reutilización
        self._model = stl_res
        self._last_scores = np.abs(resid.values)
        self._last_series_len = len(series)

        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula el score de anomalía.
        
        Score = |Residuo STL|
        
        Nota: STL es un algoritmo de procesamiento por lotes (batch). 
        Procesa la serie completa 'x' para descomponerla.
        """
        series = self._validate_input(x)
        params = self._hyperparameter
        
        period_param = params.get("period", None)
        if period_param is None or (isinstance(period_param, str) and period_param.lower() == 'auto'):
            period = int(self._infer_period(series))
        else:
            period = int(period_param)
        robust = bool(params.get("robust", True))
        seasonal = int(params.get("seasonal_smoother", 7))
        
        # Si ya tenemos un modelo ajustado para una serie de la misma longitud,
        # reutilizamos los últimos scores calculados en `fit`.
        if self._last_scores is not None and len(series) == self._last_series_len:
            return self._last_scores

        # Si la serie es más corta que el periodo, STL puede fallar.
        if len(series) < 2 * period:
            # Fallback: Retornar desviación absoluta respecto de la mediana
            return np.abs(series - series.median()).values

        try:
            res = STL(series, period=period, robust=robust, seasonal=seasonal).fit()
            # El score es la magnitud absoluta del residuo
            scores = np.abs(res.resid.values)
            # Actualizamos caché
            self._model = res
            self._last_scores = scores
            self._last_series_len = len(series)
            return scores
        except Exception as e:
            if params.get("verbose"):
                print(f"STL decomposition failed: {e}")
            return np.zeros(len(series))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Devuelve 1 si el residuo es anómalamente alto basado en la contaminación.
        """
        scores = self.decision_function(x)
        
        contamination = float(self._hyperparameter.get("contamination", 0.1))
        contamination = max(0.001, min(0.5, contamination))
        
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        
        labels = (scores > threshold).astype(int)
        return labels

    def save_model(self, path: str):
        """
        Guarda el estado entrenado (modelo STL y metadatos) en `path` usando `joblib`.
        """
        if joblib is None:
            raise ImportError("Guardar modelo requiere joblib. Instala con: pip install joblib")

        payload = {
            "model": self._model,
            "hyperparameter": self._hyperparameter,
            "resid_std_": self.resid_std_,
            "last_scores": self._last_scores,
        }
        joblib.dump(payload, path)

    def load_model(self, path: str):
        """
        Carga el estado previamente guardado con `save_model`.
        """
        if joblib is None:
            raise ImportError("Cargar modelo requiere joblib. Instala con: pip install joblib")

        payload = joblib.load(path)
        self._model = payload.get("model")
        self._hyperparameter = payload.get("hyperparameter", self.get_default_hyperparameters())
        self.resid_std_ = payload.get("resid_std_")
        self._last_scores = payload.get("last_scores")
        if self._last_scores is not None:
            self._last_series_len = len(self._last_scores)
        return self