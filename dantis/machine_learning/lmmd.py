"""
LMDD (Linear Method for Deviation Detection) implementation for DANTIS.

LMDD es un método de detección de anomalías basado en suavizado lineal.
Asume que, en una serie temporal normal, el valor en el tiempo 't' puede ser
aproximado linealmente por sus vecinos.

El algoritmo:
1. Calcula un "valor esperado" para cada punto basándose en el promedio de sus
   k vecinos más cercanos (excluyendo el punto central).
2. Calcula la desviación (residuo) entre el valor real y el esperado.
3. Cuanto mayor es la desviación, mayor es el anomaly score.

Referencia conceptual:
Arning, A., Agrawal, R., & Raghavan, P. (1996). A linear method for deviation detection in large databases.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from typing import Optional, Dict, Any

from .. import algorithmbase
from .. import utils
try:
    import joblib
except Exception:
    joblib = None

class LMDD(algorithmbase.AlgorithmBase):
    """
    Linear Method for Deviation Detection (LMDD).
    Utiliza convolución para comparar cada punto con el promedio de sus vecinos.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        """
        Retorna los hiperparámetros por defecto.
        
        - window_size: (int) Número de vecinos a considerar a cada lado.
          Ejemplo: si window_size=3, el kernel total es de 7 (3 izq + centro + 3 der).
        - contamination: (float) Proporción de anomalías esperada.
        """
        return {
            "window_size": 5,
            "contamination": 0.1,
            "verbose": False
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        self._model = None
        self._last_scores = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            self._hyperparameter = defaults.copy()
            return

        def _hp_template(window_size=5, contamination=0.1, verbose=False):
            return None

        pm = utils.ParameterManagement(_hp_template)
        try:
            coerced = pm.check_hyperparameter_type(hyperparameter)
        except Exception as e:
            raise ValueError(f"LMDD Hyperparameter Error: {e}")

        merged = pm.complete_parameters(coerced)
        
        for k, v in hyperparameter.items():
            if k not in merged:
                merged[k] = v

        self._hyperparameter = merged

    def _validate_input(self, x) -> np.ndarray:
        """Normaliza la entrada a array numpy 1D."""
        if isinstance(x, pd.DataFrame):
            if x.shape[1] >= 2:
                val = x.iloc[:, 1].values
            else:
                val = x.iloc[:, 0].values
        elif isinstance(x, pd.Series):
            val = x.values
        else:
            val = np.array(x)
            
        if val.ndim > 1:
            val = val.flatten()
        
        return val.astype(float)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        LMDD es un método basado en ventanas deslizantes (no paramétrico en el sentido de entrenamiento).
        Simplemente validamos los datos.
        """
        series = self._validate_input(x_train)
        self._model = "LMDD_Ready"
        # Para detectores basados en convolución podemos precomputar los scores
        # durante el fit si se proporciona un conjunto de entrenamiento.
        try:
            self._last_scores = self.decision_function(series)
        except Exception:
            self._last_scores = None
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula el score de anomalía.
        Score = |x_t - mean(neighbors)|
        
        Utiliza convolución 1D para máxima eficiencia.
        """
        series = self._validate_input(x)
        params = self._hyperparameter
        window_size = int(params.get("window_size", 5))
        
        n = len(series)
        if n < window_size * 2 + 1:
            # Si la serie es muy corta, fallback a desviación de la media global
            return np.abs(series - np.mean(series))

        # Construcción del Kernel de Convolución
        # Queremos el promedio de los vecinos, ignorando el centro.
        # Kernel length = 2 * window_size + 1
        # Pesos = 1/(2*w) para vecinos, 0 para el centro.
        
        kernel_len = 2 * window_size + 1
        weights = np.ones(kernel_len)
        weights[window_size] = 0  # Anular el punto central (el que queremos predecir)
        weights = weights / weights.sum() # Normalizar para que sume 1 (promedio)
        
        # Calcular la "predicción" suavizada usando convolución
        # mode='reflect' maneja los bordes suavemente replicando los extremos
        smoothed = convolve1d(series, weights, mode='reflect')
        
        # El score es la diferencia absoluta (desviación L1)
        # También podría usarse la diferencia al cuadrado (L2) para penalizar más los picos
        scores = np.abs(series - smoothed)
        # Actualizamos caché de últimos scores
        try:
            self._last_scores = scores
        except Exception:
            pass

        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Devuelve etiquetas binarias (0/1).
        """
        scores = self.decision_function(x)
        
        contamination = float(self._hyperparameter.get("contamination", 0.1))
        contamination = max(0.001, min(0.5, contamination))
        
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        
        labels = (scores > threshold).astype(int)
        return labels

    def save_model(self, path: str):
        """
        Guarda el estado (hiperparámetros y últimos scores) usando `joblib`.
        """
        if joblib is None:
            raise ImportError("Guardar modelo requiere joblib. Instala con: pip install joblib")

        payload = {
            "hyperparameter": self._hyperparameter,
            "model": self._model,
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
        self._hyperparameter = payload.get("hyperparameter", self.get_default_hyperparameters())
        self._model = payload.get("model")
        self._last_scores = payload.get("last_scores")
        return self