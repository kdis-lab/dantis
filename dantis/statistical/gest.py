"""
GESD (Generalized Extreme Studentized Deviate) implementation for DANTIS.

Este módulo implementa el test de GESD, que es una generalización iterativa del
Test de Grubbs. Mientras que Grubbs asume un solo outlier, GESD puede detectar
múltiples outliers sin sufrir el "efecto de enmascaramiento" (masking effect)
donde un outlier extremo distorsiona la media y desviación estándar ocultando a otros.

Algoritmo:
1. Se define un límite superior de outliers a buscar (max_outliers).
2. Iterativamente se elimina la observación que más se desvía de la media.
3. Se calcula un estadístico de prueba (R) y un valor crítico (Lambda) basado en la distribución t-Student.
4. Se determina el número óptimo de outliers buscando el último índice donde R > Lambda.

Nota: Si max_outliers = 1, este algoritmo es matemáticamente equivalente al Test de Grubbs.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, Any, List, Union
import joblib
import logging

# Importaciones relativas a dantis
from .. import algorithmbase
from .. import utils

class GESD(algorithmbase.AlgorithmBase):
    """
    Generalized Extreme Studentized Deviate (GESD) Detector.
    Incluye la funcionalidad del Test de Grubbs.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        """
        Retorna los hiperparámetros por defecto.

        - max_outliers: Número máximo de outliers a detectar (int) o fracción del dataset (float 0.0-0.49).
          Si se establece en 1, el comportamiento es idéntico al Test de Grubbs.
        - alpha: Nivel de significancia estadística (usualmente 0.05).
        """
        return {
            "max_outliers": 0.10,  # Por defecto busca hasta en el 10% de los datos
            "alpha": 0.05,
            "verbose": False
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        
        if hyperparameter is None:
            self._hyperparameter = defaults.copy()
            return

        def _hp_template(max_outliers=0.10, alpha=0.05, verbose=False):
            return None

        pm = utils.ParameterManagement(_hp_template)
        try:
            coerced = pm.check_hyperparameter_type(hyperparameter)
        except Exception as e:
            raise ValueError(f"GESD Hyperparameter Error: {e}")

        merged = pm.complete_parameters(coerced)
        
        for k, v in hyperparameter.items():
            if k not in merged:
                merged[k] = v

        self._hyperparameter = merged

    def _validate_input(self, x) -> np.ndarray:
        """Normaliza la entrada a un array numpy 1D de floats."""
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

    def _preprocess(self, x) -> np.ndarray:
        """Handle NaNs and basic preprocessing: linear interpolation then ffill/bfill then median."""
        arr = self._validate_input(x)
        s = pd.Series(arr)
        if s.isna().any():
            s = s.interpolate(method='linear', limit_direction='both')
            if s.isna().any():
                s = s.fillna(method='ffill').fillna(method='bfill')
            if s.isna().any():
                median = float(np.nanmedian(arr)) if np.isfinite(np.nanmedian(arr)) else 0.0
                s = s.fillna(median)
        return s.values

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        GESD es un test estadístico transductivo (analiza el conjunto actual).
        El método fit aquí es mayormente simbólico para cumplir con la API,
        pero realizamos una validación básica de que los datos son suficientes.
        """
        data = self._validate_input(x_train)
        if len(data) < 3:
            raise ValueError("GESD requiere al menos 3 puntos de datos.")

        # No hay "entrenamiento" persistente en tests estadísticos puros,
        # pero para mantener la convención del proyecto guardamos scores
        # y marcamos el modelo como ajustado.
        scores = self.decision_function(data)
        self._last_scores = scores
        self._model = "GESD_Ready"
        return scores

    def save_model(self, path_model: str):
        payload = {
            'last_scores': getattr(self, '_last_scores', None),
            '_hyperparameter': self._hyperparameter,
            '_model': getattr(self, '_model', None)
        }
        if not path_model.endswith('.joblib'):
            path_model = path_model + '.joblib'
        joblib.dump(payload, path_model)

    def load_model(self, path_model: str):
        if not path_model.endswith('.joblib'):
            raise ValueError('path_model must be a .joblib file')
        payload = joblib.load(path_model)
        self._last_scores = payload.get('last_scores')
        self._hyperparameter = payload.get('_hyperparameter', self.get_default_hyperparameters())
        self._model = payload.get('_model', None)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula el 'Anomaly Score'.
        
        Para GESD/Grubbs, la medida más natural de anomalía continua es el 
        Z-score absoluto modificado (desviación de la media dividido por std).
        
        Aunque GESD es un proceso iterativo binario, el score base ayuda a entender
        la magnitud de la desviación.
        """
        data = self._preprocess(x)
        
        # Manejo de varianza cero
        std_dev = np.std(data, ddof=1)
        if std_dev == 0:
            return np.zeros(len(data))
            
        mean_val = np.mean(data)
        z_scores = np.abs((data - mean_val) / std_dev)
        
        return z_scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Ejecuta el algoritmo iterativo GESD para devolver etiquetas binarias (0/1).
        """
        data = self._preprocess(x).copy() # Copy para no mutar original durante el loop
        original_indices = np.arange(len(data))
        n = len(data)
        params = self._hyperparameter
        
        # 1. Determinar k (max_outliers)
        max_outliers_param = params["max_outliers"]
        if isinstance(max_outliers_param, float):
            k = int(n * max_outliers_param)
        else:
            k = int(max_outliers_param)
            
        # Protecciones
        k = max(1, min(k, n - 2)) # Al menos 1, y dejar al menos 2 puntos para calcular std
        
        alpha = float(params["alpha"])
        
        # Listas para almacenar los índices eliminados en cada paso
        removed_indices = []
        test_statistics = []
        critical_values = []
        
        # Copia de trabajo que iremos reduciendo
        current_data = data.copy()
        current_indices = original_indices.copy()
        
        logger = logging.getLogger(__name__)
        # 2. Bucle GESD
        for i in range(1, k + 1):
            if len(current_data) < 3:
                break
                
            mean_val = np.mean(current_data)
            std_val = np.std(current_data, ddof=1)
            
            if std_val == 0:
                break
                
            # Calcular desviaciones absolutas estandarizadas
            residuals = np.abs(current_data - mean_val)
            max_res_idx_loc = np.argmax(residuals)
            R = residuals[max_res_idx_loc] / std_val
            
            # Guardar estadísticas
            test_statistics.append(R)
            
            # Calcular Valor Crítico (Lambda)
            # Grados de libertad
            n_curr = len(current_data)
            df = n_curr - 2
            
            # t-percentil
            p = 1 - alpha / (2 * n_curr)
            t_crit = stats.t.ppf(p, df)
            
            # Fórmula de Lambda (Valor Crítico de Rosner)
            lambda_val = ((n_curr - 1) * t_crit) / np.sqrt((n_curr - 2 + t_crit**2) * n_curr)
            critical_values.append(lambda_val)
            
            # Eliminar el outlier de los datos actuales para la siguiente iteración
            # Guardamos el índice ORIGINAL del dato eliminado
            idx_removed = current_indices[max_res_idx_loc]
            removed_indices.append(idx_removed)

            # Borrar de los arrays de trabajo
            current_data = np.delete(current_data, max_res_idx_loc)
            current_indices = np.delete(current_indices, max_res_idx_loc)

            if bool(params.get('verbose', False)):
                logger.info(f"GESD iter {i}: removed idx={int(idx_removed)}, R={float(R):.4f}, lambda={float(lambda_val):.4f}")

        # 3. Determinar el número real de outliers
        # Buscamos el k más grande tal que R_k > Lambda_k
        num_anomalies = 0
        for i in range(len(test_statistics)):
            if test_statistics[i] > critical_values[i]:
                num_anomalies = i + 1
                
        # 4. Generar etiquetas
        labels = np.zeros(n, dtype=int)
        if num_anomalies > 0:
            # Los primeros 'num_anomalies' índices en removed_indices son los outliers reales
            final_outlier_indices = removed_indices[:num_anomalies]
            labels[final_outlier_indices] = 1
            if bool(params.get('verbose', False)):
                logger.info(f"GESD: detected {len(final_outlier_indices)} anomalies at indices {final_outlier_indices}")

        return labels