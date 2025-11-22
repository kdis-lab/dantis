"""
SSA (Singular Spectrum Analysis) implementation for DANTIS.

Este algoritmo descompone una serie temporal en componentes principales (Tendencia,
Estacionalidad, Ruido) utilizando descomposición de valores singulares (SVD) sobre
una matriz de trayectoria (Matriz de Hankel).

Las anomalías se detectan calculando la diferencia (residuo) entre la serie original
y la reconstrucción utilizando los primeros `n_components`.

Algoritmo:
1. Embedding: Convertir la serie 1D en una matriz de trayectoria (Windowing).
2. SVD: Descomponer la matriz.
3. Grouping: Seleccionar los k primeros componentes (la señal).
4. Diagonal Averaging: Reconstruir la serie temporal desde la matriz reducida.
5. Scoring: Anomalía = |Original - Reconstruido|.
"""
import numpy as np
import pandas as pd
import scipy.linalg
from typing import Optional, Dict, Any
import joblib

# Importaciones relativas a dantis (asumiendo estructura del proyecto)
from .. import algorithmbase
from .. import utils


class SSA(algorithmbase.AlgorithmBase):
    """
    Singular Spectrum Analysis (SSA) para detección de anomalías.
    
    Hereda de AlgorithmBase y utiliza ParameterManagement para la gestión
    robusta de hiperparámetros.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        """
        Retorna los hiperparámetros por defecto.
        
        - window_size: Tamaño de la ventana de embedding (L). Si es float (0.0-0.5),
          se toma como fracción de la longitud de la serie. Si es int, es absoluto.
          Por defecto 10.
        - n_components: Número de valores singulares a retener para la reconstrucción.
          Estos representan la "señal" (Tendencia + Estacionalidad). El resto se asume ruido.
        - contamination: Proporción de anomalías esperadas para el método predict().
        """
        return {
            "window_size": 10,
            "n_components": 2,
            "contamination": 0.1,
            "verbose": False
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        """
        Inicializa el detector SSA.
        """
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        
        # Atributos internos del modelo
        self.singular_values_ = None  # Para inspección posterior
        self.reconstructed_series_ = None # Última reconstrucción

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        """
        Configura los hiperparámetros utilizando ParameterManagement para validación y completado.
        """
        defaults = self.get_default_hyperparameters()
        
        if hyperparameter is None:
            self._hyperparameter = defaults.copy()
            return

        # 1. Definir template para introspección de tipos
        def _hp_template(window_size=10, n_components=2, contamination=0.1, verbose=False):
            return None

        # 2. Usar ParameterManagement
        pm = utils.ParameterManagement(_hp_template)
        
        try:
            # Coerción de tipos segura
            coerced = pm.check_hyperparameter_type(hyperparameter)
        except Exception as e:
            raise ValueError(f"SSA Hyperparameter Error: {e}")

        # 3. Completar con defaults
        merged = pm.complete_parameters(coerced)
        
        # 4. Preservar llaves extras si el usuario las envió
        for k, v in hyperparameter.items():
            if k not in merged:
                merged[k] = v

        self._hyperparameter = merged

    def _validate_input(self, x) -> np.ndarray:
        """Convierte entrada a array numpy 1D float."""
        if isinstance(x, pd.DataFrame):
            # Asumimos la segunda columna si hay 2 (timestamp, value) como en s_h_esd
            if x.shape[1] >= 2:
                val = x.iloc[:, 1].values
            else:
                val = x.iloc[:, 0].values
        elif isinstance(x, pd.Series):
            val = x.values
        else:
            val = np.array(x)
        
        # Aplanar si es necesario
        if val.ndim > 1:
            val = val.flatten()
            
        return val.astype(float)

    def _embed(self, series: np.ndarray, window_size: int) -> np.ndarray:
        """
        Paso 1: Embedding. Crea la matriz de trayectoria (Hankel Matrix).
        Retorna matriz de forma (L, K) donde K = N - L + 1.
        """
        N = len(series)
        K = N - window_size + 1
        if K <= 0:
            raise ValueError(f"La serie es demasiado corta ({N}) para el window_size ({window_size})")
        
        # Enforce sliding_window_view: numpy is a dependency of la librería
        try:
            from numpy.lib.stride_tricks import sliding_window_view
        except Exception as e:
            raise ImportError("`sliding_window_view` is required from numpy >=1.20.0") from e

        X = sliding_window_view(series, window_shape=window_size).T
        return X

    def _diagonal_averaging(self, Matrix: np.ndarray, N: int) -> np.ndarray:
        """
        Paso 4: Diagonal Averaging (Hankelization).
        Reconvierte la matriz reconstruida a una serie temporal 1D.
        """
        L, K = Matrix.shape
        # L debe ser min(L, K) para la lógica estándar, transponemos si hace falta
        if L > K:
            Matrix = Matrix.T
            L, K = Matrix.shape
            
        reconstructed = np.zeros(N)
        
        # Promedio diagonal
        # Hay tres zonas en la matriz para el promedio:
        # 1. Triangular creciente
        # 2. Zona central constante
        # 3. Triangular decreciente
        
        # Diagonal averaging (Hankelization) - aggregate anti-diagonals
        count = np.zeros(N, dtype=float)
        res = np.zeros(N, dtype=float)

        for i in range(L):
            for j in range(K):
                res[i + j] += Matrix[i, j]
                count[i + j] += 1

        # Avoid division by zero
        mask = count == 0
        if np.any(mask):
            count[mask] = 1.0

        return res / count

    def _run_ssa(self, series: np.ndarray) -> np.ndarray:
        """
        Ejecuta el pipeline completo de SSA y devuelve la serie reconstruida.
        """
        N = len(series)
        params = self._hyperparameter
        
        # Determinar Window Size
        L = params["window_size"]
        if isinstance(L, float):
            if 0 < L < 1:
                L = int(N * L)
            else:
                L = int(L)
        
        # Restricciones L
        if L >= N // 2:
            # L no debe ser mayor que N/2 para asegurar simetría y separabilidad
            L = min(L, N // 2)
        if L < 2:
            L = 2
            
        # Determinar Componentes
        n_comps = int(params["n_components"])
        n_comps = min(n_comps, L)

        # 1. Embedding
        X = self._embed(series, L)
        
        # 2. SVD (Singular Value Decomposition)
        # X = U * Sigma * V.T
        # Usamos scipy.linalg.svd que es robusto
        U, Sigma, VT = scipy.linalg.svd(X, full_matrices=False)
        
        self.singular_values_ = Sigma # Guardar para debug

        # 3. Grouping (Reconstrucción de matriz elemental)
        # Solo usamos los primeros n_comps componentes
        # X_rec = Sum(sigma_i * u_i * v_i.T)
        
        X_rec = np.zeros_like(X)
        for i in range(n_comps):
            # Producto exterior de u_i y v_i
            elem = Sigma[i] * np.outer(U[:, i], VT[i, :])
            X_rec += elem
            
        # 4. Diagonal Averaging (Reconstrucción serie)
        reconstructed_series = self._diagonal_averaging(X_rec, N)
        
        return reconstructed_series

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        Para SSA, el ajuste puede consistir en aprender qué es "normal" analizando
        el error de reconstrucción sobre el conjunto de entrenamiento.
        """
        series = self._validate_input(x_train)
        
        # Ejecutar SSA
        rec = self._run_ssa(series)
        self.reconstructed_series_ = rec

        # Calcular residuos (error absoluto como score)
        residuals = np.abs(series - rec)
        self._last_scores = residuals

        # Mantener compatibilidad con la interfaz usada por otros wrappers:
        # retornamos los scores (array 1D)
        return residuals

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Retorna el score de anomalía.
        Score = Diferencia absoluta (o cuadrática) entre original y reconstrucción.
        Cuanto mayor el residuo, más anómalo es el punto (no encaja en tendencia/estacionalidad).
        """
        series = self._validate_input(x)
        
        # Ejecutamos SSA sobre la entrada actual
        # Nota: SSA es computacionalmente intensivo O(N^3) en el peor caso de SVD, 
        # pero eficiente para series < 10k puntos.
        rec = self._run_ssa(series)
        self.reconstructed_series_ = rec
        
        # Score = Error absoluto
        scores = np.abs(series - rec)
        
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Retorna predicción binaria (0/1) basada en el parámetro de contaminación.
        """
        scores = self.decision_function(x)
        
        contamination = float(self._hyperparameter.get("contamination", 0.1))
        # Clamp contamination
        contamination = max(0.001, min(0.5, contamination))
        
        # Calcular umbral
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        
        labels = (scores > threshold).astype(int)
        return labels

    def save_model(self, path_model: str):
        """Save SSA fitted state to a joblib file.

        The file will contain a dict with keys: `singular_values_`,
        `reconstructed_series_`, `_hyperparameter` and `_model`.
        """
        payload = {
            'singular_values_': self.singular_values_,
            'reconstructed_series_': self.reconstructed_series_,
            '_hyperparameter': self._hyperparameter,
            '_model': getattr(self, '_model', None)
        }
        if not path_model.endswith('.joblib'):
            path_model = path_model + '.joblib'
        joblib.dump(payload, path_model)

    def load_model(self, path_model: str):
        """Load SSA fitted state from a joblib file created by `save_model`.

        Restores `singular_values_`, `reconstructed_series_`, `_hyperparameter`
        and `_model`.
        """
        if not path_model.endswith('.joblib'):
            raise ValueError('path_model must be a .joblib file')
        payload = joblib.load(path_model)
        self.singular_values_ = payload.get('singular_values_')
        self.reconstructed_series_ = payload.get('reconstructed_series_')
        self._hyperparameter = payload.get('_hyperparameter', self.get_default_hyperparameters())
        self._model = payload.get('_model', None)