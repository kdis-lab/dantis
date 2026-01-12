"""
Robust PCA wrapper para DANTIS.

Implementa una versión sencilla de RPCA (Principal Component Pursuit) usando
el método Inexact Augmented Lagrange Multiplier (IALM). Para series 1D
construimos una matriz Hankel (embedding) y aplicamos RPCA sobre ella.
La componente dispersa reconstruida se promedia por diagonales (diagonal
averaging) para producir scores por muestra.

Notas:
- Esta implementación es autocontenida (numpy + scipy si disponible).
- Parámetros importantes: `window_size`, `lambda_`, `max_iter`, `tol`.
"""
from typing import Optional, Dict, Any, Tuple
import numpy as np
import math
import warnings
try:
    import joblib
except Exception:
    joblib = None

from .. import algorithmbase
from .. import utils


def _hankel_matrix(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    if window < 1 or window >= n:
        raise ValueError("window must be >=1 and < len(x)")
    cols = n - window + 1
    H = np.empty((window, cols), dtype=float)
    for i in range(window):
        H[i, :] = x[i:i+cols]
    return H


def _diagonal_average(mat: np.ndarray, original_length: int) -> np.ndarray:
    # Reconstruct 1D series from Hankel-like matrix via averaging anti-diagonals
    window, cols = mat.shape
    n = original_length
    out = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)
    # for each element mat[i,j] correspond to series index i+j
    for i in range(window):
        for j in range(cols):
            idx = i + j
            out[idx] += mat[i, j]
            counts[idx] += 1.0
    counts[counts == 0] = 1.0
    return out / counts


def _svd_shrink(M: np.ndarray, tau: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_shrink = np.maximum(s - tau, 0.0)
    return (U * s_shrink) @ Vt


def rpca_ialm(D: np.ndarray, lam: Optional[float] = None, mu: Optional[float] = None,
              max_iter: int = 1000, tol: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust PCA via IALM (simple implementation).

    Returns (L, S) low-rank and sparse components.
    """
    m, n = D.shape
    norm_D = np.linalg.norm(D, 'fro')
    if lam is None:
        lam = 1.0 / math.sqrt(max(m, n))

    # Initialization
    S = np.zeros_like(D)
    L = np.zeros_like(D)
    Y = D.copy() / max(np.linalg.norm(D, 2), 1e-8)

    if mu is None:
        mu = (m * n) / (4.0 * np.sum(np.abs(D)) + 1e-8)

    rho = 1.5
    for it in range(int(max_iter)):
        # update L via singular value thresholding
        temp = D - S + (1.0 / mu) * Y
        L = _svd_shrink(temp, 1.0 / mu)

        # update S via soft-threshold
        temp2 = D - L + (1.0 / mu) * Y
        S = np.sign(temp2) * np.maximum(np.abs(temp2) - lam / mu, 0.0)

        # residual
        Z = D - L - S
        err = np.linalg.norm(Z, 'fro') / (norm_D + 1e-12)
        if err < tol:
            break

        Y = Y + mu * Z
        mu = min(mu * rho, 1e7)

    return L, S


class RobustPCA(algorithmbase.AlgorithmBase):
    """Robust PCA detector que aplica RPCA a una matriz Hankel de la serie."""

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "window_size": 50,
            "lambda_": None,
            "max_iter": 500,
            "tol": 1e-6,
            "contamination": 0.1,
            "verbose": False,
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

        def _hp_template(window_size=50, lambda_=None, max_iter=500, tol=1e-6, contamination=0.1, verbose=False):
            return None

        pm = utils.ParameterManagement(_hp_template)
        try:
            coerced = pm.check_hyperparameter_type(hyperparameter)
        except Exception as e:
            raise ValueError(f"RobustPCA Hyperparameter Error: {e}")

        merged = pm.complete_parameters(coerced)
        for k, v in hyperparameter.items():
            if k not in merged:
                merged[k] = v
        self._hyperparameter = merged

    def _validate_input(self, x) -> np.ndarray:
        if isinstance(x, np.ndarray):
            arr = x.flatten().astype(float)
        else:
            try:
                import pandas as pd
                if isinstance(x, pd.Series):
                    arr = x.values.flatten().astype(float)
                elif isinstance(x, pd.DataFrame):
                    arr = x.iloc[:, 0].values.flatten().astype(float)
                else:
                    arr = np.array(x).flatten().astype(float)
            except Exception:
                arr = np.array(x).flatten().astype(float)
        return arr

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        series = self._validate_input(x_train)
        n = len(series)
        params = self._hyperparameter
        window = int(params.get('window_size', 50))
        if window >= n:
            window = max(1, n // 2)

        H = _hankel_matrix(series, window)
        L, S = rpca_ialm(H, lam=params.get('lambda_'), max_iter=int(params.get('max_iter', 500)), tol=float(params.get('tol', 1e-6)))

        sparse_series = _diagonal_average(S, n)
        self._last_scores = np.abs(sparse_series)
        self._model = {
            'window': window,
            'L': L,
            'S': S,
            'train_length': n,
        }
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        series = self._validate_input(x)
        n = len(series)
        params = self._hyperparameter

        # Reuse cached if lengths match
        if self._last_scores is not None and self._model is not None and self._model.get('train_length') == n:
            return self._last_scores

        window = int(params.get('window_size', 50))
        if window >= n:
            window = max(1, n // 2)

        H = _hankel_matrix(series, window)
        L, S = rpca_ialm(H, lam=params.get('lambda_'), max_iter=int(params.get('max_iter', 500)), tol=float(params.get('tol', 1e-6)))
        sparse_series = _diagonal_average(S, n)
        scores = np.abs(sparse_series)
        self._last_scores = scores
        self._model = {'window': window, 'L': L, 'S': S, 'train_length': n}
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        contamination = float(self._hyperparameter.get('contamination', 0.1))
        contamination = max(0.001, min(0.5, contamination))
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        return (scores > threshold).astype(int)

    def save_model(self, path: str):
        if joblib is None:
            raise ImportError("Guardar modelo requiere joblib. Instala con: pip install joblib")
        payload = {
            'hyperparameter': self._hyperparameter,
            'model': self._model,
            'last_scores': self._last_scores,
        }
        joblib.dump(payload, path)

    def load_model(self, path: str):
        if joblib is None:
            raise ImportError("Cargar modelo requiere joblib. Instala con: pip install joblib")
        payload = joblib.load(path)
        self._hyperparameter = payload.get('hyperparameter', self.get_default_hyperparameters())
        self._model = payload.get('model')
        self._last_scores = payload.get('last_scores')
        return self
