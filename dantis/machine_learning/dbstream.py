import numpy as np
import logging

from .. import algorithmbase

try:
    from river import cluster
except Exception as e:
    raise ImportError("Instala la dependencia: pip install river>=0.21") from e

logger = logging.getLogger(__name__)


class DBStreamAD(algorithmbase.AlgorithmBase):
    """DBSTREAM anomaly detector (River) con interfaz de init por argumentos.

    Parameters
    ----------
    window_size : int, default=20
        Longitud de ventana deslizante. Pon 1 para desactivar ventaneo.
    normalize : bool, default=True
        Min–max por feature a [0, 1] antes de ventanear.
    clustering_threshold : float, default=1.0
        Radio `r` de DBSTREAM (River: `clustering_threshold`).
    fading_factor : float, default=0.01
        Decaimiento temporal λ (River: `fading_factor`).
    cleanup_interval : float, default=2.0
        Intervalo de limpieza (River: `cleanup_interval`).
    intersection_factor : float, default=0.3
        Factor de intersección α (River: `intersection_factor`).
    minimum_weight : float, default=1.0
        Peso mínimo para considerar micro-clusters (River: `minimum_weight`).
    score_mode : {"cluster_frequency","macro_weight"}, default="cluster_frequency"
        Estrategia de scoring de subsecuencia. Por defecto usa frecuencia
        del cluster (robusto). "macro_weight" intentará usar pesos si tu
        build de River los expone; si no, cae a frecuencia.
    random_state : int, default=42
        Semilla para normalización (no afecta a River).

    Notes
    -----
    - `predict(X)` devuelve **scores** (no etiquetas binarias).
    - Ajusta tu umbral/percentil aguas arriba si quieres 0/1.
    """

    _model = None
    _fitted = False

    def __init__(
        self,
        window_size: int = 20,
        normalize: bool = True,
        clustering_threshold: float = 1.0,
        fading_factor: float = 0.01,
        cleanup_interval: float = 2.0,
        intersection_factor: float = 0.3,
        minimum_weight: float = 1.0,
        score_mode: str = "cluster_frequency",
        random_state: int = 42,
    ) -> None:
        hp = dict(
            window_size=window_size,
            normalize=normalize,
            clustering_threshold=clustering_threshold,
            fading_factor=fading_factor,
            cleanup_interval=cleanup_interval,
            intersection_factor=intersection_factor,
            minimum_weight=minimum_weight,
            score_mode=score_mode,
            random_state=random_state,
        )
        super().__init__(hp)

        # Estado interno
        self._x_min = None
        self._x_max = None
        self._cluster_counts_ = None
        self._max_count_ = 0

    # ------------------------------------------------------------
    # Requisitos de AlgorithmBase
    # ------------------------------------------------------------
    def fit(self, x_train: np.ndarray, y_train: np.ndarray | None = None):
        X = self._validate_X(x_train)
        Xn = self._maybe_normalize_fit_transform(X)
        Z = self._to_windows(Xn)

        self._create_model()

        labels = []
        for row in Z:
            x_dict = {i: float(v) for i, v in enumerate(row)}
            self._model.learn_one(x_dict)
            lbl = self._model.predict_one(x_dict)
            labels.append(int(lbl))

        if len(labels):
            max_lbl = max(labels)
            counts = np.bincount(labels, minlength=max_lbl + 1)
            self._cluster_counts_ = counts
            self._max_count_ = int(counts.max()) if counts.size else 0
        else:
            self._cluster_counts_ = np.array([], dtype=int)
            self._max_count_ = 0

        self._fitted = True
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_X(x)
        Xn = self._maybe_normalize_transform(X)
        Z = self._to_windows(Xn)

        z_scores = np.array([self._score_window(row) for row in Z], dtype=float)

        n = len(Xn)
        w = int(self.get_hyperparameter()["window_size"])
        if w <= 1:
            # score 1:1 con las muestras
            # si no ventaneas, Z==X, ya están alineados
            return z_scores

        # promedio de las ventanas que cubren cada punto (como en el R)
        point_scores = np.zeros(n, dtype=float)
        counts = np.zeros(n, dtype=float)
        for i, s in enumerate(z_scores):
            point_scores[i : i + w] += s
            counts[i : i + w] += 1.0
        counts[counts == 0] = 1.0
        return point_scores / counts

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        return self.decision_function(x)

    # ------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------
    def _create_model(self):
        hp = self.get_hyperparameter()
        self._model = cluster.DBSTREAM(
            clustering_threshold=float(hp["clustering_threshold"]),
            fading_factor=float(hp["fading_factor"]),
            cleanup_interval=float(hp["cleanup_interval"]),
            intersection_factor=float(hp["intersection_factor"]),
            minimum_weight=float(hp["minimum_weight"]),
        )
        return self._model

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("x must be 2D: (n_samples, n_features)")
        return X.astype(float, copy=False)

    def _maybe_normalize_fit_transform(self, X: np.ndarray) -> np.ndarray:
        if not bool(self.get_hyperparameter()["normalize"]):
            return np.array(X, copy=True)
        self._x_min = X.min(axis=0)
        self._x_max = X.max(axis=0)
        span = np.where(self._x_max > self._x_min, self._x_max - self._x_min, 1.0)
        return (X - self._x_min) / span

    def _maybe_normalize_transform(self, X: np.ndarray) -> np.ndarray:
        if not bool(self.get_hyperparameter()["normalize"]):
            return np.array(X, copy=True)
        if self._x_min is None or self._x_max is None:
            raise RuntimeError("Normalizer is not fitted. Call fit() first.")
        span = np.where(self._x_max > self._x_min, self._x_max - self._x_min, 1.0)
        return (X - self._x_min) / span

    def _to_windows(self, X: np.ndarray) -> np.ndarray:
        w = int(self.get_hyperparameter()["window_size"])
        n, d = X.shape
        if w <= 1:
            return X.copy()
        if w > n:
            raise ValueError(f"window_size={w} > n_samples={n}")
        out = np.empty((n - w + 1, w * d), dtype=float)
        for i in range(n - w + 1):
            out[i] = X[i : i + w].reshape(-1)
        return out

    def _score_window(self, row_vec: np.ndarray) -> float:
        hp = self.get_hyperparameter()
        x_dict = {i: float(v) for i, v in enumerate(row_vec)}
        lbl = int(self._model.predict_one(x_dict))
        mode = str(hp["score_mode"]).lower()

        if mode == "macro_weight":
            w = self._try_macro_weight(lbl)
            if w is not None:
                # score inverso al peso, normalizado de forma sencilla
                return float(max(1.0, (self._max_count_ + 1) - w))

        # cluster_frequency (por defecto): menos soporte ⇒ más anomalía
        if self._cluster_counts_ is None or lbl >= len(self._cluster_counts_):
            return float(self._max_count_ + 1)
        count = int(self._cluster_counts_[lbl])
        return float((self._max_count_ - count) + 1)

    def _try_macro_weight(self, lbl: int) -> float | None:
        try:
            C = getattr(self._model, "clusters", None)
            if C is None:
                return None
            if isinstance(C, (list, tuple)):
                if 0 <= lbl < len(C):
                    w = getattr(C[lbl], "weight", None)
                    return float(w) if w is not None else None
            if isinstance(C, dict) and lbl in C:
                w = getattr(C[lbl], "weight", None)
                return float(w) if w is not None else None
            return None
        except Exception:
            return None

    def _check_is_fitted(self):
        if not self._fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
