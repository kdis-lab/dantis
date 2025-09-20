from .. import algorithmbase
from .. import utils
from .wrapper_bagel.model import DonutX
from .wrapper_bagel.kpi_series import KPISeries


import numpy as np
import joblib
import logging


class Bagel(algorithmbase.AlgorithmBase):
	"""
	Adaptador para el algoritmo Bagel (DonutX) que sigue la interfaz AlgorithmBase.

	Implementa `fit`, `decision_function` y `predict` delegando en `wrapper_bagel.DonutX`.
	"""

	def __init__(self, hyperparameter: dict = None):
		# completar y validar hiperparámetros con ParameterManagement
		pm = utils.ParameterManagement(DonutX.__init__)
		hyperparameter = pm.check_hyperparameter_type(hyperparameter)
		defaults = pm.complete_parameters({} if hyperparameter is None else hyperparameter)
		super().__init__(defaults)
		self.check_hyperparams = True
		self._model = None

	def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
		"""
		Entrena el modelo Bagel usando los datos en formato numpy arrays.

		x_train expected shape: (n_samples, ) or (n_samples, 1)
		y_train optional: etiquetas binarias para usar como labels (same length)
		"""
		# convertir a KPISeries que espera wrapper_bagel

		x = np.asarray(x_train).reshape(-1)
		# generar timestamps simples (sec) si no hay
		timestamps = np.arange(len(x))
		labels = y_train if y_train is not None else np.zeros_like(x, dtype=int)

		kpi = KPISeries(value=x, timestamp=timestamps, label=labels)

		# crear instancia del modelo si no existe
		params = {} if self._hyperparameter is None else self._hyperparameter
		# mapear nombres simples, DonutX espera argumentos como inits
		model = DonutX(
			max_epoch=params.get('max_epoch', params.get('epochs', 50)),
			batch_size=params.get('batch_size', 128),
			network_size=params.get('network_size', params.get('hidden_layer_shape', [100, 100])),
			latent_dims=params.get('latent_dims', params.get('latent_size', 8)),
			window_size=params.get('window_size', 120),
			cuda=params.get('cuda', False),
			condition_dropout_left_rate=params.get('condition_dropout_left_rate', 1 - params.get('dropout', 0.1)),
			early_stopping_patience=params.get('early_stopping_patience', params.get('early_stopping_patience', 10)),
			early_stopping_delta=params.get('early_stopping_delta', params.get('early_stopping_delta', 0.05)),
			print_fn=params.get('print_fn', print)
		)

		# split validation according to split param if exists
		split = params.get('split', 0.8)
		split_idx = int(len(kpi.value) * split)
		train_kpi = KPISeries(kpi.value[:split_idx], kpi.timestamp[:split_idx], kpi.label[:split_idx])
		valid_kpi = KPISeries(kpi.value[split_idx:], kpi.timestamp[split_idx:], kpi.label[split_idx:]) if split_idx < len(kpi.value) else None

		# entrenar
		model.fit(train_kpi, valid_kpi=valid_kpi)

		self._model = model
		return self

	def decision_function(self, x: np.ndarray) -> np.ndarray:
		"""
		Devuelve scores continuos (mayor -> más anómalo).
		"""
		if self._model is None:
			raise Exception("Model not trained or loaded")

		x = np.asarray(x).reshape(-1)
		timestamps = np.arange(len(x))
		kpi = KPISeries(value=x, timestamp=timestamps)

		indicator = self._model.predict(kpi)
		return np.asarray(indicator)

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""
		Binariza los scores usando un umbral interno (detect) si está disponible.
		"""
		if self._model is None:
			raise Exception("Model not trained or loaded")

		x = np.asarray(x).reshape(-1)
		timestamps = np.arange(len(x))
		kpi = KPISeries(value=x, timestamp=timestamps)

		try:
			preds = self._model.detect(kpi)
		except Exception:
			# fallback simple: threshold por percentil 95 de decision_function
			scores = self.decision_function(x)
			thresh = np.percentile(scores, 95)
			preds = scores >= thresh

		return np.asarray(preds).astype(int)

	def save_model(self, path_model: str):
		"""Guardar el objeto entrenamiento (modelo wrapper) usando joblib"""
		if self._model is None:
			raise Exception("No model to save")
		joblib.dump(self._model, path_model)

	def load_model(self, path_model: str):
		"""Cargar modelo guardado con joblib"""
		self._model = joblib.load(path_model)

