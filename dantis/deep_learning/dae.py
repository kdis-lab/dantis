import os
import shutil
import numpy as np
from dataclasses import asdict

from .. import algorithmbase
from .. import utils

try:
	import tensorflow as tf
	from tensorflow import keras
	from keras import Input
	from keras.layers import Dense
except Exception:
	tf = None
	keras = None


def _archive_model_dir(tmp_dir: str, model_path: str):
	"""Create a zip archive from `tmp_dir` and move it to `model_path`.

	Parameters
	----------
	tmp_dir : str
		Directory containing the saved Keras model (usually 'check' temporary dir).
	model_path : str
		Target path (without extension) where the archive will be stored. The
		final file will be `model_path.zip`.
	"""
	shutil.make_archive(model_path, 'zip', tmp_dir)
	# ensure final artifact exists


class DAE(algorithmbase.AlgorithmBase):
	"""
	Denoising Autoencoder wrapper implementing AlgorithmBase using Keras.

	This class adapts the provided author implementation into the project's
	`AlgorithmBase` interface. It trains a simple autoencoder that reconstructs
	the input and uses reconstruction error as anomaly score.

	Parameters
	----------
	hyperparameter : dict
		Hyperparameters for the AutoEn model. Recognized keys include:
		- latent_size: int, size of latent (encoder) layer
		- epochs: int, training epochs
		- learning_rate: float
		- noise_ratio: float, fraction of samples to zero-out as noise
		- split: float, training split fraction
		- early_stopping_delta: float
		- early_stopping_patience: int

	Notes
	-----
	Model saving: the Keras model is saved to a temporary folder and then
	archived as a zip file for portability. Loading expects a zip archive and
	extracts it to a temporary folder before restoring the Keras model.
	"""

	def __init__(self, hyperparameter: dict = None):
		pm = utils.ParameterManagement(self._author_autoen_init)
		defaults = pm.complete_parameters({} if hyperparameter is None else hyperparameter)
		super().__init__(defaults)
		self._model = None

	@staticmethod
	def _author_autoen_init(latent_size=32, epochs=10, learning_rate=0.005, noise_ratio=0.1,
							early_stopping_patience: int = 10, early_stopping_delta: float = 1e-2,
							split: float = 0.8, **kwargs):
		"""Signature helper matching the original author's AutoEn init.

		This is used only by ParameterManagement to extract defaults.
		"""
		return None

	def _build_autoencoder(self, features: int, latent_size: int):
		inp = Input(shape=(features,))
		fc = Dense(latent_size)(inp)
		d1 = Dense(features)(fc)
		model = tf.keras.Model(inputs=inp, outputs=d1)
		return model

	def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
		"""
		Train the denoising autoencoder on non-anomalous data.

		Parameters
		----------
		x_train : np.ndarray
			Training data shape (n_samples, n_features). If `y_train` is
			provided, only samples with label 0 (normal) will be used for
			training (to follow original author's filtering).
		y_train : np.ndarray, optional
			Array of labels (0 normal, 1 anomaly). If provided, only normal
			rows are used for training.

		Returns
		-------
		self
		"""
		if tf is None or keras is None:
			raise RuntimeError("TensorFlow/Keras not available in the environment")

		params = {} if self._hyperparameter is None else self._hyperparameter
		# prepare data
		X = np.asarray(x_train)
		if y_train is not None:
			mask = np.asarray(y_train).reshape(-1) == 0
			X = X[mask]

		features = X.shape[1]
		latent_size = params.get('latent_size', 32)
		epochs = params.get('epochs', 10)
		lr = params.get('learning_rate', 0.005)
		noise_ratio = params.get('noise_ratio', 0.1)
		validation_split = 1 - params.get('split', 0.8)
		early_patience = params.get('early_stopping_patience', 10)
		early_delta = params.get('early_stopping_delta', 1e-2)

		# add noise by zeroing some rows
		noise = int(X.shape[0] * noise_ratio)
		ii = np.random.permutation(X.shape[0])[:noise] if noise > 0 else np.array([], dtype=int)
		X_noisy = X.copy()
		if ii.size > 0:
			X_noisy[ii] = 0

		# build model
		model = self._build_autoencoder(features, latent_size)
		opt = keras.optimizers.Adam(learning_rate=lr)
		model.compile(optimizer=opt, loss='mse')

		# callbacks: early stopping and checkpoint to temp dir
		tmp_dir = 'dae_check'
		if os.path.exists(tmp_dir):
			shutil.rmtree(tmp_dir)
		os.makedirs(tmp_dir, exist_ok=True)

		checkpoint_path = os.path.join(tmp_dir, 'model')
		callbacks = [
			tf.keras.callbacks.EarlyStopping(patience=early_patience, min_delta=early_delta),
			tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=False),
			tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: _archive_model_dir(tmp_dir, 'dae_model') if os.path.exists(tmp_dir) else None)
		]

		model.fit(X_noisy, X, epochs=epochs, validation_split=validation_split, callbacks=callbacks)

		# save best model to temp and archive
		model.save(tmp_dir)
		_archive_model_dir(tmp_dir, 'dae_model')

		# store model in-memory
		self._model = model
		return self

	def decision_function(self, x: np.ndarray) -> np.ndarray:
		"""
		Compute reconstruction error per sample as anomaly score (higher -> more anomalous).

		Parameters
		----------
		x : np.ndarray
			Input data of shape (n_samples, n_features).

		Returns
		-------
		scores : np.ndarray
			One-dimensional array with reconstruction error per sample.
		"""
		if self._model is None:
			raise Exception("Model not trained or loaded")
		X = np.asarray(x)
		recon = self._model.predict(X)
		scores = np.mean(np.abs(recon - X), axis=1)
		return scores

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""
		Return binary predictions applying a simple percentile-based threshold (95th percentile).

		Parameters
		----------
		x : np.ndarray
			Input data of shape (n_samples, n_features).

		Returns
		-------
		preds : np.ndarray
			Binary array (1 anomaly, 0 normal).
		"""
		scores = self.decision_function(x)
		thresh = np.percentile(scores, 95)
		return (scores >= thresh).astype(int)

	def save_model(self, path_model: str):
		"""Save Keras model to a zip archive at `path_model` (without extension or with .zip)."""
		if self._model is None:
			raise Exception("No model to save")
		tmp_dir = 'dae_saved_tmp'
		if os.path.exists(tmp_dir):
			shutil.rmtree(tmp_dir)
		os.makedirs(tmp_dir, exist_ok=True)
		self._model.save(tmp_dir)
		shutil.make_archive(path_model, 'zip', tmp_dir)

	def load_model(self, path_model: str):
		"""Load a Keras model from a zip archive located at `path_model` (with or without .zip)."""
		if not path_model.endswith('.zip'):
			zip_path = path_model + '.zip'
		else:
			zip_path = path_model
		tmp_dir = 'dae_load_tmp'
		if os.path.exists(tmp_dir):
			shutil.rmtree(tmp_dir)
		os.makedirs(tmp_dir, exist_ok=True)
		shutil.unpack_archive(zip_path, tmp_dir, 'zip')
		# load the model directory inside tmp_dir (Keras saves model directly into the folder)
		# find first subdirectory or use tmp_dir
		candidates = [os.path.join(tmp_dir, p) for p in os.listdir(tmp_dir)]
		model_dir = candidates[0] if candidates else tmp_dir
		self._model = keras.models.load_model(model_dir)


def set_random_state(config) -> None:
	"""
	Set random seeds for reproducibility using the provided `AlgorithmArgs` object.

	Parameters
	----------
	config : AlgorithmArgs
		Object holding `customParameters.random_state` attribute.
	"""
	seed = config.customParameters.random_state
	import random
	random.seed(seed)
	np.random.seed(seed)
	if tf is not None:
		tf.random.set_seed(seed)


if __name__ == '__main__':
    # Minimal unit test to validate wrapping without heavy training. This test
    # creates a tiny autoencoder, assigns it to the wrapper and checks outputs.
    def _unit_test_dae():
        import numpy as _np
        X = _np.random.rand(200, 10).astype(np.float32)
        dae = DAE({})
        # build a tiny keras model and attach directly
        if keras is None:
            print('Keras not available; skipping unit test')
            return
        model = dae._build_autoencoder(features=10, latent_size=4)
        dae._model = model
        scores = dae.decision_function(X)
        preds = dae.predict(X)
        assert scores.shape[0] == X.shape[0]
        assert preds.shape[0] == X.shape[0]
        print('DAE unit test passed')

    _unit_test_dae()
