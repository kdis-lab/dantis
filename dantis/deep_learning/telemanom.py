"""
Telemanom integrated into DANTIS.

This file ports the essential classes from the original Telemanom package
(helpers.Config, Channel, Model, Errors) and exposes a `Telemanom` class
that implements the `AlgorithmBase` interface used by DANTIS.

Notes:
- The original Telemanom uses Keras; this port preserves the training and
  prediction logic but adapts configuration and parameter management to the
  DANTIS conventions (ParameterManagement).
- Many helper functions were copied and minimally adapted to integrate with
  DANTIS (logging, save/load, config by dict).

Limitations:
- This port is faithful to the algorithm structure but expects the runtime
  environment to have TensorFlow/Keras installed to train or load models.
"""
from typing import Optional, Dict, Any, List
import os
import logging
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from keras.models import Sequential, load_model
    from keras.callbacks import History, EarlyStopping, ModelCheckpoint
    from keras.layers.recurrent import LSTM
    from keras.layers.core import Dense, Activation, Dropout
except Exception:
    Sequential = None
    load_model = None
    History = None
    EarlyStopping = None
    ModelCheckpoint = None
    LSTM = None
    Dense = None
    Activation = None
    Dropout = None

try:
    import joblib
except Exception:
    joblib = None

from .. import algorithmbase
from .. import utils


logger = logging.getLogger('telemanom_dantis')
logger.setLevel(logging.INFO)
stdout = logging.StreamHandler()
stdout.setLevel(logging.INFO)
logger.addHandler(stdout)


class Config:
    """Minimal config object compatible with Telemanom original settings.

    This class can be created from a dict using `from_dict` to adapt the
    DANTIS hyperparameter dict into Telemanom settings.
    """
    def __init__(self):
        self.dictionary = {}

    @staticmethod
    def from_dict(d: dict) -> 'Config':
        c = Config()
        c.dictionary = dict(d)
        for k, v in c.dictionary.items():
            setattr(c, k, v)
        return c


class Channel:
    """Port of Telemanom Channel class: shapes data for LSTM ingestion."""
    def __init__(self, config: Config, chan_id: str = 'chan'):
        self.id = chan_id
        self.config = config
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_hat = None
        self.train = None
        self.test = None

    def shape_data(self, arr, train=True):
        # Accept arr with shape (timesteps, n_features) or (timesteps, 1, n_features)
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr2 = arr.reshape(arr.shape[0], arr.shape[2])
        elif arr.ndim == 2:
            arr2 = arr
        else:
            arr2 = arr.reshape(arr.shape[0], -1)

        n_features = arr2.shape[1]
        seq_len = self.config.window_size + self.config.prediction_window_size
        data = []
        for i in range(len(arr2) - seq_len):
            data.append(arr2[i:i + seq_len])
        data = np.array(data)

        if data.size == 0:
            data = np.zeros((0, seq_len, n_features))

        assert data.ndim == 3

        if train:
            if len(data) > 0:
                np.random.shuffle(data)
                self.X_train = data[:, :-self.config.prediction_window_size, :]
                # y_train shape: (samples, prediction_window_size, n_features)
                self.y_train = data[:, -self.config.prediction_window_size:, :]
            else:
                self.X_train = np.empty((0, self.config.window_size, n_features))
                self.y_train = np.empty((0, self.config.prediction_window_size, n_features))
        else:
            if len(data) > 0:
                self.X_test = data[:, :-self.config.prediction_window_size, :]
                self.y_test = data[:, -self.config.prediction_window_size:, :]
            else:
                self.X_test = np.empty((0, self.config.window_size, n_features))
                self.y_test = np.empty((0, self.config.prediction_window_size, n_features))

    def set_data(self, data: np.ndarray, train: bool):
        if train:
            self.train = data
        else:
            self.test = data
        self.shape_data(data, train=train)


class Model:
    """Port of Telemanom Model that trains/predicts an LSTM using Keras."""
    def __init__(self, config: Config, run_id: str, channel: Channel, model_path: Optional[os.PathLike] = None):
        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None
        self.model_path = model_path or os.path.join('data', getattr(self.config, 'use_id', 'run'), 'models', self.chan_id + '.h5')

        if not getattr(self.config, 'train', False):
            try:
                self.load()
            except Exception:
                logger.warning('Training new model, couldn\'t find existing model at {}'.format(self.model_path))
                self.train_new(channel)
                self.save()
        else:
            self.train_new(channel)
            self.save()

    def load(self):
        if load_model is None:
            raise ImportError('Keras not available to load model')
        logger.info('Loading pre-trained model')
        self.model = load_model(self.model_path)

    def train_new(self, channel: Channel):
        if Sequential is None:
            raise ImportError('Keras not available to train model')

        c = self.config

        cbs = [History(),
               EarlyStopping(monitor='val_loss', patience=c.patience, min_delta=c.min_delta, verbose=0),
               ModelCheckpoint(filepath=self.model_path, verbose=0, save_weights_only=False, period=1)
               ]
        self.model = Sequential()

        n_features = int(channel.X_train.shape[2])
        # Dense must output prediction_window_size * n_features values per sample
        out_units = int(c.prediction_window_size * n_features)

        self.model.add(LSTM(c.layers[0], input_shape=(None, n_features), return_sequences=True))
        self.model.add(Dropout(c.dropout))
        self.model.add(LSTM(c.layers[1], return_sequences=False))
        self.model.add(Dropout(c.dropout))

        self.model.add(Dense(out_units))
        self.model.add(Activation('linear'))

        self.model.compile(loss=c.loss_metric, optimizer=c.optimizer)

        # reshape y_train to (samples, out_units)
        y_train = channel.y_train.reshape(channel.y_train.shape[0], out_units)

        self.model.fit(channel.X_train, y_train, batch_size=c.lstm_batch_size, epochs=c.epochs,
                       validation_split=c.validation_split, callbacks=cbs, verbose=1)

    def save(self):
        if self.model is None:
            return
        self.model.save(self.model_path)

    def aggregate_predictions(self, y_hat_batch, method='first'):
        # y_hat_batch expected shape: (batch_len, out_units) where out_units = pred_window * n_features
        batch_len = y_hat_batch.shape[0]
        out_units = y_hat_batch.shape[1]
        pred_window = int(self.config.prediction_window_size)
        n_features = out_units // pred_window

        # reshape to (batch_len, pred_window, n_features)
        y_hat_batch = y_hat_batch.reshape(batch_len, pred_window, n_features)

        # Determine aggregation method: can be 'first', 'mean', 'median', 'max'
        if method is None:
            method = getattr(self.config, 'aggregation', 'mean')

        # For each timestep t in the batch, aggregate diagonal predictions across the sliding preds
        agg_y_hat_batch = []
        for t in range(batch_len):
            start_idx = t - pred_window
            start_idx = start_idx if start_idx >= 0 else 0
            window_preds = y_hat_batch[start_idx:t+1]  # shape (k, pred_window, n_features)
            # collect diagonals per feature
            diag_vals = np.array([np.flipud(window_preds[:, :, f]).diagonal() for f in range(n_features)])
            # diag_vals shape (n_features, k)
            if method == 'first':
                chosen = diag_vals[:, 0]
            elif method == 'mean':
                chosen = diag_vals.mean(axis=1)
            elif method == 'median':
                chosen = np.median(diag_vals, axis=1)
            elif method == 'max':
                chosen = diag_vals.max(axis=1)
            else:
                # fallback to mean
                chosen = diag_vals.mean(axis=1)
            agg_y_hat_batch.append(chosen)

        agg_y_hat_batch = np.array(agg_y_hat_batch)  # shape (batch_len, n_features)

        if self.y_hat.size == 0:
            self.y_hat = agg_y_hat_batch
        else:
            self.y_hat = np.vstack([self.y_hat, agg_y_hat_batch])

    def batch_predict(self, channel: Channel, save_y_hat: bool = True):
        num_batches = int((channel.y_test.shape[0] - self.config.window_size) / self.config.batch_size)
        if num_batches < 0:
            raise ValueError('l_s ({}) too large for stream length {}.'.format(self.config.window_size, channel.y_test.shape[0]))

        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size
            if i + 1 == num_batches + 1:
                idx = channel.y_test.shape[0]

            X_test_batch = channel.X_test[prior_idx:idx]
            y_hat_batch = self.model.predict(X_test_batch)
            self.aggregate_predictions(y_hat_batch)

        # channel.y_hat is (n_timesteps, n_features)
        channel.y_hat = self.y_hat

        if save_y_hat:
            os.makedirs(os.path.join('data', self.run_id, 'y_hat'), exist_ok=True)
            np.save(os.path.join('data', self.run_id, 'y_hat', '{}.npy'.format(self.chan_id)), self.y_hat)

        return channel


class TelemanomErrors:
    """Port of the Errors class (simplified). Computes smoothed errors and holds anomaly sequences."""
    def __init__(self, channel: Channel, config: Config, run_id: str, save: bool = True):
        self.config = config
        self.window_size = int(self.config.smoothing_window_size)
        self.n_windows = int((channel.y_test.shape[0] - (self.config.batch_size * self.window_size)) / self.config.batch_size)
        self.i_anom = np.array([])
        self.E_seq = []
        self.anom_scores = []

        # compute per-timestep error as Euclidean norm between predicted vector and true first-step vector
        self.e = []
        for y_h, y_t in zip(channel.y_hat, channel.y_test):
            # y_t shape: (prediction_window_size, n_features)
            true_vec = np.asarray(y_t[0])
            pred_vec = np.asarray(y_h)
            # if shapes mismatch, try to align by truncation
            if pred_vec.shape != true_vec.shape:
                minlen = min(pred_vec.size, true_vec.size)
                pred_vec = pred_vec.flatten()[:minlen]
                true_vec = true_vec.flatten()[:minlen]
            diff = pred_vec - true_vec
            self.e.append(np.linalg.norm(diff))

        smoothing_window = int(self.config.batch_size * self.config.smoothing_window_size * self.config.smoothing_perc)
        if not len(channel.y_hat) == len(channel.y_test):
            raise ValueError('len(y_hat) != len(y_test): {}, {}'.format(len(channel.y_hat), len(channel.y_test)))

        self.e_s = pd.DataFrame(self.e).ewm(span=smoothing_window).mean().values.flatten()
        if not channel.id == 'C-2':
            self.e_s[:self.config.window_size] = [np.mean(self.e_s[:self.config.window_size * 2])] * self.config.window_size

        if save and self.config and getattr(self.config, 'use_id', None):
            os.makedirs(os.path.join('data', run_id, 'smoothed_errors'), exist_ok=True)
            np.save(os.path.join('data', run_id, 'smoothed_errors', '{}.npy'.format(channel.id)), np.array(self.e_s))

        self.normalized = np.mean(self.e / np.ptp(channel.y_test))


class TelemanomDetector(algorithmbase.AlgorithmBase):
    """DANTIS wrapper for Telemanom (per-channel detector).

    This class is designed to operate on a single time series (1D array).
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            'window_size': 50,
            'prediction_window_size': 1,
            'batch_size': 32,
            'lstm_batch_size': 32,
            'epochs': 10,
            'layers': [64, 32],
            'dropout': 0.2,
            'loss_metric': 'mse',
            'optimizer': 'adam',
            'patience': 3,
            'min_delta': 0.0001,
            'validation_split': 0.1,
            'train': True,
            'predict': False,
            'use_id': 'telemanom_run',
            'smoothing_window_size': 5,
            'smoothing_perc': 0.5,
            'error_buffer': 2,
            'p': 0.1,
            'aggregation': 'mean',
            'contamination': 0.1,
            'verbose': False,
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        self._model = None
        self._last_scores = None
        self._tele_config = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            self._hyperparameter = defaults.copy()
            return

        def _hp_template(window_size=50, prediction_window_size=1, batch_size=32, lstm_batch_size=32, epochs=10,
                         layers=None, dropout=0.2, loss_metric='mse', optimizer='adam', patience=3, min_delta=1e-4,
                         validation_split=0.1, train=True, predict=False, use_id='telemanom_run', smoothing_window_size=5,
                         smoothing_perc=0.5, error_buffer=2, p=0.1, contamination=0.1, verbose=False):
            return None

        pm = utils.ParameterManagement(_hp_template)
        try:
            coerced = pm.check_hyperparameter_type(hyperparameter)
        except Exception as e:
            raise ValueError(f"Telemanom Hyperparameter Error: {e}")
        merged = pm.complete_parameters(coerced)
        for k, v in hyperparameter.items():
            if k not in merged:
                merged[k] = v
        self._hyperparameter = merged

    def _build_config(self):
        # convert hyperparameter dict into Telemanom Config object
        c = Config.from_dict(self._hyperparameter)
        # ensure attributes exist
        for k, v in self._hyperparameter.items():
            setattr(c, k, v)
        self._tele_config = c
        return c

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        series = np.asarray(x_train)
        if series.ndim == 1:
            series = series.reshape(-1, 1)
        c = self._build_config()
        chan = Channel(c, chan_id='chan')
        # create synthetic train and test split: first 50% train, rest test if no explicit train provided
        if y_train is None:
            split = max(100, series.shape[0] // 2)
            train = series[:split]
            test = series[split:]
        else:
            train = series
            test = y_train

        # pass arrays with shape (timesteps, n_features)
        if train.ndim == 1:
            train = train.reshape(-1, 1)
        if isinstance(test, np.ndarray) and test.ndim == 1:
            test = test.reshape(-1, 1)

        chan.set_data(train, train=True)
        chan.set_data(test, train=False)

        run_id = getattr(c, 'use_id', 'telemanom_run')
        model = Model(c, run_id, chan)
        chan = model.batch_predict(chan, save_y_hat=False)

        errors = TelemanomErrors(chan, c, run_id, save=False)
        self._last_scores = errors.e_s
        self._model = model
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        series = np.asarray(x)
        if series.ndim == 1:
            series = series.reshape(-1, 1)
        c = self._tele_config or self._build_config()
        chan = Channel(c, chan_id='chan')
        # For decision, treat whole series as test and require a short train (last window_size)
        # Use a rolling split: first part as train, full series as test
        if len(series) < (c.window_size + c.prediction_window_size + 1):
            raise ValueError('Series too short for Telemanom configuration')

        train = series[:max(c.window_size*2, 100)]
        test = series

        if train.ndim == 1:
            train = train.reshape(-1, 1)
        if test.ndim == 1:
            test = test.reshape(-1, 1)

        chan.set_data(train, train=True)
        chan.set_data(test, train=False)

        # If no keras model loaded/trained, train on the fly (unless predict-only)
        if self._model is None:
            model = Model(c, getattr(c, 'use_id', 'telemanom_run'), chan)
            self._model = model
        else:
            model = self._model

        chan = model.batch_predict(chan, save_y_hat=False)
        errors = TelemanomErrors(chan, c, getattr(c, 'use_id', 'telemanom_run'), save=False)
        self._last_scores = errors.e_s
        return errors.e_s

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        contamination = float(self._hyperparameter.get('contamination', 0.1))
        contamination = max(0.001, min(0.5, contamination))
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        labels = (scores > threshold).astype(int)
        return labels

    def save_model(self, path: str):
        if joblib is None:
            raise ImportError('save_model requires joblib')
        payload = {
            'hyperparameter': self._hyperparameter,
            'model_path': getattr(self._model, 'model_path', None),
        }
        joblib.dump(payload, path)

    def load_model(self, path: str):
        if joblib is None:
            raise ImportError('load_model requires joblib')
        payload = joblib.load(path)
        self._hyperparameter = payload.get('hyperparameter', self.get_default_hyperparameters())
        model_path = payload.get('model_path')
        if model_path and load_model is not None:
            # load keras model wrapper
            dummy_config = self._build_config()
            dummy_channel = Channel(dummy_config, 'chan')
            self._model = Model(dummy_config, getattr(dummy_config, 'use_id', 'telemanom_run'), dummy_channel, model_path=model_path)
        return self
