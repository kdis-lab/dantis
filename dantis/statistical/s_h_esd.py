"""
S_H_ESD adapter implementing AlgorithmBase.

Este módulo proporciona una implementación autocontenida de las funciones
necesarias para realizar la detección S-H-ESD (normalización de
timestamps, descomposición STL y test ESD iterativo). No contiene
referencias externas a otros paquetes de la colección de ejemplos.

La clase `S_H_ESD` implementa `fit`, `decision_function` y `predict`
siguiendo el contrato de `AlgorithmBase`.
"""
from __future__ import annotations

import re
from typing import Optional, Dict, Any
import datetime
from heapq import nlargest
from itertools import groupby
from math import sqrt

import numpy as np
import pandas as pd

from scipy.stats import t as student_t
from statsmodels.robust.scale import mad
from sklearn.preprocessing import MinMaxScaler

try:
    from rstl import STL
except Exception:  # pragma: no cover - helpful message if dependency missing
    STL = None

from .. import algorithmbase
from .. import utils


def datetimes_from_ts(column: pd.Series) -> pd.Series:
    # convert posix integer timestamps (seconds) to UTC datetimes
    return pd.to_datetime(column.astype(int), unit='s', utc=True)


def date_format(column: pd.Series, fmt: str) -> pd.Series:
    return column.map(lambda datestring: datetime.datetime.strptime(datestring, fmt))


def format_timestamp(indf: pd.DataFrame, index: int = 0) -> pd.DataFrame:
    # if already datetime64, return
    if indf.dtypes[index].type is np.datetime64:
        return indf

    column = indf.iloc[:, index].astype(str)

    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \+\d{4}$", column.iloc[0]):
        column = date_format(column, "%Y-%m-%d %H:%M:%S")
    elif re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", column.iloc[0]):
        column = date_format(column, "%Y-%m-%d %H:%M:%S")
    elif re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", column.iloc[0]):
        column = date_format(column, "%Y-%m-%d %H:%M")
    elif re.match(r"^\d{2}/\d{2}/\d{2}$", column.iloc[0]):
        column = date_format(column, "%m/%d/%y")
    elif re.match(r"^\d{2}/\d{2}/\d{4}$", column.iloc[0]):
        column = date_format(column, "%Y%m%d")
    elif re.match(r"^\d{4}\d{2}\d{2}$", column.iloc[0]):
        column = date_format(column, "%Y/%m/%d/%H")
    elif re.match(r"^\d{10}$", column.iloc[0]):
        column = datetimes_from_ts(column)

    indf.iloc[:, index] = column
    return indf


def get_gran(tsdf: pd.DataFrame, index: int = 0) -> str:
    col = tsdf.iloc[:, index]
    n = len(col)
    largest, second_largest = nlargest(2, col)
    gran = int(round(np.timedelta64(largest - second_largest) / np.timedelta64(1, 's')))

    if gran >= 86400:
        return "day"
    elif gran >= 3600:
        return "hr"
    elif gran >= 60:
        return "min"
    elif gran >= 1:
        return "sec"
    else:
        return "ms"


def detect_anoms(data: pd.DataFrame, k=0.49, alpha=0.05, num_obs_per_period=None,
                 use_decomp=True, one_tail=True,
                 upper_tail=True, verbose=False):
    if STL is None:
        raise RuntimeError("rstl.STL is required for S-H-ESD (install rstl package)")

    if num_obs_per_period is None:
        raise ValueError("must supply period length for time series decomposition")

    if list(data.columns.values) != ["timestamp", "value"]:
        data = data.copy()
        data.columns = ["timestamp", "value"]

    num_obs = len(data)
    if num_obs < num_obs_per_period * 2:
        raise ValueError("Anom detection needs at least 2 periods worth of data")

    # drop/validate nulls
    if (len(list(map(lambda x: x[0], list(groupby(pd.isnull(
            pd.concat([pd.Series([np.nan]),
                       data.value,
                       pd.Series([np.nan])])))))) ) > 3):
        raise ValueError("Data contains non-leading NAs. Replace NAs with interpolation before detection.")
    else:
        data = data.dropna()

    data = data.set_index('timestamp')

    decomp = STL(data.value, num_obs_per_period, "periodic", robust=True)

    d = {
        'timestamp': data.index,
        'value': data.value - decomp.seasonal - data.value.median()
    }
    data = pd.DataFrame(d)

    p = {
        'timestamp': data.index,
        'value': pd.to_numeric(pd.Series(decomp.trend + decomp.seasonal).truncate())
    }
    data_decomp = pd.DataFrame(p)

    max_outliers = int(num_obs * k)
    if max_outliers == 0:
        raise ValueError("Not enough observations for anomaly detection")

    n = len(data.timestamp)
    R_idx = list(range(max_outliers))
    num_anoms = 0

    for i in range(1, max_outliers + 1):
        if one_tail:
            if upper_tail:
                ares = data.value - data.value.median()
            else:
                ares = data.value.median() - data.value
        else:
            ares = (data.value - data.value.median()).abs()

        data_sigma = mad(data.value)
        if data_sigma == 0:
            break

        ares = ares / float(data_sigma)
        R = ares.max()
        temp_max_idx = ares[ares == R].index.tolist()[0]
        R_idx[i - 1] = temp_max_idx
        data = data[data.index != R_idx[i - 1]]

        if one_tail:
            p = 1 - alpha / float(n - i + 1)
        else:
            p = 1 - alpha / float(2 * (n - i + 1))

        t = student_t.ppf(p, (n - i - 1))
        lam = t * (n - i) / float(sqrt((n - i - 1 + t**2) * (n - i + 1)))

        if R > lam:
            num_anoms = i

    if num_anoms > 0:
        R_idx = R_idx[:num_anoms]
    else:
        R_idx = None

    return {
        'anoms': R_idx,
        'stl': data_decomp
    }


from collections import namedtuple
Direction = namedtuple('Direction', ['one_tail', 'upper_tail'])


def detect_ts(df: pd.DataFrame, max_anoms=0.10, direction='pos',
              alpha=0.05, longterm=False,
              piecewise_median_period_weeks=2, verbose=False):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("data must be a single data frame.")
    else:
        if len(df.columns) != 2 or not df.iloc[:,1].map(np.isreal).all():
            raise ValueError(("data must be a 2 column DataFrame, with the"
                              "first column timestamps and second numeric values."))

        if (not (df.dtypes[0].type is np.datetime64)
            and not (df.dtypes[0].type is np.int64)):
            df = format_timestamp(df)

    if list(df.columns.values) != ["timestamp", "value"]:
        df = df.copy()
        df.columns = ["timestamp", "value"]

    if max_anoms > 0.49:
        length = len(df.value)
        raise ValueError("max_anoms must be less than 50% of the data points")

    if not direction in ['pos', 'neg', 'both']:
        raise ValueError("direction options are: pos | neg | both.")

    if not (0.01 <= alpha or alpha <= 0.1):
        if verbose:
            import warnings
            warnings.warn(("alpha is the statistical significance, usually between 0.01 and 0.1"))

    if not isinstance(longterm, bool):
        raise ValueError("longterm must be a boolean")

    if piecewise_median_period_weeks < 2:
        raise ValueError("piecewise_median_period_weeks must be > 2 weeks")

    gran = get_gran(df)

    if gran == 'sec':
        df.timestamp = df.timestamp.map(lambda d: ":".join(str(d).split(":")[:-1] + ["00"]))
        df = format_timestamp(df.groupby('timestamp').aggregate(np.sum).reset_index())

    gran_period = {
        'min': 1440,
        'hr': 24,
        'day': 7
    }
    period = gran_period.get(gran)
    if not period:
        raise ValueError('%s granularity detected. Not supported.' % gran)
    num_obs = len(df.value)

    clamp = (1 / float(num_obs))
    if max_anoms < clamp:
        max_anoms = clamp

    if longterm:
        if gran == "day":
            num_obs_in_period = period * piecewise_median_period_weeks + 1
            num_days_in_period = 7 * piecewise_median_period_weeks + 1
        else:
            num_obs_in_period = period * 7 * piecewise_median_period_weeks
            num_days_in_period = 7 * piecewise_median_period_weeks

        last_date = df.timestamp.iloc[-1]
        all_data = []

        for j in range(0, len(df.timestamp), num_obs_in_period):
            start_date = df.timestamp.iloc[j]
            end_date = min(start_date + datetime.timedelta(days=num_days_in_period), df.timestamp.iloc[-1])
            if (end_date - start_date).days == num_days_in_period:
                sub_df = df[(df.timestamp >= start_date) & (df.timestamp < end_date)]
            else:
                sub_df = df[(df.timestamp > (last_date - datetime.timedelta(days=num_days_in_period))) & (df.timestamp <= last_date)]
            all_data.append(sub_df)
    else:
        all_data = [df]

    all_anoms = pd.DataFrame(columns=['timestamp', 'value'])
    seasonal_plus_trend = pd.DataFrame(columns=['timestamp', 'value'])
    timestamp_ordering = {}

    for i in range(len(all_data)):
        directions = {
            'pos': Direction(True, True),
            'neg': Direction(True, False),
            'both': Direction(False, True)
        }
        anomaly_direction = directions[direction]

        det_result = detect_anoms(all_data[i], k=max_anoms, alpha=alpha,
                                  num_obs_per_period=period,
                                  use_decomp=True,
                                  one_tail=anomaly_direction.one_tail,
                                  upper_tail=anomaly_direction.upper_tail,
                                  verbose=verbose)

        data_decomp = det_result['stl']
        detected_indices = det_result['anoms']
        if detected_indices:
            anoms = all_data[i][all_data[i].timestamp.isin(detected_indices)]
            timestamp_ordering.update({t: len(detected_indices) - idx for idx, t in enumerate(detected_indices)})
        else:
            anoms = pd.DataFrame(columns=['timestamp', 'value'])

        all_anoms = pd.concat([all_anoms, anoms], ignore_index=True)
        seasonal_plus_trend = pd.concat([seasonal_plus_trend, data_decomp], ignore_index=True)

    try:
        all_anoms.drop_duplicates(subset=['timestamp'], inplace=True)
        seasonal_plus_trend.drop_duplicates(subset=['timestamp'], inplace=True)
    except TypeError:
        all_anoms = all_anoms.drop_duplicates()
        seasonal_plus_trend = seasonal_plus_trend.drop_duplicates()

    anom_pct = (len(df.value) / float(num_obs)) * 100
    if anom_pct == 0:
        return None

    anoms = df.merge(all_anoms, how="left", on="timestamp")
    def get_score(row):
        return 0 if pd.isna(row.get("value_y")) else timestamp_ordering.get(row["timestamp"], 0)
    anoms["value_y"] = anoms.apply(get_score, axis=1)
    anoms.rename(columns={"value_y": "scores"}, inplace=True)
    return anoms


class S_H_ESD(algorithmbase.AlgorithmBase):
    """S-H-ESD detector adapter implementing AlgorithmBase.

    Input: DataFrame with two columns (timestamp, value) or a 1-D numpy array.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "max_anoms": 0.10,
            "direction": "pos",
            "alpha": 0.05,
            "longterm": False,
            "piecewise_median_period_weeks": 2,
            "verbose": False,
            "contamination": 0.1,
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        self._last_scores = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            # Use defaults copy to avoid accidental external mutation
            self._hyperparameter = defaults.copy()
            return

        # Create a template function whose signature carries the defaults
        def _hp_template(max_anoms=0.10, direction='pos', alpha=0.05,
                         longterm=False, piecewise_median_period_weeks=2,
                         verbose=False, contamination=0.1):
            return None

        pm = utils.ParameterManagement(_hp_template)
        # First coerce types for known keys
        try:
            coerced = pm.check_hyperparameter_type(hyperparameter)
        except Exception as e:
            raise ValueError(f"Invalid hyperparameter types: {e}")

        # Merge with defaults, preserving defaults for missing keys
        merged = pm.complete_parameters(coerced)

        # Keep any unknown keys provided by user as-is
        for k, v in (hyperparameter.items()):
            if k not in merged:
                merged[k] = v

        self._hyperparameter = merged

    def _to_dataframe(self, x) -> pd.DataFrame:
        # If DataFrame with 2 columns, return as-is (ensure column names)
        if isinstance(x, pd.DataFrame):
            df = x.copy()
            if df.shape[1] >= 2:
                df = df.iloc[:, :2]
            df.columns = ["timestamp", "value"]
            return df

        arr = np.asarray(x)
        if arr.ndim == 1:
            return pd.DataFrame({"timestamp": np.arange(len(arr)), "value": arr})
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            return pd.DataFrame({"timestamp": arr[:,0], "value": arr[:,1]})
        else:
            raise ValueError("Unsupported input shape for S_H_ESD")

    def fit(self, x_train, y_train=None):
        df = self._to_dataframe(x_train)
        params = self._hyperparameter
        anoms = detect_ts(df, max_anoms=float(params.get("max_anoms", 0.10)),
                          direction=str(params.get("direction", "pos")),
                          alpha=float(params.get("alpha", 0.05)),
                          longterm=bool(params.get("longterm", False)),
                          piecewise_median_period_weeks=int(params.get("piecewise_median_period_weeks", 2)),
                          verbose=bool(params.get("verbose", False)))
        if anoms is None:
            scores = np.zeros(len(df))
        else:
            scores = anoms["scores"].values
        self._last_scores = scores
        return scores

    def decision_function(self, x):
        df = self._to_dataframe(x)
        params = self._hyperparameter
        anoms = detect_ts(df, max_anoms=float(params.get("max_anoms", 0.10)),
                          direction=str(params.get("direction", "pos")),
                          alpha=float(params.get("alpha", 0.05)),
                          longterm=bool(params.get("longterm", False)),
                          piecewise_median_period_weeks=int(params.get("piecewise_median_period_weeks", 2)),
                          verbose=bool(params.get("verbose", False)))
        if anoms is None:
            return np.zeros(len(df))
        return anoms["scores"].values

    def predict(self, x):
        scores = self.get_anomaly_score(x)
        contamination = float(self._hyperparameter.get("contamination", 0.1))
        if not (0 < contamination < 0.5):
            contamination = max(0.001, min(0.5, contamination))
        valid = np.isfinite(scores)
        if not np.any(valid):
            return np.zeros_like(scores, dtype=int)
        threshold = np.percentile(scores[valid], 100.0 * (1.0 - contamination))
        labels = np.zeros_like(scores, dtype=int)
        labels[valid] = (scores[valid] >= threshold).astype(int)
        return labels
