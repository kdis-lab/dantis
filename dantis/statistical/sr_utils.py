"""
Utilities for Spectral Residual detector implemented in pure Python (numpy)
and optional numba acceleration. These are ported/adapted from the
reference implementation but reimplemented to avoid importing external
packages or Cython modules.

Functions provided:
- median_filter(data, window, need_two_end=True)
- deanomaly_entire(values, entire_anomalies)
- calculate_boundary_unit_last(data)
- calculate_boundary_unit_entire(data, is_anomaly)
- calculate_margin(unit, sensitivity)
- calculate_anomaly_score(value, expected_value, unit, is_anomaly)
- calculate_anomaly_scores(values, expected_values, units, is_anomaly)
- calculate_expected_value(values, anomaly_index)

Numba is used when available to speed sliding-window operations; otherwise
pure numpy fallbacks are provided.
"""
from __future__ import annotations

from typing import Sequence, List
import numpy as np
import bisect

try:
    from numba import njit
except Exception:
    njit = None


# Precomputed factors array (copied from original boundary_utils)
factors = [
    184331.62871148242, 141902.71648305038, 109324.12672037778, 84289.9974713784, 65038.57829581667,
    50222.84038287002, 38812.08684920403, 30017.081863266845, 23233.035497884553, 17996.15452973242,
    13950.50738738947, 10822.736530170265, 8402.745753237783, 6528.939979205737, 5076.93622022219,
    3950.92312857758, 3077.042935029268, 2398.318733460069, 1870.7634426365591, 1460.393007522685,
    1140.9320371270976, 892.0500681212648, 698.0047481387048, 546.5972968979678, 428.36778753759233,
    335.97473532360186, 263.71643275007995, 207.16137686573444, 162.8627176617409, 128.13746472206208,
    100.8956415134347, 79.50799173635517, 62.70346351447568, 49.48971074544253, 39.09139869308257,
    30.90229145698227, 24.448015393182175, 19.35709849024717, 15.338429865489042, 12.163703303322,
    9.653732780414286, 7.667778221139226, 6.095213212352326, 4.8490160798347866, 3.8606815922251485,
    3.076240312529999, 2.4531421949999994, 1.9578149999999996, 1.5637499999999998, 1.25, 1.0,
    0.8695652173913044, 0.7554867223208555, 0.655804446459076, 0.5687809596349316, 0.4928777813127657,
    0.4267340097946024, 0.36914706729636887, 0.3190553736355825, 0.27552277516026125, 0.23772456873189068,
    0.20493497304473338, 0.17651591132190647, 0.1519069804835684, 0.13061649224726435, 0.11221348131208278,
    0.09632058481723846, 0.08260770567516164, 0.0707863801843716, 0.06060477755511267, 0.051843265658779024,
    0.0443104834690419, 0.03783986632710667, 0.03228657536442549, 0.027524787181948417, 0.02344530424356765,
    0.019953450420057577, 0.01696721974494692, 0.014415649740821513, 0.012237393667929978, 0.010379468759906684,
    0.008796159966022614, 0.0074480609365136455, 0.006301235986898177, 0.00532648857725966, 0.004498723460523362,
    0.0037963911059268884, 0.0032010043051660104, 0.002696718032995797, 0.0022699646742388863, 0.0019091376570554135,
    0.0011570531254881296, 0.000697019955113331, 0.00041737721863073713, 0.000248438820613534, 0.00014700521929794912,
    8.647365841055832e-05, 5.056939088336744e-05, 2.9400808653120604e-05, 1.6994687082728674e-05, 9.767061541798089e-06
]


def _median_filter_numpy(data: np.ndarray, window: int, need_two_end: bool = True) -> np.ndarray:
    # Simple sliding-window median with edge handling.
    n = len(data)
    if n == 0:
        return np.array([])
    half = window // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        out[i] = np.median(data[start:end])
    return out


if njit is not None:
    @njit
    def _median_filter_numba(data, window, need_two_end=True):
        n = len(data)
        out = np.empty(n, dtype=np.float64)
        half = window // 2
        for i in range(n):
            start = i - half
            if start < 0:
                start = 0
            end = i + half + 1
            if end > n:
                end = n
            # copy window
            tmp = np.empty(end - start, dtype=np.float64)
            for j in range(start, end):
                tmp[j - start] = data[j]
            # simple selection sort median (small windows typical)
            tmp.sort()
            out[i] = tmp[len(tmp)//2]
        return out


def median_filter(data: Sequence[float], window: int, need_two_end: bool = True) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if njit is not None:
        try:
            return _median_filter_numba(arr, window, need_two_end)
        except Exception:
            return _median_filter_numpy(arr, window, need_two_end)
    else:
        return _median_filter_numpy(arr, window, need_two_end)


def deanomaly_entire(values: np.ndarray, entire_anomalies: Sequence[int]) -> np.ndarray:
    deanomaly_data = np.copy(values)
    min_points_to_fit = 4
    length = len(values)
    anomalies = set(int(i) for i in entire_anomalies)
    for idx in anomalies:
        step = 1
        start = max(idx - step, 0)
        end = min(length - 1, idx + step)
        fit_values = [(i, values[i]) for i in range(start, end+1) if i not in anomalies]
        while len(fit_values) < min_points_to_fit and (start > 0 or end < length-1):
            step = step + 2
            start = max(idx - step, 0)
            end = min(length - 1, idx + step)
            fit_values = [(i, values[i]) for i in range(start, end+1) if i not in anomalies]

        if len(fit_values) > 1:
            x, y = zip(*fit_values)
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)
            n = len(x)
            sum_x = x.sum()
            sum_y = y.sum()
            sum_xx = (x * x).sum()
            sum_xy = (x * y).sum()
            denom = (n * sum_xx - sum_x * sum_x)
            if denom == 0:
                a = 0.0
                b = y.mean()
            else:
                a = (n * sum_xy - sum_x * sum_y) / denom
                b = (sum_xx * sum_y - sum_x * sum_xy) / denom
            deanomaly_data[idx] = a * idx + b

    return deanomaly_data


def calculate_boundary_unit_last(data: Sequence[float]) -> float:
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return 0.0
    calculation_size = len(data) - 1
    window = int(min(calculation_size // 3, 512))
    if window <= 0:
        return 1.0
    trends = np.abs(np.asarray(median_filter(data[:calculation_size], window, need_two_end=True), dtype=float))
    unit = max(np.mean(trends), 1.0)
    if not np.isfinite(unit):
        raise Exception('Not finite unit value')
    return unit


def calculate_boundary_unit_entire(data: Sequence[float], is_anomaly: Sequence[bool]) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    is_anomaly = np.asarray(is_anomaly, dtype=bool)
    if len(data) == 0:
        return np.array([])
    window = int(min(len(data)//3, 512))
    if window <= 0:
        return np.ones_like(data)
    trends = np.abs(np.asarray(median_filter(data, window, need_two_end=True), dtype=float))
    valid_trend = [t for a, t in zip(is_anomaly, trends) if not a]
    if len(valid_trend) > 0:
        average_part = float(np.mean(valid_trend))
        trend_fraction = 0.5
        units = trend_fraction * trends + average_part * (1 - trend_fraction)
    else:
        units = trends
    if not np.all(np.isfinite(units)):
        raise Exception('Not finite unit values')
    units = np.clip(units, 1.0, max(np.max(units), 1.0))
    return units


def calculate_margin(unit: float, sensitivity: float) -> float:
    def calculate_margin_core(unit, sensitivity):
        lb = int(sensitivity)
        return (factors[lb + 1] + (factors[lb] - factors[lb + 1]) * (1 - sensitivity + lb)) * unit

    if 0 > sensitivity or sensitivity > 100:
        raise Exception('sensitivity should be integer in [0, 100]')
    if unit <= 0:
        raise Exception('unit should be a positive number')
    if sensitivity == 100:
        return 0.0
    return calculate_margin_core(unit, sensitivity)


def calculate_anomaly_score(value: float, expected_value: float, unit: float, is_anomaly: bool) -> float:
    if not is_anomaly:
        return 0.0
    distance = abs(expected_value - value)
    margins = [calculate_margin(unit, i) for i in range(101)][::-1]
    lb = bisect.bisect_left(margins, distance)
    if lb == 0:
        return 0.0
    elif lb >= 100:
        return 1.0
    else:
        a, b = margins[lb-1], margins[lb]
        score = lb - 1 + (distance - a) / (b - a)
    return score / 100.0


def calculate_anomaly_scores(values: Sequence[float], expected_values: Sequence[float], units: Sequence[float], is_anomaly: Sequence[bool]) -> List[float]:
    return [calculate_anomaly_score(v, ev, u, ia) for v, ev, u, ia in zip(values, expected_values, units, is_anomaly)]


def calculate_expected_value(values: Sequence[float], anomaly_index: Sequence[int]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ex = deanomaly_entire(values, anomaly_index)
    length = len(ex)
    fft_coef = np.fft.fft(ex)
    # zero out mid-band between 3/8 and 5/8
    low = int(length * 3 / 8)
    high = int(length * 5 / 8)
    for i in range(length):
        if low <= i < high:
            fft_coef.real[i] = 0
            fft_coef.imag[i] = 0
    exps = np.fft.ifft(fft_coef)
    return exps.real
