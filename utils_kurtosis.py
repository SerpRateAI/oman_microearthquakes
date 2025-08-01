from numpy import ndarray, asarray, full, arange, nan, inf, where
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kurtosis
from typing import Literal


"""
rolling_kurtosis.py

Compute rolling (windowed) kurtosis of a 1-D time series using SciPy only.
The window is specified in SECONDS; sampling rate `fs` defaults to 1000 Hz.

For mode="same", each kurtosis estimate is assigned to the **last** element
of its window (right-aligned). mode="valid" returns the compact array of
kurtosis estimates without NaN padding.

Author: Tianze Liu
"""


def rolling_kurtosis(
    x: ndarray,
    win_sec: float,
    *,
    fs: float = 1000.0,
    fisher: bool = True,
    bias: bool = False,
    mode: Literal["same", "valid"] = "same",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> ndarray:
    """
    Parameters
    ----------
    x : ndarray
        1-D input signal.
    win_sec : float
        Window length in seconds (>0 and yielding ≥4 samples).
    fs : float, default 1000.0
        Sampling rate in Hz (>0).
    fisher, bias, nan_policy : see scipy.stats.kurtosis.
    mode : {"same", "valid"}, default "same"
        "same"  – return an array the same length as `x`, right-aligned,
                  with NaNs padding the first `window-1` positions.
        "valid" – return only the kurtosis values (length n-window+1).

    Returns
    -------
    ndarray
        Rolling kurtosis array.
    """
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if win_sec <= 0:
        raise ValueError("win_sec must be positive.")

    window = int(round(win_sec * fs))
    if window < 4:
        raise ValueError(
            f"win_sec × fs yields window={window} < 4; increase win_sec or fs."
        )

    x = asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array.")
    n = x.size
    if n < window:
        return full(n if mode == "same" else 0, nan, dtype=float)

    # View into rolling windows: shape (n - window + 1, window)
    W = sliding_window_view(x, window_shape=window)

    # Kurtosis along the window axis
    k_valid = kurtosis(
        W, axis=1, fisher=fisher, bias=bias, nan_policy=nan_policy
    )  # length n - window + 1

    if mode == "valid":
        return k_valid

    if mode == "same":           # right-aligned output
        k = full(n, nan, dtype=float)
        k[window - 1 :] = k_valid
        return k

    raise ValueError("mode must be 'same' or 'valid'.")

    
"""
Estimate the arrival times from the kurtosis
"""
def get_arrival_time(kurtosis: ndarray,
                           timeax: ndarray,
                           threshold: float = 4.0):
    
    # Determine if the kurtosis and time axis have the same length
    if len(kurtosis) != len(timeax):
        raise ValueError("The kurtosis and time axis must have the same length")
    
    # Get the indices of the first kurtosis above the threshold
    indices = where(kurtosis >= threshold)[0]

    if len(indices) == 0:
        return nan
    
    # Get the time of the first kurtosis above the threshold
    time_pick = timeax[indices[0]]

    return time_pick





