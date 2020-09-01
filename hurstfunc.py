import numpy as np
from sklearn.linear_model import LinearRegression


def RS_func(series):
    """
    Get rescaled range from a time-series of values.
    """
    L = np.log10(series[1:] / series[:-1])  # logarithmic ratio
    Z = np.mean(L)                          # mean of log series
    C = np.cumsum(L-Z)                      # cumulative deviation of series
    R = max(C) - min(C)                     # range of series
    S = np.std(L)                           # standard deviation of series

    if R == 0 or S == 0:
        return 0                            # return 0 to skip this interval due undefined R/S

    return R / S                            # return rescaled range of series


def hurst_func(series):
    """
    interpretation of return value
    hurst < 0.5 - input_ts is mean reverting
    hurst = 0.5 - input_ts is effectively random/geometric brownian motion
    hurst > 0.5 - input_ts is trending
    """
    min_window = 10
    max_window = len(series) - 1
    by_factor = np.log10(2.0)
    window_sizes = list(map(lambda x: int(10 ** x),
                            np.arange(np.log10(min_window), np.log10(max_window), by_factor)))
    window_sizes.append(len(series))

    RS = []
    for w in window_sizes:
        rs = []
        for start in range(0, len(series), w):
            if (start + w) > len(series):
                break
            res = RS_func(series[start:start + w].astype(np.float64))
            if res != 0:
                rs.append(res)
        RS.append(np.mean(rs))
    lm1 = LinearRegression().fit(np.log10(window_sizes).reshape(-1, 1),
                                 np.log10(RS).reshape(-1, 1))
    hurst_exp = lm1.coef_[0][0]
    return hurst_exp


if __name__ == '__main__':
    np.random.seed(42)
    random_changes = 1. + np.random.randn(99999) / 1000.
    series = np.cumprod(random_changes)
    print(hurst_func(series))
