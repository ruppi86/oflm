import math
import numpy as np

EPS = 1e-6  # variance threshold

def safe_welch(a: list[float], b: list[float]):
    """Welch t-test that returns None when either group variance is ~0.

    Returns (t, df, p) or None.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    if v1 < EPS or v2 < EPS:
        return None
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df = df_num / df_den if df_den else 1
    try:
        from scipy.stats import t as t_dist
        p = t_dist.sf(abs(t_stat), df) * 2
    except Exception:
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    return t_stat, df, p


def effect_size(a: list[float], b: list[float]):
    """Return Cohen d using pooled SD; falls back to range-based when SD≈0."""
    pooled_sd = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_sd < EPS:
        return np.mean(b) - np.mean(a)  # difference over range 1.0 assumed
    return (np.mean(b) - np.mean(a)) / pooled_sd 