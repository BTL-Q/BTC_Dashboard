"""회귀 계산 — statsmodels 없이 단순 OLS를 직접 구현.

의존성을 numpy/scipy로만 유지하려고 정규방정식을 직접 푼다.
검증: 총분산 = β²·Var(x) + 잔차분산 이 소수 5자리까지 일치함을 확인했다.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def market_model(y: np.ndarray, x: np.ndarray, bars_per_year: float) -> dict | None:
    """y = α + β·x + ε 단순회귀.

    y = 비교 코인 수익률, x = 기준 코인 수익률.
    표본이 30봉 미만이면 None.
    """
    n = len(x)
    if n < 30:
        return None

    xbar, ybar = float(x.mean()), float(y.mean())
    dx, dy = x - xbar, y - ybar
    sxx = float((dx * dx).sum())
    sst = float((dy * dy).sum())
    if sxx <= 0 or sst <= 0:
        return None

    beta = float((dx * dy).sum() / sxx)
    alpha = ybar - beta * xbar
    resid = y - (alpha + beta * x)
    sse = float((resid * resid).sum())

    dof = n - 2
    s2 = sse / dof
    se_beta = math.sqrt(s2 / sxx)
    se_alpha = math.sqrt(s2 * (1.0 / n + xbar * xbar / sxx))

    t_zero = beta / se_beta if se_beta > 0 else np.nan          # H0: β = 0 (관계 없음)
    t_one = (beta - 1.0) / se_beta if se_beta > 0 else np.nan   # H0: β = 1 (기준 코인과 동일)
    crit = float(stats.t.ppf(0.975, dof))

    return {
        "beta": beta,
        "se_beta": se_beta,
        "beta_lo": beta - crit * se_beta,
        "beta_hi": beta + crit * se_beta,
        "t_zero": float(t_zero),
        "p_zero": float(2 * stats.t.sf(abs(t_zero), dof)),
        "t_one": float(t_one),
        "p_one": float(2 * stats.t.sf(abs(t_one), dof)),
        "alpha_bar": alpha,
        "alpha_ann": alpha * bars_per_year,
        "se_alpha": se_alpha,
        "p_alpha": float(2 * stats.t.sf(abs(alpha / se_alpha), dof)) if se_alpha > 0 else np.nan,
        "r2": 1.0 - sse / sst,
        "rho": float(np.corrcoef(x, y)[0, 1]),
        "resid_vol": float(resid.std(ddof=1)) * math.sqrt(bars_per_year),
        "total_vol": float(y.std(ddof=1)) * math.sqrt(bars_per_year),
        "n": n,
    }


def rolling_beta_r2(y: pd.Series, x: pd.Series, window: int):
    """롤링 β와 R².

    단순회귀에서 β = cov(x,y)/var(x), R² = corr² 라는 성질을 이용한다.
    봉마다 회귀를 새로 돌리는 것과 같은 값이면서 훨씬 빠르다.
    """
    beta = y.rolling(window).cov(x) / x.rolling(window).var()
    r2 = y.rolling(window).corr(x) ** 2
    return beta, r2


def lead_lag_profile(y_vals: tuple, x_vals: tuple, max_lag: int) -> pd.DataFrame:
    """기준 코인을 k봉 밀어가며 회귀.

    k > 0 이면 기준 코인이 선행 (= 보고 나서 따라 사도 되는지),
    k = 0 은 동시점, k < 0 이면 비교 코인이 먼저 움직인다.
    """
    y = pd.Series(y_vals)
    x = pd.Series(x_vals)
    rows = []
    for k in range(-max_lag, max_lag + 1):
        pair = pd.concat([y, x.shift(k)], axis=1).dropna()
        if len(pair) < 30:
            continue
        yy = pair.iloc[:, 0].values
        xx = pair.iloc[:, 1].values
        n = len(xx)
        xb, yb = xx.mean(), yy.mean()
        dx, dy = xx - xb, yy - yb
        sxx = float((dx * dx).sum())
        sst = float((dy * dy).sum())
        if sxx <= 0 or sst <= 0:
            continue
        beta = float((dx * dy).sum() / sxx)
        alpha = yb - beta * xb
        resid = yy - (alpha + beta * xx)
        sse = float((resid * resid).sum())
        dof = n - 2
        se = math.sqrt((sse / dof) / sxx)
        t = beta / se if se > 0 else np.nan
        rows.append({
            "k": k,
            "beta": beta,
            "r2": 1.0 - sse / sst,
            "p": float(2 * stats.t.sf(abs(t), dof)) if se > 0 else np.nan,
            "n": n,
        })
    return pd.DataFrame(rows)
