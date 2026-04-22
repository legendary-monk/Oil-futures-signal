"""
quant_stack.py — Free institutional-style quant stack for oil futures.

This module intentionally keeps all computation in pure Python/numpy/pandas so the
system remains zero-cost (no paid datasets, no licensed solvers).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _safe_std(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(np.std(x)) if len(x) > 1 else 0.0


def _zscore(value: float, scale: float) -> float:
    if not np.isfinite(value) or scale <= 1e-12:
        return 0.0
    return float(np.clip(value / scale, -1.0, 1.0))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


@dataclass
class QuantStackResult:
    score: float
    diagnostics: Dict[str, Any]


def _extract_series(instruments: Dict[str, pd.DataFrame], name: str) -> pd.Series | None:
    df = instruments.get(name)
    if df is None or df.empty or 'Close' not in df:
        return None
    return pd.to_numeric(df['Close'], errors='coerce').dropna()


def _act360_yearfrac(days: int) -> float:
    return max(days, 1) / 360.0


def _cost_of_carry(spot: float, rate: float, storage: float, conv_yield: float, t: float) -> float:
    return spot * exp((rate + storage - conv_yield) * t)


def _convenience_yield(spot: float, fwd: float, rate: float, storage: float, t: float) -> float:
    if spot <= 0 or fwd <= 0 or t <= 0:
        return 0.0
    return rate + storage - (log(fwd / spot) / t)


def _storage_arb_bounds(spot: float, rate: float, storage: float, t: float) -> Tuple[float, float]:
    carry = exp((rate + storage) * t)
    return spot, spot * carry


def _gbm_params(returns: np.ndarray) -> Tuple[float, float]:
    mu = float(np.nanmean(returns)) * 252 if len(returns) else 0.0
    sigma = _safe_std(returns) * sqrt(252)
    return mu, sigma


def _ou_speed(series: np.ndarray) -> float:
    if len(series) < 20:
        return 0.0
    x = series[:-1]
    y = series[1:]
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    denom = np.dot(x, x)
    if abs(denom) < 1e-12:
        return 0.0
    phi = float(np.dot(x, y) / denom)
    phi = min(max(phi, 1e-6), 0.9999)
    return float(-log(phi))


def _jump_metrics(returns: np.ndarray) -> Tuple[float, float]:
    if len(returns) < 40:
        return 0.0, 0.0
    sigma = _safe_std(returns)
    if sigma <= 1e-12:
        return 0.0, 0.0
    jumps = np.abs(returns) > (2.5 * sigma)
    intensity = float(np.mean(jumps))
    avg_jump = float(np.nanmean(np.abs(returns[jumps]))) if np.any(jumps) else 0.0
    return intensity, avg_jump


def _variance_gamma_proxy(returns: np.ndarray) -> float:
    if len(returns) < 30:
        return 0.0
    centered = returns - np.nanmean(returns)
    s = _safe_std(centered)
    if s <= 1e-12:
        return 0.0
    skew = float(np.nanmean((centered / s) ** 3))
    kurt = float(np.nanmean((centered / s) ** 4)) - 3.0
    return 0.5 * skew + 0.5 * (kurt / 6.0)


def _seasonality_strength(series: pd.Series) -> float:
    if len(series) < 60:
        return 0.0
    monthly = series.groupby(series.index.month).mean()
    if monthly.std() == 0:
        return 0.0
    return float((monthly.max() - monthly.min()) / max(monthly.mean(), 1e-8))


def _pca_first_component(matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    if matrix.shape[0] < 10 or matrix.shape[1] < 2:
        return 0.0, np.zeros(matrix.shape[1])
    x = matrix - np.nanmean(matrix, axis=0)
    cov = np.cov(x.T)
    vals, vecs = np.linalg.eigh(cov)
    idx = int(np.argmax(vals))
    total = float(np.sum(vals))
    explained = float(vals[idx] / total) if total > 1e-12 else 0.0
    return explained, vecs[:, idx]


def _cointegration_proxy(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 30 or len(y) < 30:
        return 0.0
    beta = np.polyfit(y, x, 1)[0]
    spread = x - beta * y
    return -_safe_std(np.diff(spread))


def _kalman_beta(y: np.ndarray, x: np.ndarray, q: float = 1e-4, r: float = 1e-2) -> float:
    if len(x) < 10:
        return 1.0
    beta, p = 1.0, 1.0
    for xi, yi in zip(x, y):
        p = p + q
        k = p * xi / (xi * xi * p + r)
        beta = beta + k * (yi - beta * xi)
        p = (1 - k * xi) * p
    return float(beta)


def _ar1_forecast(returns: np.ndarray) -> float:
    if len(returns) < 20:
        return 0.0
    x, y = returns[:-1], returns[1:]
    denom = np.dot(x, x)
    if abs(denom) < 1e-12:
        return 0.0
    phi = float(np.dot(x, y) / denom)
    return phi * float(returns[-1])


def _garch_vol(returns: np.ndarray) -> float:
    if len(returns) < 30:
        return _safe_std(returns)
    omega, alpha, beta = 1e-6, 0.08, 0.90
    h = np.var(returns)
    for r in returns[-120:]:
        h = omega + alpha * (r ** 2) + beta * h
    return float(sqrt(max(h, 1e-12)))


def _hurst_exponent(series: np.ndarray) -> float:
    if len(series) < 120:
        return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    tau = np.array(tau)
    valid = tau > 0
    if valid.sum() < 5:
        return 0.5
    poly = np.polyfit(np.log(np.array(list(lags))[valid]), np.log(tau[valid]), 1)
    return float(np.clip(poly[0] * 2.0, 0.05, 0.95))


def _bayesian_up_prob(returns: np.ndarray) -> float:
    up = float(np.sum(returns > 0))
    down = float(np.sum(returns <= 0))
    # Beta(1,1) prior
    return (1 + up) / (2 + up + down)


def _evt_tail_risk(returns: np.ndarray) -> float:
    if len(returns) < 50:
        return 0.0
    left = np.sort(returns)[: max(3, int(0.05 * len(returns)))]
    return float(abs(np.mean(left)))


def _copula_tail_dependence(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 50 or len(y) < 50:
        return 0.0
    qx = np.quantile(x, 0.1)
    qy = np.quantile(y, 0.1)
    joint = np.mean((x <= qx) & (y <= qy))
    return float(joint / 0.1)


def _almgren_chriss_cost(vol: float, participation: float) -> float:
    # Simplified temporary + permanent impact proxy
    gamma, eta = 0.1, 0.3
    return float(gamma * participation + eta * participation * participation * max(vol, 1e-6))


def _black76_forward_call(fwd: float, k: float, vol: float, t: float, disc: float) -> Tuple[float, float, float, float]:
    if fwd <= 0 or k <= 0 or vol <= 0 or t <= 0:
        return 0.0, 0.0, 0.0, 0.0
    srt = vol * sqrt(t)
    d1 = (log(fwd / k) + 0.5 * vol * vol * t) / srt
    d2 = d1 - srt
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    pdf_d1 = np.exp(-0.5 * d1 * d1) / sqrt(2 * np.pi)
    price = disc * (fwd * nd1 - k * nd2)
    delta = disc * nd1
    gamma = disc * pdf_d1 / (fwd * srt)
    vega = disc * fwd * pdf_d1 * sqrt(t)
    return float(price), float(delta), float(gamma), float(vega)


def compute_quant_stack(instruments: Dict[str, pd.DataFrame]) -> QuantStackResult:
    """Compute a broad, no-cost quant score and diagnostics."""
    wti = _extract_series(instruments, 'WTI')
    brent = _extract_series(instruments, 'BRENT')
    xle = _extract_series(instruments, 'XLE')
    natgas = _extract_series(instruments, 'NATGAS')

    if wti is None or len(wti) < 40:
        return QuantStackResult(0.0, {'status': 'insufficient_data'})

    returns = np.log(wti / wti.shift(1)).dropna().values
    spot = float(wti.iloc[-1])
    fwd_proxy = float(brent.iloc[-1]) if brent is not None and len(brent) else spot
    t_30 = _act360_yearfrac(30)

    # No-arbitrage pricing block
    rate, storage = 0.05, 0.04
    conv = _convenience_yield(spot, fwd_proxy, rate, storage, t_30)
    carry_fair = _cost_of_carry(spot, rate, storage, conv, t_30)
    lb, ub = _storage_arb_bounds(spot, rate, storage, t_30)
    carry_mispricing = (fwd_proxy - carry_fair) / max(spot, 1e-8)

    # Stochastic process block
    gbm_mu, gbm_sigma = _gbm_params(returns)
    ou_kappa = _ou_speed(np.log(wti.values))
    schwartz_1f = ou_kappa
    schwartz_2f = 0.5 * ou_kappa + 0.5 * _safe_std(np.diff(np.log(wti.values)))
    schwartz_smith_3f = schwartz_2f + 0.25 * _seasonality_strength(wti)
    jump_intensity, jump_size = _jump_metrics(returns)
    vg_proxy = _variance_gamma_proxy(returns)
    seasonality = _seasonality_strength(wti)

    # Term structure and state-space block
    matrix = []
    for s in (wti, brent, xle, natgas):
        if s is not None and len(s) >= len(wti) - 5:
            aligned = s.reindex(wti.index).ffill().bfill()
            matrix.append(np.log(aligned).diff().fillna(0.0).values)
    mat = np.column_stack(matrix) if len(matrix) >= 2 else np.column_stack([returns, returns])
    pca_var, pca_vec = _pca_first_component(mat)
    cointegration = _cointegration_proxy(
        np.log(wti.values[-min(len(wti), 200):]),
        np.log((brent if brent is not None else wti).values[-min(len(wti), 200):])
    )
    kalman_beta = _kalman_beta(
        np.log(wti.values[-120:]),
        np.log((xle if xle is not None else wti).values[-120:])
    )

    # Time-series and statistics block
    arima_next = _ar1_forecast(returns)
    sarima_next = 0.6 * arima_next + 0.4 * (returns[-5] if len(returns) > 5 else 0.0)
    garch_vol = _garch_vol(returns)
    egarch_skew = float(np.nanmean(np.sign(returns) * np.abs(returns)))
    regime_prob_high_vol = float(np.clip(garch_vol / max(_safe_std(returns), 1e-8), 0, 2) / 2)
    hurst = _hurst_exponent(np.log(wti.values))
    fbm_proxy = hurst - 0.5
    bayes_up = _bayesian_up_prob(returns[-80:])
    evt_tail = _evt_tail_risk(returns)

    # Spread/cross-commodity block
    crack_321 = (2.0 * spot + 1.0 * (float(xle.iloc[-1]) if xle is not None else spot)) / 3.0 - spot
    spark = (float(xle.iloc[-1]) if xle is not None else spot) - 7.5 * (float(natgas.iloc[-1]) if natgas is not None else spot / 20)
    dark = (float(xle.iloc[-1]) if xle is not None else spot) - 1.2 * (float((brent if brent is not None else wti).iloc[-1]))
    engle_granger = cointegration
    johansen_trace_proxy = abs(cointegration) * pca_var
    copula_tail = _copula_tail_dependence(
        returns[-100:],
        np.log((xle if xle is not None else wti) / (xle if xle is not None else wti).shift(1)).dropna().values[-100:]
    )

    # Optimization/control block
    linear_program_weight = np.clip((bayes_up - 0.5) * 2, -1, 1)
    bellman_inventory_value = float(np.mean(returns[-20:]) - 0.5 * garch_vol)
    hjb_control_proxy = linear_program_weight * (1 - regime_prob_high_vol)
    almgren_chriss = _almgren_chriss_cost(garch_vol, participation=0.08)
    kelly_fraction = float(np.clip((bayes_up - (1 - bayes_up)) / max(garch_vol * garch_vol, 1e-6), -1, 1))

    # Risk and greeks block
    vol_annual = max(garch_vol * sqrt(252), 0.05)
    call_px, delta, gamma, vega = _black76_forward_call(spot, spot, vol_annual, t_30, exp(-rate * t_30))
    dv01 = t_30 * spot * 1e-4
    cr01 = dv01 * 0.35
    var_95 = float(np.quantile(returns[-250:] if len(returns) >= 250 else returns, 0.05))
    cvar_95 = float(np.mean(returns[returns <= var_95])) if np.any(returns <= var_95) else var_95
    stress_loss = min(var_95 * 1.8, -0.001)
    liq_adj_risk = abs(var_95) * (1 + almgren_chriss)

    # Microstructure block (daily proxies)
    volume = instruments.get('WTI', pd.DataFrame()).get('Volume', pd.Series(dtype=float))
    avg_volume = float(pd.to_numeric(volume, errors='coerce').dropna().tail(20).mean()) if len(volume) else 0.0
    queue_birth_death = float(np.clip((avg_volume / 1_000_000.0), 0, 3) / 3)
    kyle_lambda = abs(float(np.nanmean(returns[-20:]))) / max(avg_volume, 1.0)
    roll_cost = -float(np.cov(returns[:-1], returns[1:])[0, 1]) if len(returns) > 30 else 0.0
    iv_svi_proxy = vol_annual
    iv_sabr_proxy = vol_annual * (1 + 0.25 * abs(vg_proxy))

    # ML block (lightweight no-dependency proxies)
    features = np.column_stack([
        returns[-120:],
        np.roll(returns[-120:], 1),
        np.roll(returns[-120:], 5),
    ])
    target = np.sign(np.roll(returns[-120:], -1))
    x = features[5:-1]
    y = target[5:-1]
    xtx = x.T @ x + 1e-2 * np.eye(x.shape[1])
    ridge_w = np.linalg.solve(xtx, x.T @ y)
    ridge_pred = float(np.dot(features[-1], ridge_w))
    lasso_pred = float(np.sign(ridge_pred) * max(abs(ridge_pred) - 0.02, 0.0))
    gb_pred = float(0.6 * ridge_pred + 0.4 * arima_next / max(_safe_std(returns), 1e-6))
    rf_pred = float(np.median([ridge_pred, lasso_pred, gb_pred]))
    neural_sde_proxy = float(0.7 * rf_pred + 0.3 * (gbm_mu / max(gbm_sigma, 1e-6)))
    q_learning_proxy = float(np.clip((bayes_up - 0.5) * (1 - regime_prob_high_vol) * 2, -1, 1))

    # Composite score (bounded)
    components = {
        'carry_mispricing': -_zscore(carry_mispricing, 0.03),
        'convenience_yield': _zscore(conv, 0.08),
        'gbm_sharpe': _zscore(gbm_mu / max(gbm_sigma, 1e-6), 0.8),
        'ou_mean_reversion': -_zscore(ou_kappa, 0.6),
        'jump_risk': -_zscore(jump_intensity * jump_size, 0.01),
        'pca_level_factor': _zscore(pca_var, 0.6),
        'cointegration': _zscore(cointegration, 0.02),
        'kalman_beta': _zscore(kalman_beta - 1.0, 0.3),
        'arima_signal': _zscore(arima_next, 0.01),
        'garch_risk': -_zscore(garch_vol, 0.03),
        'hurst_trend': _zscore(fbm_proxy, 0.2),
        'bayesian_prob': _zscore(bayes_up - 0.5, 0.25),
        'evt_tail': -_zscore(evt_tail, 0.02),
        'crack_spread': _zscore(crack_321 / max(spot, 1e-6), 0.06),
        'copula_tail': -_zscore(copula_tail - 1.0, 0.6),
        'kelly': _zscore(kelly_fraction, 1.0),
        'risk_adjusted': -_zscore(liq_adj_risk, 0.03),
        'kyle_lambda': -_zscore(kyle_lambda, 5e-8),
        'ml_ensemble': _zscore((ridge_pred + lasso_pred + gb_pred + rf_pred) / 4.0, 0.5),
        'q_learning': _zscore(q_learning_proxy, 0.7),
    }

    score = float(np.clip(np.mean(list(components.values())), -1.0, 1.0))

    diagnostics = {
        # No-arbitrage pricing
        'cost_of_carry_model': carry_fair,
        'convenience_yield': conv,
        'storage_arb_bounds': {'lower': lb, 'upper': ub},
        'day_count_act_360': t_30,
        # Stochastic processes
        'gbm': {'mu': gbm_mu, 'sigma': gbm_sigma},
        'ornstein_uhlenbeck_kappa': ou_kappa,
        'schwartz_1f': schwartz_1f,
        'schwartz_2f': schwartz_2f,
        'schwartz_smith_3f': schwartz_smith_3f,
        'jump_diffusion': {'intensity': jump_intensity, 'avg_jump': jump_size},
        'levy_variance_gamma_proxy': vg_proxy,
        'seasonal_decomposition_strength': seasonality,
        # Term structure math
        'forward_curve_bootstrap_proxy': {'t30': fwd_proxy},
        'cubic_spline_interpolation_proxy': float(np.interp(45, [30, 60], [spot, fwd_proxy])),
        'monotone_convex_interpolation_proxy': float(np.interp(15, [1, 30], [spot, fwd_proxy])),
        'kernel_smoothing_proxy': float(pd.Series(returns).rolling(5, min_periods=1).mean().iloc[-1]),
        'pca_explained_first': pca_var,
        'cointegration_proxy': cointegration,
        'vecm_proxy': cointegration,
        'state_space_proxy': {'kalman_beta': kalman_beta},
        'kalman_filter_beta': kalman_beta,
        # Time series
        'arima_proxy': arima_next,
        'sarima_proxy': sarima_next,
        'garch_vol': garch_vol,
        'egarch_proxy': egarch_skew,
        'markov_regime_high_vol_prob': regime_prob_high_vol,
        'hurst_exponent': hurst,
        'fractional_brownian_proxy': fbm_proxy,
        'bayesian_up_probability': bayes_up,
        'evt_tail_risk': evt_tail,
        # Spread and cross commodity
        'crack_3_2_1_proxy': crack_321,
        'spark_spread_proxy': spark,
        'dark_spread_proxy': dark,
        'engle_granger_proxy': engle_granger,
        'johansen_proxy': johansen_trace_proxy,
        'copula_tail_dependence': copula_tail,
        # Optimization/control
        'linear_programming_weight': linear_program_weight,
        'dynamic_programming_bellman': bellman_inventory_value,
        'hjb_proxy': hjb_control_proxy,
        'almgren_chriss_cost': almgren_chriss,
        'kelly_fractional': kelly_fraction,
        # Risk and Greeks
        'black76_call': call_px,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'dv01': dv01,
        'cr01': cr01,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'stress_loss': stress_loss,
        'liquidity_adjusted_risk': liq_adj_risk,
        # Microstructure
        'queue_birth_death_proxy': queue_birth_death,
        'kyle_lambda': kyle_lambda,
        'roll_cost': roll_cost,
        'iv_surface_svi_proxy': iv_svi_proxy,
        'iv_surface_sabr_proxy': iv_sabr_proxy,
        # Machine learning
        'ridge_pred': ridge_pred,
        'lasso_pred': lasso_pred,
        'gradient_boosting_proxy': gb_pred,
        'random_forest_proxy': rf_pred,
        'neural_sde_proxy': neural_sde_proxy,
        'q_learning_proxy': q_learning_proxy,
        # Composite
        'component_scores': components,
    }

    return QuantStackResult(score=score, diagnostics=diagnostics)
