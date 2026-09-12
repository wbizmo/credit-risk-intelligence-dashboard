from __future__ import annotations

import numpy as np


def amortizing_balance(principal: float, annual_rate: float, term_months: int, elapsed_months: int) -> float:
    values = (principal, annual_rate)
    if not all(np.isfinite(value) for value in values) or principal < 0 or term_months <= 0 or elapsed_months < 0:
        raise ValueError("invalid amortization inputs")
    if elapsed_months == 0:
        return float(principal)
    if elapsed_months >= term_months or principal == 0:
        return 0.0
    monthly_rate = annual_rate / 12.0
    if abs(monthly_rate) < 1e-15:
        return float(principal * (term_months - elapsed_months) / term_months)
    if monthly_rate <= -1:
        raise ValueError("monthly interest rate must be greater than -100%")
    growth = (1 + monthly_rate) ** term_months
    payment = principal * monthly_rate * growth / (growth - 1)
    elapsed_growth = (1 + monthly_rate) ** elapsed_months
    balance = principal * elapsed_growth - payment * (elapsed_growth - 1) / monthly_rate
    return float(max(balance, 0.0))


def observed_installment_ead(funded_amount: np.ndarray, principal_received: np.ndarray) -> np.ndarray:
    funded = np.asarray(funded_amount, dtype=float)
    received = np.asarray(principal_received, dtype=float)
    if funded.shape != received.shape or not np.isfinite(funded).all() or not np.isfinite(received).all():
        raise ValueError("funded amount and principal received must be finite and aligned")
    if (funded < 0).any() or (received < 0).any():
        raise ValueError("funded amount and principal received must be non-negative")
    return np.maximum(funded - received, 0.0)


def ccf_proxy(limit: np.ndarray, prior_balance: np.ndarray, next_balance: np.ndarray) -> np.ndarray:
    credit_limit = np.asarray(limit, dtype=float)
    prior = np.asarray(prior_balance, dtype=float)
    following = np.asarray(next_balance, dtype=float)
    if credit_limit.shape != prior.shape or prior.shape != following.shape:
        raise ValueError("CCF proxy inputs must align")
    if not np.isfinite(credit_limit).all() or not np.isfinite(prior).all() or not np.isfinite(following).all():
        raise ValueError("CCF proxy inputs must be finite")
    undrawn = credit_limit - prior
    result = np.full(credit_limit.shape, np.nan, dtype=float)
    valid = undrawn > 0
    result[valid] = (following[valid] - prior[valid]) / undrawn[valid]
    return result
