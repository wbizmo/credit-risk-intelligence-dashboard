from __future__ import annotations

import numpy as np


def economic_lgd(
    ead: np.ndarray,
    recoveries: np.ndarray,
    recovery_costs: np.ndarray,
    *,
    discount_factor: float | np.ndarray = 1.0,
) -> np.ndarray:
    exposure = np.asarray(ead, dtype=float)
    recovered = np.asarray(recoveries, dtype=float)
    costs = np.asarray(recovery_costs, dtype=float)
    discount = np.asarray(discount_factor, dtype=float)
    try:
        recovered, costs, discount = np.broadcast_arrays(recovered, costs, discount)
        exposure, recovered = np.broadcast_arrays(exposure, recovered)
    except ValueError as error:
        raise ValueError("LGD inputs must be broadcast-compatible") from error
    if not all(np.isfinite(value).all() for value in (exposure, recovered, costs, discount)):
        raise ValueError("LGD inputs must be finite")
    if (exposure <= 0).any():
        raise ValueError("EAD must be greater than zero for LGD")
    if (recovered < 0).any() or (costs < 0).any() or (discount < 0).any():
        raise ValueError("recoveries, recovery costs and discount factors must be non-negative")
    net_recovery = (recovered - costs) * discount
    return (exposure - net_recovery) / exposure
