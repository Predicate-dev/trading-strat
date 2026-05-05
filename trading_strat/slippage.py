"""Transaction cost and spread models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SlippageModel:
    """Simulate a 1-2 tick spread plus variable commission."""

    tick_size: float = 0.01
    min_spread_ticks: int = 1
    max_spread_ticks: int = 2
    base_commission_bps: float = 0.5
    variable_commission_bps: float = 0.5
    volatility_window: int = 20

    def cost_rate(self, prices: pd.Series, turnover: pd.Series) -> pd.Series:
        """Return cost as a fraction of traded notional for each bar."""
        clean_prices = prices.astype(float).replace(0.0, np.nan)
        abs_returns = clean_prices.pct_change().abs()
        vol = abs_returns.rolling(self.volatility_window).mean()
        high_vol = vol.gt(vol.rolling(self.volatility_window * 3, min_periods=self.volatility_window).median())
        spread_ticks = pd.Series(
            np.where(high_vol.fillna(False), self.max_spread_ticks, self.min_spread_ticks),
            index=prices.index,
            dtype=float,
        )
        spread_bps = (spread_ticks * self.tick_size / clean_prices) * 10_000.0
        commission_bps = self.base_commission_bps + self.variable_commission_bps * turnover.clip(lower=0.0)
        return ((spread_bps + commission_bps) / 10_000.0).fillna(0.0)
