"""Trading strategy interfaces and vectorized implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """Strategy interface.

    Strategies emit desired market exposure for each bar:
    1.0 is long, -1.0 is short, and 0.0 is flat. The backtester shifts signals
    before execution to prevent look-ahead bias.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate raw unshifted strategy signals."""


@dataclass(slots=True)
class MeanReversionStrategy(BaseStrategy):
    """Buy oversold pullbacks and sell overbought extensions."""

    lower_rsi: float = 30.0
    upper_rsi: float = 70.0
    use_bollinger_filter: bool = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self._validate_columns(data)
        long_condition = data["rsi"].lt(self.lower_rsi)
        short_condition = data["rsi"].gt(self.upper_rsi)

        if self.use_bollinger_filter:
            long_condition &= data["close"].lt(data["bb_lower"])
            short_condition &= data["close"].gt(data["bb_upper"])

        signals = pd.Series(
            np.select([long_condition, short_condition], [1.0, -1.0], default=0.0),
            index=data.index,
            name="signal",
        )
        return signals.ffill().fillna(0.0)

    @staticmethod
    def _validate_columns(data: pd.DataFrame) -> None:
        required = {"close", "rsi", "bb_lower", "bb_upper"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"MeanReversionStrategy missing columns: {sorted(missing)}")


@dataclass(slots=True)
class MomentumStrategy(BaseStrategy):
    """Trade in the direction of trend and MACD confirmation."""

    lookback: int = 63
    volatility_target: float | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self._validate_columns(data)
        trend_return = data["close"].pct_change(self.lookback)
        long_condition = trend_return.gt(0.0) & data["macd"].gt(data["macd_signal"])
        short_condition = trend_return.lt(0.0) & data["macd"].lt(data["macd_signal"])

        signal = pd.Series(
            np.select([long_condition, short_condition], [1.0, -1.0], default=0.0),
            index=data.index,
            name="signal",
        )

        if self.volatility_target is not None:
            vol = data["rolling_volatility"].replace(0.0, np.nan)
            leverage = (self.volatility_target / vol).clip(upper=1.0)
            signal = signal.mul(leverage).fillna(0.0)

        return signal.fillna(0.0)

    @staticmethod
    def _validate_columns(data: pd.DataFrame) -> None:
        required = {"close", "macd", "macd_signal", "rolling_volatility"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"MomentumStrategy missing columns: {sorted(missing)}")


@dataclass(slots=True)
class RegimeAwareMomentumStrategy(BaseStrategy):
    """Momentum strategy gated by market regime labels."""

    lookback: int = 63
    volatility_target: float | None = 0.20
    regime_column: str = "regime"
    long_regimes: Iterable[str] = ("Bull Trend",)
    hedge_high_volatility: bool = False

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self._validate_columns(data)
        trend_return = data["close"].pct_change(self.lookback)
        momentum_long = trend_return.gt(0.0) & data["macd"].gt(data["macd_signal"])
        bull_regime = data[self.regime_column].isin(tuple(self.long_regimes))
        high_volatility = data[self.regime_column].eq("High Volatility")

        signal = pd.Series(np.where(momentum_long & bull_regime, 1.0, 0.0), index=data.index, name="signal")
        if self.hedge_high_volatility:
            signal = signal.mask(high_volatility, -0.25)
        else:
            signal = signal.mask(high_volatility, 0.0)

        if self.volatility_target is not None:
            vol = data["rolling_volatility"].replace(0.0, np.nan)
            leverage = (self.volatility_target / vol).clip(upper=1.0)
            signal = signal.mul(leverage).fillna(0.0)

        return signal.fillna(0.0)

    def _validate_columns(self, data: pd.DataFrame) -> None:
        required = {"close", "macd", "macd_signal", "rolling_volatility", self.regime_column}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"RegimeAwareMomentumStrategy missing columns: {sorted(missing)}")
