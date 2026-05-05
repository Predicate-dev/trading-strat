"""Trading strategy interfaces and vectorized implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

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


@dataclass(slots=True)
class AdaptiveTrendStrategy(BaseStrategy):
    """Breakout/trend-following strategy designed to stay with strong upside trends."""

    breakout_window: int = 55
    exit_window: int = 21
    fast_ema: int = 20
    slow_ema: int = 50
    long_ema: int = 100
    volatility_target: float | None = 0.35
    max_exposure: float = 1.0
    regime_column: str | None = "regime"
    allowed_regimes: Iterable[str] = ("Bull Trend", "Sideways")
    allow_bullish_high_volatility: bool = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns:
            raise ValueError("AdaptiveTrendStrategy requires a 'close' column.")

        close = data["close"].astype(float)
        fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        slow = close.ewm(span=self.slow_ema, adjust=False).mean()
        long = close.ewm(span=self.long_ema, adjust=False).mean()
        prior_breakout = close.rolling(self.breakout_window).max().shift(1)
        prior_exit = close.rolling(self.exit_window).min().shift(1)

        trend_ok = fast.gt(slow) & slow.gt(long)
        breakout = close.gt(prior_breakout)
        pullback_hold = close.gt(slow) & close.gt(prior_exit)
        desired_long = (breakout | (trend_ok & pullback_hold)).fillna(False)

        if self.regime_column is not None and self.regime_column in data.columns:
            allowed = data[self.regime_column].isin(tuple(self.allowed_regimes))
            if self.allow_bullish_high_volatility:
                bullish_high_vol = data[self.regime_column].eq("High Volatility") & trend_ok & close.gt(slow)
                allowed = allowed | bullish_high_vol
            desired_long &= allowed

        signal = pd.Series(0.0, index=data.index, name="signal")
        signal = signal.mask(desired_long, 1.0)
        signal = self._hold_until_exit(signal, close, slow, prior_exit)

        if self.volatility_target is not None and "rolling_volatility" in data.columns:
            vol = data["rolling_volatility"].replace(0.0, np.nan)
            exposure = (self.volatility_target / vol).clip(upper=self.max_exposure)
            signal = signal.mul(exposure).fillna(0.0)
        return signal.clip(0.0, self.max_exposure).fillna(0.0)

    @staticmethod
    def _hold_until_exit(entry_signal: pd.Series, close: pd.Series, slow: pd.Series, prior_exit: pd.Series) -> pd.Series:
        held = np.zeros(len(entry_signal), dtype=float)
        active = False
        for idx, wants_entry in enumerate(entry_signal.to_numpy(dtype=float) > 0.0):
            exit_now = close.iloc[idx] < slow.iloc[idx] or close.iloc[idx] < prior_exit.iloc[idx]
            if wants_entry:
                active = True
            elif active and exit_now:
                active = False
            held[idx] = 1.0 if active else 0.0
        return pd.Series(held, index=entry_signal.index, name="signal")


@dataclass(slots=True)
class MLSignalStrategy(BaseStrategy):
    """Convert ML prediction columns into bounded market exposure."""

    prediction_column: str = "predicted_return"
    probability_column: str = "probability_up"
    confidence_column: str = "confidence"
    long_threshold: float = 0.02
    short_threshold: float = 0.02
    min_confidence: float = 0.10
    allow_short: bool = False
    volatility_target: float | None = 0.20
    regime_column: str | None = "regime"
    allowed_regimes: Iterable[str] = ("Bull Trend", "Sideways")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if self.prediction_column not in data.columns:
            raise ValueError(f"MLSignalStrategy missing column: {self.prediction_column}")

        prediction = data[self.prediction_column].fillna(0.0)
        confidence = data[self.confidence_column].fillna(0.0) if self.confidence_column in data.columns else prediction.abs()
        long_condition = prediction.gt(self.long_threshold) & confidence.ge(self.min_confidence)
        short_condition = prediction.lt(-self.short_threshold) & confidence.ge(self.min_confidence)

        if self.regime_column is not None and self.regime_column in data.columns:
            allowed = data[self.regime_column].isin(tuple(self.allowed_regimes))
            long_condition &= allowed
            short_condition &= allowed

        signal = pd.Series(0.0, index=data.index, name="signal")
        signal = signal.mask(long_condition, 1.0)
        if self.allow_short:
            signal = signal.mask(short_condition, -1.0)

        signal = signal * confidence.clip(0.0, 1.0)
        if self.volatility_target is not None and "rolling_volatility" in data.columns:
            vol = data["rolling_volatility"].replace(0.0, np.nan)
            signal = signal.mul((self.volatility_target / vol).clip(upper=1.0)).fillna(0.0)
        return signal.clip(-1.0, 1.0).fillna(0.0)


@dataclass(slots=True)
class EnsembleStrategy(BaseStrategy):
    """Blend rule-based and ML strategies into one exposure stream."""

    strategies: Sequence[BaseStrategy]
    weights: Sequence[float] | None = None
    clip: float = 1.0

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.strategies:
            raise ValueError("EnsembleStrategy requires at least one child strategy.")

        weights = np.asarray(self.weights if self.weights is not None else [1.0] * len(self.strategies), dtype=float)
        if len(weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies.")
        if np.isclose(np.abs(weights).sum(), 0.0):
            raise ValueError("Strategy weights cannot sum to zero.")
        weights = weights / np.abs(weights).sum()

        blended = pd.Series(0.0, index=data.index, name="signal")
        for strategy, weight in zip(self.strategies, weights, strict=False):
            blended = blended.add(strategy.generate_signals(data).reindex(data.index).fillna(0.0) * weight, fill_value=0.0)
        return blended.clip(-self.clip, self.clip).fillna(0.0)
