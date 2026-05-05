"""Vectorized feature engineering and alpha-factor calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureGenerator:
    """Generate reusable alpha factors from OHLCV data."""

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    volatility_window: int = 20
    atr_period: int = 14
    annualization_factor: int = 252

    def add_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return OHLCV data with RSI, MACD, Bollinger Bands, ATR, and volatility."""
        features = data.copy()
        features["return"] = features["close"].pct_change()
        features["log_return"] = np.log(features["close"]).diff()
        features = self.add_rsi(features)
        features = self.add_macd(features)
        features = self.add_bollinger_bands(features)
        features = self.add_atr(features)
        features = self.add_rolling_volatility(features)
        return features

    def add_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        delta = result["close"].diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        avg_gain = gains.ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        result["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        result["rsi"] = result["rsi"].fillna(50.0)
        return result

    def add_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        fast_ema = result["close"].ewm(span=self.macd_fast, adjust=False).mean()
        slow_ema = result["close"].ewm(span=self.macd_slow, adjust=False).mean()
        result["macd"] = fast_ema - slow_ema
        result["macd_signal"] = result["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        result["macd_hist"] = result["macd"] - result["macd_signal"]
        return result

    def add_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        rolling_mean = result["close"].rolling(self.bollinger_window).mean()
        rolling_std = result["close"].rolling(self.bollinger_window).std(ddof=0)
        result["bb_middle"] = rolling_mean
        result["bb_upper"] = rolling_mean + self.bollinger_std * rolling_std
        result["bb_lower"] = rolling_mean - self.bollinger_std * rolling_std
        result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / rolling_mean
        result["bb_percent"] = (result["close"] - result["bb_lower"]) / (result["bb_upper"] - result["bb_lower"])
        return result

    def add_rolling_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        returns = result["close"].pct_change()
        result["rolling_volatility"] = returns.rolling(self.volatility_window).std(ddof=0) * np.sqrt(
            self.annualization_factor
        )
        return result

    def add_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        previous_close = result["close"].shift(1)
        true_range = pd.concat(
            [
                result["high"] - result["low"],
                (result["high"] - previous_close).abs(),
                (result["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        result["atr"] = true_range.ewm(alpha=1 / self.atr_period, min_periods=self.atr_period, adjust=False).mean()
        result["atr_percent"] = result["atr"] / result["close"]
        return result
