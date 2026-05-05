"""Vectorized feature engineering, alpha factors, and target construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

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
    lagged_return_windows: tuple[int, ...] = (1, 2, 5, 10)
    rolling_return_windows: tuple[int, ...] = (5, 10, 21, 63)
    momentum_windows: tuple[int, ...] = (21, 63, 126)
    volatility_windows: tuple[int, ...] = (10, 21, 63)

    def add_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return OHLCV data with technical indicators and reusable alpha factors."""
        return self.add_alpha_factors(data)

    def add_alpha_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add current/past-only features for single-symbol or MultiIndex data."""
        if isinstance(data.index, pd.MultiIndex):
            frames = []
            for symbol, frame in data.groupby(level="symbol", sort=False):
                enriched = self._add_single_symbol_factors(frame.droplevel("symbol"))
                enriched["symbol"] = symbol
                frames.append(enriched)
            result = pd.concat(frames).set_index("symbol", append=True).sort_index()
            result.index = result.index.set_names(["date", "symbol"])
            return self.add_cross_sectional_ranks(result)
        return self._add_single_symbol_factors(data)

    def _add_single_symbol_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        features = data.copy().sort_index()
        features["return"] = features["close"].pct_change()
        features["log_return"] = np.log(features["close"]).diff()
        features = self.add_rsi(features)
        features = self.add_macd(features)
        features = self.add_bollinger_bands(features)
        features = self.add_atr(features)
        features = self.add_rolling_volatility(features)
        features = self.add_general_alpha_factors(features)
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

    def add_general_alpha_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        returns = result["close"].pct_change()
        log_returns = np.log(result["close"]).diff()

        for window in self.lagged_return_windows:
            result[f"return_lag_{window}"] = returns.shift(window)
        for window in self.rolling_return_windows:
            result[f"rolling_return_{window}"] = result["close"].pct_change(window)
        for window in self.momentum_windows:
            result[f"momentum_{window}"] = result["close"] / result["close"].shift(window) - 1.0
            rolling_max = result["close"].rolling(window).max()
            result[f"drawdown_{window}"] = result["close"] / rolling_max - 1.0
            moving_average = result["close"].rolling(window).mean()
            result[f"trend_strength_{window}"] = (result["close"] - moving_average) / moving_average
        for window in self.volatility_windows:
            rolling_vol = returns.rolling(window).std(ddof=0) * np.sqrt(self.annualization_factor)
            result[f"volatility_{window}"] = rolling_vol
            result[f"price_zscore_{window}"] = self._zscore(result["close"], window)
            result[f"return_skew_{window}"] = log_returns.rolling(window).skew()
            result[f"return_kurtosis_{window}"] = log_returns.rolling(window).kurt()

        result["volume_change"] = result["volume"].pct_change()
        result["volume_zscore"] = self._zscore(result["volume"].replace(0.0, np.nan), self.volatility_window)
        result["dollar_volume"] = result["close"] * result["volume"]
        return result.replace([np.inf, -np.inf], np.nan)

    def add_cross_sectional_ranks(
        self,
        data: pd.DataFrame,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Add per-date cross-sectional percentile ranks for multi-symbol frames."""
        if not isinstance(data.index, pd.MultiIndex):
            return data
        result = data.copy()
        default_columns = [
            "return",
            "rolling_return_21",
            "momentum_63",
            "volatility_21",
            "atr_percent",
            "volume_zscore",
        ]
        rank_columns = [column for column in (columns or default_columns) if column in result.columns]
        for column in rank_columns:
            result[f"cs_rank_{column}"] = result.groupby(level="date")[column].rank(pct=True)
        return result

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std(ddof=0)
        return (series - rolling_mean) / rolling_std.replace(0.0, np.nan)


@dataclass(slots=True)
class TargetBuilder:
    """Build forward-return and classification targets without feature leakage."""

    horizons: tuple[int, ...] = (1, 5, 10)
    classification_threshold: float = 0.0
    drop_last_horizon_rows: bool = True
    target_columns: list[str] = field(default_factory=list)

    def add_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data.index, pd.MultiIndex):
            frames = []
            for symbol, frame in data.groupby(level="symbol", sort=False):
                labeled = self._add_single_symbol_targets(frame.droplevel("symbol"))
                labeled["symbol"] = symbol
                frames.append(labeled)
            result = pd.concat(frames).set_index("symbol", append=True).sort_index()
            result.index = result.index.set_names(["date", "symbol"])
            return result
        return self._add_single_symbol_targets(data)

    def _add_single_symbol_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy().sort_index()
        generated: list[str] = []
        for horizon in self.horizons:
            future_return = result["close"].shift(-horizon) / result["close"] - 1.0
            return_col = f"forward_return_{horizon}"
            label_col = f"target_up_{horizon}"
            result[return_col] = future_return
            result[label_col] = (future_return > self.classification_threshold).astype(float)
            result.loc[future_return.isna(), label_col] = np.nan
            generated.extend([return_col, label_col])
        self.target_columns = generated
        if self.drop_last_horizon_rows and self.horizons:
            result = result.iloc[: -max(self.horizons)]
        return result
