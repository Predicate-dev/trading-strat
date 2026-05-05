"""Vectorized research backtesting engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from trading_strat.slippage import SlippageModel
from trading_strat.strategies import BaseStrategy

LOGGER = logging.getLogger(__name__)

SizingMode = Literal["fractional", "fixed"]


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Execution and portfolio assumptions for a backtest."""

    initial_capital: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    slippage_model: SlippageModel | None = None
    sizing_mode: SizingMode = "fractional"
    fractional_exposure: float = 1.0
    fixed_notional: float = 10_000.0
    atr_stop_multiple: float | None = 3.0
    atr_column: str = "atr"
    take_profit: float | None = 0.16
    allow_short: bool = True
    annualization_factor: int = 252


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Backtest outputs aligned on the input price index."""

    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.Series
    trade_returns: pd.Series
    drawdown: pd.Series
    stats: dict[str, float]


class VectorizedBacktester:
    """Backtest vectorized strategy signals with realistic trading frictions.

    Signal generation, returns, turnover, costs, and statistics are vectorized.
    Stop-loss and take-profit handling is path-dependent, so the engine applies
    those exits in a compact NumPy loop after shifting signals by one bar.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, data: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        """Execute a strategy on OHLCV/features data."""
        if "close" not in data.columns:
            raise ValueError("Backtester requires a 'close' column.")
        if data.empty:
            raise ValueError("Cannot backtest an empty data frame.")

        try:
            prices = data["close"].astype(float)
            raw_signals = strategy.generate_signals(data).reindex(prices.index).fillna(0.0)
            executable_signals = self._prepare_executable_signals(raw_signals)
            positions = self._apply_exit_logic(data, executable_signals)
            weights = self._position_weights(prices, positions)
            returns = self._calculate_net_returns(prices, weights)
            equity_curve = self.config.initial_capital * (1.0 + returns).cumprod()
            drawdown = self._calculate_drawdown(equity_curve)
            trades = weights.diff().abs().fillna(weights.abs())
            trade_returns = self._trade_returns(prices, weights)
            stats = self._calculate_stats(returns, equity_curve, drawdown, trades, trade_returns)
            return BacktestResult(
                equity_curve=equity_curve.rename("equity"),
                returns=returns.rename("strategy_return"),
                positions=weights.rename("position"),
                trades=trades.rename("turnover"),
                trade_returns=trade_returns,
                drawdown=drawdown.rename("drawdown"),
                stats=stats,
            )
        except Exception:
            LOGGER.exception("Backtest failed.")
            raise

    def _prepare_executable_signals(self, raw_signals: pd.Series) -> pd.Series:
        signals = raw_signals.clip(lower=-1.0, upper=1.0)
        if not self.config.allow_short:
            signals = signals.clip(lower=0.0)
        return signals.shift(1).fillna(0.0)

    def _position_weights(self, prices: pd.Series, signals: pd.Series) -> pd.Series:
        if self.config.sizing_mode == "fractional":
            weights = signals * self.config.fractional_exposure
        elif self.config.sizing_mode == "fixed":
            capital_fraction = self.config.fixed_notional / self.config.initial_capital
            weights = signals * capital_fraction
        else:
            raise ValueError(f"Unsupported sizing mode: {self.config.sizing_mode}")

        return weights.reindex(prices.index).fillna(0.0)

    def _calculate_net_returns(self, prices: pd.Series, weights: pd.Series) -> pd.Series:
        asset_returns = prices.pct_change().fillna(0.0)
        gross_returns = weights.fillna(0.0) * asset_returns
        turnover = weights.diff().abs().fillna(weights.abs())
        if self.config.slippage_model is not None:
            cost_rate = self.config.slippage_model.cost_rate(prices, turnover)
        else:
            cost_rate = pd.Series(
                (self.config.commission_bps + self.config.slippage_bps) / 10_000.0,
                index=prices.index,
            )
        costs = turnover * cost_rate
        return (gross_returns - costs).fillna(0.0)

    def _apply_exit_logic(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        prices = data["close"].astype(float)
        if self.config.atr_stop_multiple is None and self.config.take_profit is None:
            return signals
        if self.config.atr_stop_multiple is not None and self.config.atr_column not in data.columns:
            raise ValueError(f"ATR trailing stop requires '{self.config.atr_column}' column.")

        price_array = prices.to_numpy(dtype=float)
        atr_array = (
            data[self.config.atr_column].ffill().fillna(0.0).to_numpy(dtype=float)
            if self.config.atr_stop_multiple is not None
            else np.zeros_like(price_array)
        )
        signal_array = signals.to_numpy(dtype=float)
        position_array = np.zeros_like(signal_array)
        active_position = 0.0
        entry_price = np.nan
        trailing_stop = np.nan

        for idx, desired_position in enumerate(signal_array):
            current_price = price_array[idx]

            if active_position != 0.0 and not np.isnan(entry_price):
                if self.config.atr_stop_multiple is not None and atr_array[idx] > 0.0:
                    stop_distance = self.config.atr_stop_multiple * atr_array[idx]
                    if active_position > 0.0:
                        candidate_stop = current_price - stop_distance
                        trailing_stop = candidate_stop if np.isnan(trailing_stop) else max(trailing_stop, candidate_stop)
                    else:
                        candidate_stop = current_price + stop_distance
                        trailing_stop = candidate_stop if np.isnan(trailing_stop) else min(trailing_stop, candidate_stop)

                pnl = active_position * ((current_price / entry_price) - 1.0)
                stopped = False
                if self.config.atr_stop_multiple is not None and not np.isnan(trailing_stop):
                    stopped = current_price <= trailing_stop if active_position > 0.0 else current_price >= trailing_stop
                profit_taken = self.config.take_profit is not None and pnl >= self.config.take_profit
                if stopped or profit_taken:
                    active_position = 0.0
                    entry_price = np.nan
                    trailing_stop = np.nan
                    position_array[idx] = active_position
                    continue

            if desired_position != active_position:
                active_position = desired_position
                entry_price = current_price if active_position != 0.0 else np.nan
                if self.config.atr_stop_multiple is None or active_position == 0.0:
                    trailing_stop = np.nan
                elif active_position > 0.0 and atr_array[idx] > 0.0:
                    trailing_stop = current_price - self.config.atr_stop_multiple * atr_array[idx]
                elif active_position < 0.0 and atr_array[idx] > 0.0:
                    trailing_stop = current_price + self.config.atr_stop_multiple * atr_array[idx]
                else:
                    trailing_stop = np.nan

            position_array[idx] = active_position

        return pd.Series(position_array, index=signals.index, name="position_signal")

    def _calculate_stats(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        drawdown: pd.Series,
        trades: pd.Series,
        trade_returns: pd.Series,
    ) -> dict[str, float]:
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
        years = max(len(returns) / self.config.annualization_factor, 1 / self.config.annualization_factor)
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / years) - 1.0
        volatility = returns.std(ddof=0) * np.sqrt(self.config.annualization_factor)
        sharpe = np.nan if volatility == 0.0 else returns.mean() / returns.std(ddof=0) * np.sqrt(
            self.config.annualization_factor
        )
        downside = returns[returns < 0.0].std(ddof=0) * np.sqrt(self.config.annualization_factor)
        sortino = np.nan if downside == 0.0 else returns.mean() * self.config.annualization_factor / downside
        max_drawdown = abs(float(drawdown.min()))
        calmar = np.nan if max_drawdown == 0.0 else cagr / max_drawdown
        win_rate = np.nan if trade_returns.empty else float((trade_returns > 0.0).mean())

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "annualized_volatility": float(volatility),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "max_drawdown": float(drawdown.min()),
            "win_rate": float(win_rate),
            "number_of_trades": float((trades > 0.0).sum()),
            "average_daily_turnover": float(trades.mean()),
        }

    @staticmethod
    def _calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
        running_max = equity_curve.cummax()
        return equity_curve / running_max - 1.0

    @staticmethod
    def _trade_returns(prices: pd.Series, weights: pd.Series) -> pd.Series:
        trade_returns: list[float] = []
        active_direction = 0.0
        entry_price = np.nan

        for price, weight in zip(prices.to_numpy(dtype=float), weights.to_numpy(dtype=float), strict=False):
            direction = float(np.sign(weight))
            if active_direction == 0.0 and direction != 0.0:
                active_direction = direction
                entry_price = price
            elif active_direction != 0.0 and direction != active_direction:
                trade_returns.append(active_direction * (price / entry_price - 1.0))
                active_direction = direction
                entry_price = price if direction != 0.0 else np.nan

        if active_direction != 0.0 and not np.isnan(entry_price):
            trade_returns.append(active_direction * (prices.iloc[-1] / entry_price - 1.0))

        return pd.Series(trade_returns, name="trade_return", dtype=float)
