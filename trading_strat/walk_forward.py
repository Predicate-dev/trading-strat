"""Walk-forward validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from trading_strat.backtester import BacktestResult, VectorizedBacktester
from trading_strat.strategies import BaseStrategy

StrategyFactory = Callable[[pd.DataFrame], BaseStrategy]


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    fold: int


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    fold_results: list[BacktestResult]
    combined_returns: pd.Series
    stats: dict[str, float]


class WalkForwardValidator:
    """Evaluate strategies on rolling train/test windows."""

    def __init__(
        self,
        train_size: int = 504,
        test_size: int = 126,
        step_size: int | None = None,
        backtester: VectorizedBacktester | None = None,
    ) -> None:
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.backtester = backtester or VectorizedBacktester()

    def split(self, data: pd.DataFrame) -> list[WalkForwardSplit]:
        splits: list[WalkForwardSplit] = []
        fold = 0
        start = 0
        while start + self.train_size + self.test_size <= len(data):
            train = data.iloc[start : start + self.train_size]
            test = data.iloc[start + self.train_size : start + self.train_size + self.test_size]
            splits.append(WalkForwardSplit(train=train, test=test, fold=fold))
            fold += 1
            start += self.step_size
        return splits

    def evaluate(self, data: pd.DataFrame, strategy_factory: StrategyFactory) -> WalkForwardResult:
        fold_results: list[BacktestResult] = []
        returns: list[pd.Series] = []
        for split in self.split(data):
            strategy = strategy_factory(split.train)
            result = self.backtester.run(split.test, strategy)
            fold_results.append(result)
            returns.append(result.returns)

        if not returns:
            raise ValueError("No walk-forward splits were produced. Increase data length or reduce split sizes.")

        combined_returns = pd.concat(returns).sort_index()
        equity_curve = self.backtester.config.initial_capital * (1.0 + combined_returns).cumprod()
        drawdown = equity_curve / equity_curve.cummax() - 1.0
        trades = pd.concat([result.trades for result in fold_results]).sort_index()
        trade_returns = pd.concat([result.trade_returns for result in fold_results], ignore_index=True)
        stats = self.backtester._calculate_stats(combined_returns, equity_curve, drawdown, trades, trade_returns)
        return WalkForwardResult(fold_results=fold_results, combined_returns=combined_returns, stats=stats)
