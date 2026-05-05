"""Optuna walk-forward optimization for regime-aware trading."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

import numpy as np
import optuna
import pandas as pd

from trading_strat import (
    BacktestConfig,
    DataHandler,
    FeatureGenerator,
    RegimeAwareMomentumStrategy,
    RegimeFilter,
    SlippageModel,
    VectorizedBacktester,
)
from trading_strat.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    symbol: str = "AAPL"
    years: int = 5
    train_size: int = 504
    test_size: int = 126
    n_trials: int = 30
    random_state: int = 42


def add_walk_forward_regimes(
    features: pd.DataFrame,
    train_size: int,
    test_size: int,
    regime_lookback: int,
    random_state: int,
) -> list[pd.DataFrame]:
    """Create out-of-sample test folds with regimes predicted from train-fitted GMMs."""
    folds: list[pd.DataFrame] = []
    start = 0
    while start + train_size + test_size <= len(features):
        train = features.iloc[start : start + train_size].copy()
        test = features.iloc[start + train_size : start + train_size + test_size].copy()
        context = pd.concat([train.tail(regime_lookback * 2), test])

        regime_filter = RegimeFilter(lookback=regime_lookback, random_state=random_state).fit(train)
        context["regime"] = regime_filter.predict(context)
        folds.append(context.loc[test.index].copy())
        start += test_size

    if not folds:
        raise ValueError("No walk-forward folds produced. Increase history or reduce train/test windows.")
    return folds


def run_fold_backtests(
    folds: list[pd.DataFrame],
    strategy: RegimeAwareMomentumStrategy,
    config: BacktestConfig,
) -> dict[str, float]:
    backtester = VectorizedBacktester(config)
    returns: list[pd.Series] = []
    trades: list[pd.Series] = []
    trade_returns: list[pd.Series] = []

    for fold in folds:
        result = backtester.run(fold, strategy)
        returns.append(result.returns)
        trades.append(result.trades)
        trade_returns.append(result.trade_returns)

    combined_returns = pd.concat(returns).sort_index()
    equity_curve = config.initial_capital * (1.0 + combined_returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    combined_trades = pd.concat(trades).sort_index()
    combined_trade_returns = pd.concat(trade_returns, ignore_index=True)
    return backtester._calculate_stats(combined_returns, equity_curve, drawdown, combined_trades, combined_trade_returns)


def objective_factory(raw_data: pd.DataFrame, config: OptimizationConfig) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        macd_fast = trial.suggest_int("macd_fast", 6, 18)
        macd_slow = trial.suggest_int("macd_slow", macd_fast + 6, 50)
        feature_generator = FeatureGenerator(
            rsi_period=trial.suggest_int("rsi_period", 7, 30),
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=trial.suggest_int("macd_signal", 5, 18),
            bollinger_window=trial.suggest_int("bollinger_window", 10, 50),
            bollinger_std=trial.suggest_float("bollinger_std", 1.5, 2.8),
            volatility_window=trial.suggest_int("volatility_window", 10, 40),
            atr_period=trial.suggest_int("atr_period", 10, 30),
        )
        features = feature_generator.add_all_features(raw_data).dropna()
        folds = add_walk_forward_regimes(
            features,
            train_size=config.train_size,
            test_size=config.test_size,
            regime_lookback=trial.suggest_int("regime_lookback", 42, 126),
            random_state=config.random_state,
        )
        backtest_config = BacktestConfig(
            allow_short=False,
            atr_stop_multiple=trial.suggest_float("atr_stop_multiple", 1.5, 5.0),
            take_profit=None,
            slippage_model=SlippageModel(
                base_commission_bps=trial.suggest_float("base_commission_bps", 0.2, 1.5),
                variable_commission_bps=trial.suggest_float("variable_commission_bps", 0.0, 1.0),
            ),
        )
        strategy = RegimeAwareMomentumStrategy(
            lookback=trial.suggest_int("momentum_lookback", 21, 126),
            volatility_target=trial.suggest_float("volatility_target", 0.10, 0.30),
        )
        stats = run_fold_backtests(folds, strategy, backtest_config)
        calmar = stats["calmar"]
        return -10.0 if np.isnan(calmar) else calmar

    return objective


def fetch_raw_data(symbol: str, years: int) -> pd.DataFrame:
    end = pd.Timestamp(datetime.now(tz=UTC).date())
    start = end - pd.DateOffset(years=years)
    return DataHandler(provider="yfinance").fetch_historical(symbol, start=start, end=end, interval="1d")


def optimize_parameters(raw_data: pd.DataFrame, config: OptimizationConfig) -> optuna.study.Study:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
        study_name=f"{config.symbol}_calmar_walk_forward",
    )
    study.optimize(objective_factory(raw_data, config), n_trials=config.n_trials)
    return study


def backtest_before_after(raw_data: pd.DataFrame, best_params: dict[str, float | int], config: OptimizationConfig) -> pd.DataFrame:
    baseline_features = FeatureGenerator().add_all_features(raw_data).dropna()
    baseline_folds = add_walk_forward_regimes(
        baseline_features,
        train_size=config.train_size,
        test_size=config.test_size,
        regime_lookback=63,
        random_state=config.random_state,
    )
    baseline_stats = run_fold_backtests(
        baseline_folds,
        RegimeAwareMomentumStrategy(lookback=63, volatility_target=0.20),
        BacktestConfig(allow_short=False, atr_stop_multiple=3.0, take_profit=None, slippage_model=SlippageModel()),
    )

    macd_fast = int(best_params["macd_fast"])
    optimized_features = FeatureGenerator(
        rsi_period=int(best_params["rsi_period"]),
        macd_fast=macd_fast,
        macd_slow=int(best_params["macd_slow"]),
        macd_signal=int(best_params["macd_signal"]),
        bollinger_window=int(best_params["bollinger_window"]),
        bollinger_std=float(best_params["bollinger_std"]),
        volatility_window=int(best_params["volatility_window"]),
        atr_period=int(best_params["atr_period"]),
    ).add_all_features(raw_data).dropna()
    optimized_folds = add_walk_forward_regimes(
        optimized_features,
        train_size=config.train_size,
        test_size=config.test_size,
        regime_lookback=int(best_params["regime_lookback"]),
        random_state=config.random_state,
    )
    optimized_stats = run_fold_backtests(
        optimized_folds,
        RegimeAwareMomentumStrategy(
            lookback=int(best_params["momentum_lookback"]),
            volatility_target=float(best_params["volatility_target"]),
        ),
        BacktestConfig(
            allow_short=False,
            atr_stop_multiple=float(best_params["atr_stop_multiple"]),
            take_profit=None,
            slippage_model=SlippageModel(
                base_commission_bps=float(best_params["base_commission_bps"]),
                variable_commission_bps=float(best_params["variable_commission_bps"]),
            ),
        ),
    )

    table = pd.DataFrame(
        {
            "Before": {
                "Sharpe Ratio": baseline_stats["sharpe"],
                "Max Drawdown": baseline_stats["max_drawdown"],
                "Win Rate": baseline_stats["win_rate"],
                "Calmar Ratio": baseline_stats["calmar"],
            },
            "After": {
                "Sharpe Ratio": optimized_stats["sharpe"],
                "Max Drawdown": optimized_stats["max_drawdown"],
                "Win Rate": optimized_stats["win_rate"],
                "Calmar Ratio": optimized_stats["calmar"],
            },
        }
    )
    return table


def to_markdown_table(table: pd.DataFrame) -> str:
    rounded = table.round(4)
    lines = ["| Metric | Before | After |", "|---|---:|---:|"]
    for metric, row in rounded.iterrows():
        lines.append(f"| {metric} | {row['Before']} | {row['After']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward Optuna optimization for the trading framework.")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--train-size", type=int, default=504)
    parser.add_argument("--test-size", type=int, default=126)
    args = parser.parse_args()

    configure_logging()
    config = OptimizationConfig(
        symbol=args.symbol,
        years=args.years,
        train_size=args.train_size,
        test_size=args.test_size,
        n_trials=args.trials,
    )
    raw_data = fetch_raw_data(config.symbol, config.years)
    study = optimize_parameters(raw_data, config)
    comparison = backtest_before_after(raw_data, study.best_params, config)

    LOGGER.info("Best Calmar: %.4f", study.best_value)
    LOGGER.info("Best params: %s", study.best_params)
    print("\nBefore vs. After Walk-Forward Comparison")
    print(to_markdown_table(comparison))


if __name__ == "__main__":
    main()
