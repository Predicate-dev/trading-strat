"""Run a sample AAPL backtest for the last two years."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trading_strat import (
    BacktestConfig,
    DataHandler,
    FeatureGenerator,
    Plotter,
    RegimeAwareMomentumStrategy,
    RegimeFilter,
    RiskManager,
    SlippageModel,
    VectorizedBacktester,
)
from trading_strat.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


def run_sample_backtest() -> None:
    end = pd.Timestamp(datetime.now(tz=UTC).date())
    start = end - pd.DateOffset(years=2)

    data_handler = DataHandler(provider="yfinance")
    features = FeatureGenerator().add_all_features(
        data_handler.fetch_historical("AAPL", start=start, end=end, interval="1d")
    )
    features = RegimeFilter().add_regimes(features)

    strategy = RegimeAwareMomentumStrategy(lookback=63, volatility_target=0.20)
    config = BacktestConfig(
        initial_capital=100_000.0,
        slippage_model=SlippageModel(base_commission_bps=0.5, variable_commission_bps=0.5),
        sizing_mode="fractional",
        fractional_exposure=1.0,
        atr_stop_multiple=3.0,
        take_profit=None,
        allow_short=False,
    )
    result = VectorizedBacktester(config).run(features, strategy)

    benchmark = data_handler.fetch_historical("SPY", start=start, end=end, interval="1d")
    benchmark_returns = benchmark["close"].pct_change().fillna(0.0)

    risk = RiskManager().summary(result.returns, result.equity_curve)
    LOGGER.info("Backtest stats: %s", {key: round(value, 4) for key, value in result.stats.items()})
    LOGGER.info("Risk summary: %s", {key: round(value, 4) for key, value in risk.items()})

    figure = Plotter(title="AAPL Momentum Strategy vs SPY").cumulative_returns_figure(
        result.returns,
        benchmark_returns,
        strategy_name="AAPL Momentum",
        benchmark_name="SPY Buy & Hold",
    )
    output_path = Plotter().save_html(figure, Path("reports") / "aapl_backtest.html")
    LOGGER.info("Saved report to %s", output_path.resolve())


if __name__ == "__main__":
    configure_logging()
    run_sample_backtest()
