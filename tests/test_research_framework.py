from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_strat import (
    AdaptiveTrendStrategy,
    BacktestConfig,
    DataHandler,
    FeatureGenerator,
    MLSignalStrategy,
    ModelConfig,
    Plotter,
    RegimeAwareMomentumStrategy,
    RegimeFilter,
    TargetBuilder,
    TimeSeriesModelTrainer,
    VectorizedBacktester,
)
from trading_strat.dashboard import build_dashboard_figure


def synthetic_ohlcv(rows: int = 260, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=rows, freq="B", name="date")
    trend = np.linspace(0, 0.35, rows)
    cycle = np.sin(np.linspace(0, 16, rows)) * 0.04
    noise = rng.normal(0, 0.01, rows).cumsum()
    close = 100 * (1 + trend + cycle + noise).clip(0.2)
    open_ = close * (1 + rng.normal(0, 0.003, rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, rows))
    volume = rng.integers(500_000, 2_000_000, rows)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def test_data_validation_cleaning_and_parquet_cache(tmp_path: Path) -> None:
    data = synthetic_ohlcv(30)
    dirty = pd.concat([data, data.iloc[[5]]]).sort_index()
    dirty.iloc[3, dirty.columns.get_loc("close")] = np.nan
    dirty.iloc[4, dirty.columns.get_loc("volume")] = -1

    handler = DataHandler(cache_dir=tmp_path)
    report = handler.validate_ohlcv(dirty)
    assert report.duplicate_rows > 0
    assert report.missing_values > 0
    assert report.bad_volume_rows > 0

    cleaned = handler.handle_missing_values(dirty)
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.has_duplicates
    assert cleaned["volume"].ge(0).all()

    path = tmp_path / "sample.parquet"
    handler._write_cache(path, cleaned)
    cached = handler._read_cache(path)
    assert len(cached) == len(cleaned)


def test_features_targets_are_aligned_without_lookahead() -> None:
    data = synthetic_ohlcv(120)
    features = FeatureGenerator().add_all_features(data)
    labeled = TargetBuilder(horizons=(5,), drop_last_horizon_rows=False).add_targets(features)
    expected = data["close"].shift(-5) / data["close"] - 1.0
    pd.testing.assert_series_equal(labeled["forward_return_5"], expected, check_names=False)
    assert "momentum_63" in labeled.columns
    assert "volume_zscore" in labeled.columns
    assert "atr_percent" in labeled.columns


def test_multi_symbol_cross_sectional_ranks() -> None:
    frames = []
    for symbol, seed in [("AAA", 1), ("BBB", 2), ("CCC", 3)]:
        frame = synthetic_ohlcv(100, seed=seed)
        frame["symbol"] = symbol
        frames.append(frame)
    multi = pd.concat(frames).set_index("symbol", append=True).sort_index()
    multi.index = multi.index.set_names(["date", "symbol"])
    features = FeatureGenerator().add_all_features(multi)
    rank_columns = [column for column in features.columns if column.startswith("cs_rank_")]
    assert rank_columns
    assert features[rank_columns].max().max() <= 1.0
    with_regimes = RegimeFilter().add_regimes(features)
    assert "regime" in with_regimes.columns
    assert set(with_regimes.index.names) == {"date", "symbol"}


def test_ml_walk_forward_prediction_shape_and_bounds() -> None:
    data = RegimeFilter().add_regimes(FeatureGenerator().add_all_features(synthetic_ohlcv(240)))
    labeled = TargetBuilder(horizons=(5,)).add_targets(data)
    config = ModelConfig(
        model_type="random_forest_classifier",
        target_column="target_up_5",
        train_size=80,
        test_size=40,
        embargo=5,
        model_params={"n_estimators": 20, "min_samples_leaf": 3},
    )
    predictions = TimeSeriesModelTrainer(config).walk_forward_predict(labeled)
    assert {"probability_up", "confidence", "predicted_return", "model_name", "fold"}.issubset(predictions.columns)
    assert predictions["probability_up"].between(0.0, 1.0).all()
    assert predictions["confidence"].between(0.0, 1.0).all()


def test_ml_signal_bounds_and_backtest_ledger_columns() -> None:
    data = RegimeFilter().add_regimes(FeatureGenerator().add_all_features(synthetic_ohlcv(180))).dropna()
    data["predicted_return"] = np.linspace(-0.02, 0.03, len(data))
    data["confidence"] = 0.8
    strategy = MLSignalStrategy(long_threshold=0.005, short_threshold=0.005, min_confidence=0.5, allow_short=True)
    signals = strategy.generate_signals(data)
    assert signals.between(-1.0, 1.0).all()

    result = VectorizedBacktester(BacktestConfig(atr_stop_multiple=3.0, take_profit=None)).run(data, strategy)
    required = {
        "date",
        "close",
        "raw_signal",
        "executable_signal",
        "position",
        "turnover",
        "cost",
        "strategy_return",
        "equity",
        "drawdown",
        "trade_action",
        "reason",
        "confidence",
        "predicted_return",
        "stop_exit",
        "take_profit_exit",
    }
    assert required.issubset(result.decision_ledger.columns)
    assert "profit_factor" in result.stats
    assert "average_holding_period" in result.stats


def test_adaptive_trend_stays_in_uptrend() -> None:
    data = RegimeFilter().add_regimes(FeatureGenerator().add_all_features(synthetic_ohlcv(220)))
    data = data.dropna(subset=["rolling_volatility"])
    signals = AdaptiveTrendStrategy(
        breakout_window=30,
        exit_window=15,
        fast_ema=10,
        slow_ema=25,
        long_ema=50,
        volatility_target=None,
    ).generate_signals(data)
    assert signals.between(0.0, 1.0).all()
    assert signals.tail(80).mean() > 0.25


def test_plot_generation(tmp_path: Path) -> None:
    data = RegimeFilter().add_regimes(FeatureGenerator().add_all_features(synthetic_ohlcv(220)))
    data = data.dropna(subset=["atr", "rolling_volatility", "bb_lower", "bb_upper"])
    data["probability_up"] = np.linspace(0.35, 0.65, len(data))
    data["confidence"] = (data["probability_up"] - 0.5).abs() * 2.0
    data["predicted_return"] = data["probability_up"] - 0.5
    result = VectorizedBacktester(BacktestConfig(atr_stop_multiple=3.0, take_profit=None, allow_short=False)).run(
        data,
        RegimeAwareMomentumStrategy(lookback=20, volatility_target=None),
    )
    figure = Plotter(title="Synthetic Decisions").candlestick_decision_figure(data, result.decision_ledger)
    output = Plotter().save_html(figure, tmp_path / "plot.html")
    assert output.exists()
    assert output.read_text().startswith("<html>")

    for mode in ["decision", "price", "equity", "signals", "model"]:
        dashboard_figure = build_dashboard_figure(data, result.decision_ledger, mode, title=f"Mode {mode}")
        assert dashboard_figure.data
