"""Command-line entry point for trading-strat research workflows."""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trading_strat import (
    AdaptiveTrendStrategy,
    BacktestConfig,
    DataHandler,
    DataRequest,
    FeatureGenerator,
    MLSignalStrategy,
    ModelConfig,
    Plotter,
    RegimeAwareMomentumStrategy,
    RegimeFilter,
    RiskManager,
    SlippageModel,
    TargetBuilder,
    TimeSeriesModelTrainer,
    TrainedModel,
    VectorizedBacktester,
)
from trading_strat.dashboard import run_dashboard
from trading_strat.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)
DEFAULT_UNIVERSE = "AAPL,MSFT,NVDA,AMZN,META,GOOGL,SPY,QQQ"


def date_range(years: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(datetime.now(tz=UTC).date())
    return end - pd.DateOffset(years=years), end


def parse_symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(symbol.strip().upper() for symbol in value.split(",") if symbol.strip())
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def universe_label(symbols: tuple[str, ...], model_type: str = "random_forest_classifier", horizon: int = 5) -> str:
    if len(symbols) == 1:
        prefix = symbols[0]
    else:
        prefix = f"universe{len(symbols)}"
    return f"{prefix}_{model_type}_{horizon}"


def load_features(symbol: str, years: int, refresh: bool = False) -> pd.DataFrame:
    start, end = date_range(years)
    data_handler = DataHandler(provider="yfinance", cache_dir=Path(".cache/data"))
    raw = data_handler.fetch_historical(symbol, start=start, end=end, interval="1d", use_cache=True, refresh=refresh)
    features = FeatureGenerator().add_all_features(raw)
    return RegimeFilter().add_regimes(features)


def load_universe_features(symbols: tuple[str, ...], years: int, refresh: bool = False) -> pd.DataFrame:
    start, end = date_range(years)
    request = DataRequest(
        symbols=symbols,
        start=start,
        end=end,
        interval="1d",
        cache_dir=Path(".cache/data"),
        use_cache=True,
        refresh=refresh,
    )
    raw = DataHandler(provider="yfinance", cache_dir=Path(".cache/data")).fetch_many(request)
    features = FeatureGenerator().add_all_features(raw)
    return RegimeFilter().add_regimes(features)


def select_symbol_frame(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(data.index, pd.MultiIndex):
        return data.xs(symbol, level="symbol").copy()
    return data.copy()


def run_demo(args: argparse.Namespace) -> None:
    features = load_features(args.symbol, args.years, refresh=args.refresh)
    strategy = AdaptiveTrendStrategy()
    result = run_backtest_frame(features, strategy)

    start, end = date_range(args.years)
    benchmark = DataHandler(provider="yfinance", cache_dir=Path(".cache/data")).fetch_historical(
        "SPY",
        start=start,
        end=end,
        interval="1d",
        use_cache=True,
        refresh=args.refresh,
    )
    benchmark_returns = benchmark["close"].pct_change().fillna(0.0)
    risk = RiskManager().summary(result.returns, result.equity_curve)
    LOGGER.info("Backtest stats: %s", {key: round(value, 4) for key, value in result.stats.items()})
    LOGGER.info("Risk summary: %s", {key: round(value, 4) for key, value in risk.items()})

    reports = Path("reports")
    Plotter(title=f"{args.symbol} Strategy vs SPY").save_html(
        Plotter(title=f"{args.symbol} Strategy vs SPY").cumulative_returns_figure(
            result.returns,
            benchmark_returns,
            strategy_name=f"{args.symbol} Strategy",
            benchmark_name="SPY Buy & Hold",
        ),
        reports / f"{args.symbol.lower()}_backtest.html",
    )
    Plotter(title=f"{args.symbol} Decisions").save_html(
        Plotter(title=f"{args.symbol} Decisions").candlestick_decision_figure(features, result.decision_ledger),
        reports / f"{args.symbol.lower()}_decisions.html",
    )


def ingest(args: argparse.Namespace) -> None:
    start = pd.Timestamp(args.start)
    end = None if args.end is None else pd.Timestamp(args.end)
    symbols = tuple(symbol.strip().upper() for symbol in args.symbols.split(","))
    request = DataRequest(
        symbols=symbols,
        start=start,
        end=end,
        interval=args.interval,
        cache_dir=Path(".cache/data"),
        refresh=args.refresh,
    )
    data = DataHandler(provider=args.provider, cache_dir=Path(".cache/data")).fetch_many(request)
    LOGGER.info("Ingested %s rows for %s", len(data), ", ".join(symbols))


def train(args: argparse.Namespace) -> None:
    symbols = parse_symbols(args.symbols)
    base_features = (
        load_universe_features(symbols, args.years, refresh=args.refresh)
        if len(symbols) > 1
        else load_features(symbols[0], args.years, refresh=args.refresh)
    )
    features = TargetBuilder(horizons=(args.horizon,)).add_targets(base_features)
    model_name = args.model_name or universe_label(symbols, args.model_type, args.horizon)
    config = ModelConfig(
        model_type=args.model_type,
        target_column=f"target_up_{args.horizon}" if "classifier" in args.model_type or args.model_type == "logistic_regression" else f"forward_return_{args.horizon}",
        prediction_horizon=args.horizon,
        train_size=args.train_size,
        test_size=args.test_size,
        embargo=args.embargo,
        model_name=model_name,
    )
    trainer = TimeSeriesModelTrainer(config)
    model = trainer.fit(features)
    predictions = trainer.walk_forward_predict(features)
    output = model.save(Path(".cache/models") / f"{config.model_name}.joblib")
    LOGGER.info("Saved model to %s", output.resolve())
    LOGGER.info("Trained on symbols: %s", ", ".join(symbols))
    LOGGER.info("Walk-forward predictions: %s rows", len(predictions))
    if model.feature_importance_ is not None:
        LOGGER.info("Top features: %s", model.feature_importance_.head(10).round(4).to_dict())


def backtest(args: argparse.Namespace) -> None:
    symbols = parse_symbols(args.symbols)
    model = None
    scored_universe: pd.DataFrame | None = None
    if args.strategy == "ml" and args.prediction_mode == "model":
        model_path = args.model_path or Path(".cache/models") / f"{args.model_name}.joblib"
        model = TrainedModel.load(model_path)
    elif args.strategy == "ml":
        universe_features = load_universe_features(symbols, args.years, refresh=args.refresh)
        labeled = TargetBuilder(horizons=(args.horizon,)).add_targets(universe_features)
        config = ModelConfig(
            model_type=args.model_type,
            target_column=f"target_up_{args.horizon}",
            prediction_horizon=args.horizon,
            train_size=args.train_size,
            test_size=args.test_size,
            embargo=args.embargo,
            model_name=f"oos_{universe_label(symbols, args.model_type, args.horizon)}",
            model_params={"n_estimators": args.n_estimators, "min_samples_leaf": args.min_samples_leaf}
            if args.model_type == "random_forest_classifier"
            else {},
        )
        predictions = TimeSeriesModelTrainer(config).walk_forward_predict(labeled)
        scored_universe = universe_features.join(predictions, how="left")

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    summary_rows: list[dict[str, float | str]] = []

    for symbol in symbols:
        features = (
            select_symbol_frame(scored_universe, symbol)
            if scored_universe is not None
            else load_features(symbol, args.years, refresh=args.refresh)
        )
        if args.strategy == "ml":
            if args.prediction_mode == "model":
                assert model is not None
                features = features.join(model.predict_frame(features), how="left")
            features = features.loc[features["predicted_return"].notna()].copy()
            strategy = MLSignalStrategy(
                long_threshold=args.long_threshold,
                short_threshold=args.short_threshold,
                min_confidence=args.min_confidence,
                allow_short=args.allow_short,
                volatility_target=args.volatility_target,
                regime_column=None if args.disable_regime_filter else "regime",
            )
        elif args.strategy == "regime":
            strategy = RegimeAwareMomentumStrategy(lookback=args.lookback, volatility_target=args.volatility_target)
        else:
            strategy = AdaptiveTrendStrategy(
                breakout_window=args.breakout_window,
                exit_window=args.exit_window,
                fast_ema=args.fast_ema,
                slow_ema=args.slow_ema,
                long_ema=args.long_ema,
                volatility_target=args.volatility_target,
            )

        result = run_backtest_frame(features, strategy, allow_short=args.allow_short)
        ledger_path = reports / f"{symbol.lower()}_{args.strategy}_ledger.csv"
        result.decision_ledger.to_csv(ledger_path, index=False)
        html_path = Plotter(title=f"{symbol} {args.strategy} Decisions").save_html(
            Plotter(title=f"{symbol} {args.strategy} Decisions").candlestick_decision_figure(features, result.decision_ledger),
            reports / f"{symbol.lower()}_{args.strategy}_decisions.html",
        )
        trade_count = int(result.decision_ledger["trade_action"].ne("hold").sum())
        summary_rows.append(
            {
                "symbol": symbol,
                "sharpe": result.stats["sharpe"],
                "calmar": result.stats["calmar"],
                "max_drawdown": result.stats["max_drawdown"],
                "win_rate": result.stats["win_rate"],
                "exposure_time": result.stats["exposure_time"],
                "decision_count": trade_count,
            }
        )
        LOGGER.info("%s stats: %s", symbol, {key: round(value, 4) for key, value in result.stats.items()})
        LOGGER.info("Saved %s ledger to %s and report to %s", symbol, ledger_path.resolve(), html_path.resolve())

    summary = pd.DataFrame(summary_rows)
    summary_path = reports / f"{args.strategy}_universe_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Universe summary:\n%s", summary.round(4).to_string(index=False))
    LOGGER.info("Saved universe summary to %s", summary_path.resolve())


def optimize(args: argparse.Namespace) -> None:
    import optimize as optimizer

    config = optimizer.OptimizationConfig(symbol=args.symbol, years=args.years, n_trials=args.trials)
    raw_data = optimizer.fetch_raw_data(config.symbol, config.years)
    study = optimizer.optimize_parameters(raw_data, config)
    comparison = optimizer.backtest_before_after(raw_data, study.best_params, config)
    LOGGER.info("Best Calmar: %.4f", study.best_value)
    print(optimizer.to_markdown_table(comparison))


def optimize_ml(args: argparse.Namespace) -> None:
    import optuna

    symbols = parse_symbols(args.symbols)
    features = load_universe_features(symbols, args.years, refresh=args.refresh)
    labeled = TargetBuilder(horizons=(args.horizon,)).add_targets(features)
    config = ModelConfig(
        model_type=args.model_type,
        target_column=f"target_up_{args.horizon}",
        prediction_horizon=args.horizon,
        train_size=args.train_size,
        test_size=args.test_size,
        embargo=args.embargo,
        model_name=f"oos_{universe_label(symbols, args.model_type, args.horizon)}",
        model_params={"n_estimators": args.n_estimators, "min_samples_leaf": args.min_samples_leaf}
        if args.model_type == "random_forest_classifier"
        else {},
    )
    predictions = TimeSeriesModelTrainer(config).walk_forward_predict(labeled)
    scored = features.join(predictions, how="left")

    def objective(trial: optuna.Trial) -> float:
        long_threshold = trial.suggest_float("long_threshold", 0.0, 0.18)
        min_confidence = trial.suggest_float("min_confidence", 0.0, 0.45)
        volatility_target = trial.suggest_float("volatility_target", 0.10, 0.35)
        use_regime = trial.suggest_categorical("use_regime_filter", [False, True])
        atr_stop_multiple = trial.suggest_float("atr_stop_multiple", 1.5, 5.0)

        rows = []
        for symbol in symbols:
            symbol_frame = select_symbol_frame(scored, symbol)
            symbol_frame = symbol_frame.loc[symbol_frame["predicted_return"].notna()].copy()
            if len(symbol_frame) < 40:
                continue
            strategy = MLSignalStrategy(
                long_threshold=long_threshold,
                short_threshold=long_threshold,
                min_confidence=min_confidence,
                allow_short=args.allow_short,
                volatility_target=volatility_target,
                regime_column="regime" if use_regime else None,
            )
            result = VectorizedBacktester(
                BacktestConfig(
                    allow_short=args.allow_short,
                    take_profit=None,
                    atr_stop_multiple=atr_stop_multiple,
                    slippage_model=SlippageModel(),
                )
            ).run(symbol_frame, strategy)
            rows.append(
                {
                    "calmar": result.stats["calmar"],
                    "sharpe": result.stats["sharpe"],
                    "exposure_time": result.stats["exposure_time"],
                    "decisions": result.decision_ledger["trade_action"].ne("hold").sum(),
                }
            )
        if not rows:
            return -100.0
        summary = pd.DataFrame(rows).replace([float("inf"), float("-inf")], pd.NA).dropna()
        if summary.empty:
            return -100.0
        decision_penalty = max(0.0, args.min_decisions - float(summary["decisions"].mean())) * 0.05
        exposure_penalty = max(0.0, args.min_exposure - float(summary["exposure_time"].mean())) * 2.0
        return float(summary["calmar"].mean() + 0.25 * summary["sharpe"].mean() - decision_penalty - exposure_penalty)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials)
    LOGGER.info("Best active ML objective: %.4f", study.best_value)
    LOGGER.info("Best ML signal params: %s", study.best_params)
    Path("reports").mkdir(exist_ok=True)
    output_path = Path("reports") / "ml_signal_optimized_params.json"
    pd.Series(study.best_params).to_json(output_path, indent=2)
    LOGGER.info("Saved optimized ML signal params to %s", output_path.resolve())


def dashboard(args: argparse.Namespace) -> None:
    run_dashboard(symbol=args.symbol, host=args.host, port=args.port, debug=args.debug, model_path=args.model_path)


def run_backtest_frame(features: pd.DataFrame, strategy, allow_short: bool = False):
    return VectorizedBacktester(
        BacktestConfig(
            initial_capital=100_000.0,
            slippage_model=SlippageModel(base_commission_bps=0.5, variable_commission_bps=0.5),
            sizing_mode="fractional",
            fractional_exposure=1.0,
            atr_stop_multiple=3.0,
            take_profit=None,
            allow_short=allow_short,
        )
    ).run(features, strategy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trading-strat research CLI")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Run the default regime-aware backtest")
    demo_parser.add_argument("--symbol", default="AAPL")
    demo_parser.add_argument("--years", type=int, default=2)
    demo_parser.add_argument("--refresh", action="store_true")
    demo_parser.set_defaults(func=run_demo)

    ingest_parser = subparsers.add_parser("ingest", help="Fetch and cache OHLCV bars")
    ingest_parser.add_argument("--symbols", default="AAPL,MSFT,SPY")
    ingest_parser.add_argument("--provider", default="yfinance", choices=["yfinance", "ccxt"])
    ingest_parser.add_argument("--interval", default="1d")
    ingest_parser.add_argument("--start", default="2020-01-01")
    ingest_parser.add_argument("--end", default=None)
    ingest_parser.add_argument("--refresh", action="store_true")
    ingest_parser.set_defaults(func=ingest)

    train_parser = subparsers.add_parser("train", help="Train an sklearn model")
    train_parser.add_argument("--symbols", default=DEFAULT_UNIVERSE)
    train_parser.add_argument("--years", type=int, default=5)
    train_parser.add_argument("--horizon", type=int, default=5)
    train_parser.add_argument("--model-type", default="random_forest_classifier")
    train_parser.add_argument("--model-name", default=None)
    train_parser.add_argument("--train-size", type=int, default=504)
    train_parser.add_argument("--test-size", type=int, default=126)
    train_parser.add_argument("--embargo", type=int, default=5)
    train_parser.add_argument("--refresh", action="store_true")
    train_parser.set_defaults(func=train)

    backtest_parser = subparsers.add_parser("backtest", help="Run a rule-based or ML backtest")
    backtest_parser.add_argument("--symbols", default=DEFAULT_UNIVERSE)
    backtest_parser.add_argument("--years", type=int, default=5)
    backtest_parser.add_argument("--strategy", choices=["trend", "regime", "ml"], default="trend")
    backtest_parser.add_argument("--prediction-mode", choices=["walk-forward", "model"], default="walk-forward")
    backtest_parser.add_argument("--horizon", type=int, default=5)
    backtest_parser.add_argument("--model-type", default="random_forest_classifier")
    backtest_parser.add_argument("--model-name", default="universe8_random_forest_classifier_5")
    backtest_parser.add_argument("--model-path", type=Path, default=None)
    backtest_parser.add_argument("--train-size", type=int, default=504)
    backtest_parser.add_argument("--test-size", type=int, default=126)
    backtest_parser.add_argument("--embargo", type=int, default=5)
    backtest_parser.add_argument("--n-estimators", type=int, default=150)
    backtest_parser.add_argument("--min-samples-leaf", type=int, default=5)
    backtest_parser.add_argument("--lookback", type=int, default=63)
    backtest_parser.add_argument("--breakout-window", type=int, default=55)
    backtest_parser.add_argument("--exit-window", type=int, default=21)
    backtest_parser.add_argument("--fast-ema", type=int, default=20)
    backtest_parser.add_argument("--slow-ema", type=int, default=50)
    backtest_parser.add_argument("--long-ema", type=int, default=100)
    backtest_parser.add_argument("--volatility-target", type=float, default=0.20)
    backtest_parser.add_argument("--long-threshold", type=float, default=0.02)
    backtest_parser.add_argument("--short-threshold", type=float, default=0.02)
    backtest_parser.add_argument("--min-confidence", type=float, default=0.10)
    backtest_parser.add_argument("--disable-regime-filter", action="store_true")
    backtest_parser.add_argument("--allow-short", action="store_true")
    backtest_parser.add_argument("--refresh", action="store_true")
    backtest_parser.set_defaults(func=backtest)

    optimize_parser = subparsers.add_parser("optimize", help="Run Optuna walk-forward optimization")
    optimize_parser.add_argument("--symbol", default="AAPL")
    optimize_parser.add_argument("--years", type=int, default=5)
    optimize_parser.add_argument("--trials", type=int, default=30)
    optimize_parser.set_defaults(func=optimize)

    optimize_ml_parser = subparsers.add_parser("optimize-ml", help="Tune active ML signal thresholds on a universe")
    optimize_ml_parser.add_argument("--symbols", default=DEFAULT_UNIVERSE)
    optimize_ml_parser.add_argument("--years", type=int, default=5)
    optimize_ml_parser.add_argument("--horizon", type=int, default=5)
    optimize_ml_parser.add_argument("--model-type", default="random_forest_classifier")
    optimize_ml_parser.add_argument("--train-size", type=int, default=504)
    optimize_ml_parser.add_argument("--test-size", type=int, default=126)
    optimize_ml_parser.add_argument("--embargo", type=int, default=5)
    optimize_ml_parser.add_argument("--trials", type=int, default=50)
    optimize_ml_parser.add_argument("--n-estimators", type=int, default=150)
    optimize_ml_parser.add_argument("--min-samples-leaf", type=int, default=5)
    optimize_ml_parser.add_argument("--min-decisions", type=float, default=4.0)
    optimize_ml_parser.add_argument("--min-exposure", type=float, default=0.08)
    optimize_ml_parser.add_argument("--allow-short", action="store_true")
    optimize_ml_parser.add_argument("--refresh", action="store_true")
    optimize_ml_parser.set_defaults(func=optimize_ml)

    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the Dash research dashboard")
    dashboard_parser.add_argument("--symbol", default="AAPL")
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=8050)
    dashboard_parser.add_argument("--debug", action="store_true")
    dashboard_parser.add_argument("--model-path", type=Path, default=None)
    dashboard_parser.set_defaults(func=dashboard)
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["demo"])
    args.func(args)


if __name__ == "__main__":
    main()
