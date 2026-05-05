"""Dash research dashboard for cached/fetched bars and decision inspection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trading_strat.backtester import BacktestConfig, VectorizedBacktester
from trading_strat.data import DataHandler
from trading_strat.features import FeatureGenerator
from trading_strat.ml import TrainedModel
from trading_strat.plotter import Plotter
from trading_strat.regime import RegimeFilter
from trading_strat.slippage import SlippageModel
from trading_strat.strategies import AdaptiveTrendStrategy, MLSignalStrategy, RegimeAwareMomentumStrategy

VisualizationMode = str


def create_dashboard_app(
    default_symbol: str = "AAPL",
    cache_dir: Path = Path(".cache/data"),
    model_path: Path | None = None,
):
    """Create a Dash app without starting the server."""
    try:
        from dash import Dash, Input, Output, State, dcc, html
    except ImportError as exc:
        raise ImportError("Install dash to launch the research dashboard.") from exc

    app = Dash(__name__)
    app.title = "Trading Research Dashboard"
    app.layout = html.Div(
        [
            html.Div(
                [
                    dcc.Input(id="symbol", value=default_symbol, type="text", debounce=True),
                    dcc.Dropdown(
                        id="strategy",
                        value="adaptive_trend",
                        clearable=False,
                        options=[
                            {"label": "Adaptive Trend", "value": "adaptive_trend"},
                            {"label": "Regime Momentum", "value": "regime_momentum"},
                            {"label": "ML Signal", "value": "ml_signal"},
                        ],
                    ),
                    dcc.Dropdown(
                        id="visualization-mode",
                        value="decision",
                        clearable=False,
                        options=[
                            {"label": "Decision Candles", "value": "decision"},
                            {"label": "Price / Regimes", "value": "price"},
                            {"label": "Equity / Drawdown", "value": "equity"},
                            {"label": "Signals / Costs", "value": "signals"},
                            {"label": "Model Diagnostics", "value": "model"},
                        ],
                    ),
                    dcc.Input(id="years", value=2, type="number", min=1, max=20),
                    html.Button("Refresh", id="refresh", n_clicks=0),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 120px 120px", "gap": "12px"},
            ),
            dcc.Loading(dcc.Graph(id="decision-figure")),
            html.Pre(id="stats", style={"whiteSpace": "pre-wrap"}),
        ],
        style={"fontFamily": "system-ui, sans-serif", "padding": "16px"},
    )

    @app.callback(
        Output("decision-figure", "figure"),
        Output("stats", "children"),
        Input("refresh", "n_clicks"),
        State("symbol", "value"),
        State("strategy", "value"),
        State("visualization-mode", "value"),
        State("years", "value"),
    )
    def refresh(_: int, symbol: str, strategy_name: str, visualization_mode: VisualizationMode, years: int):
        end = pd.Timestamp(datetime.now(tz=UTC).date())
        start = end - pd.DateOffset(years=int(years or 2))
        data_handler = DataHandler(provider="yfinance", cache_dir=cache_dir)
        raw = data_handler.fetch_historical(symbol.upper(), start=start, end=end, interval="1d", use_cache=True, refresh=True)
        features = FeatureGenerator().add_all_features(raw)
        features = RegimeFilter().add_regimes(features)

        if strategy_name == "ml_signal":
            if model_path is None or not Path(model_path).exists():
                raise ValueError("ML Signal strategy requires an existing model path.")
            model = TrainedModel.load(model_path)
            predictions = model.predict_frame(features)
            features = features.join(predictions, how="left")
            strategy = MLSignalStrategy(allow_short=False)
        elif strategy_name == "regime_momentum":
            strategy = RegimeAwareMomentumStrategy(lookback=63, volatility_target=0.20)
        else:
            strategy = AdaptiveTrendStrategy()

        result = VectorizedBacktester(
            BacktestConfig(
                allow_short=False,
                take_profit=None,
                atr_stop_multiple=3.0,
                slippage_model=SlippageModel(),
            )
        ).run(features, strategy)
        figure = build_dashboard_figure(
            raw,
            result.decision_ledger,
            visualization_mode,
            title=f"{symbol.upper()} {strategy_name}",
        )
        stats = "\n".join(f"{key}: {value:.4f}" for key, value in result.stats.items())
        return figure, stats

    return app


def build_dashboard_figure(
    raw: pd.DataFrame,
    decision_ledger: pd.DataFrame,
    visualization_mode: VisualizationMode,
    title: str,
):
    """Route dashboard visualization modes to specific Plotly figures."""
    plotter = Plotter(title=title)
    if visualization_mode == "price":
        return plotter.price_regime_figure(raw, decision_ledger, title=title)
    if visualization_mode == "equity":
        return plotter.equity_drawdown_figure(decision_ledger, title=title)
    if visualization_mode == "signals":
        return plotter.signal_diagnostics_figure(decision_ledger, title=title)
    if visualization_mode == "model":
        return plotter.model_diagnostics_figure(decision_ledger, title=title)
    return plotter.candlestick_decision_figure(raw, decision_ledger, title=title)


def run_dashboard(
    symbol: str = "AAPL",
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    model_path: Path | None = None,
) -> None:
    app = create_dashboard_app(default_symbol=symbol, model_path=model_path)
    app.run(host=host, port=port, debug=debug)
