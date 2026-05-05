"""Visualization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(slots=True)
class Plotter:
    """Plot backtest results against a benchmark."""

    title: str = "Strategy vs Benchmark"

    def cumulative_returns_figure(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        strategy_name: str = "Strategy",
        benchmark_name: str = "SPY",
    ) -> go.Figure:
        aligned = pd.concat(
            [strategy_returns.rename(strategy_name), benchmark_returns.rename(benchmark_name)],
            axis=1,
            join="inner",
        ).fillna(0.0)
        cumulative = (1.0 + aligned).cumprod() - 1.0

        figure = go.Figure()
        for column in cumulative.columns:
            figure.add_trace(
                go.Scatter(
                    x=cumulative.index,
                    y=cumulative[column],
                    mode="lines",
                    name=column,
                )
            )

        figure.update_layout(
            title=self.title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            yaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        return figure

    def save_html(self, figure: go.Figure, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(path)
        return path

    def candlestick_decision_figure(
        self,
        ohlcv: pd.DataFrame,
        decision_ledger: pd.DataFrame,
        title: str | None = None,
    ) -> go.Figure:
        """Build a multi-panel research figure for candles, decisions, model fields, and equity."""
        ledger = decision_ledger.copy()
        dates = pd.to_datetime(ledger["date"]) if "date" in ledger.columns else pd.to_datetime(ledger.index)
        figure = make_subplots(
            rows=6,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.34, 0.12, 0.12, 0.16, 0.10, 0.16],
            specs=[
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "table"}],
            ],
            subplot_titles=("OHLC Decisions", "Position", "Model Confidence / Prediction", "Equity", "Drawdown", "Ledger"),
        )
        figure.add_trace(
            go.Candlestick(
                x=ohlcv.index,
                open=ohlcv["open"],
                high=ohlcv["high"],
                low=ohlcv["low"],
                close=ohlcv["close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        self._add_decision_markers(figure, ledger, dates)
        if "regime" in ledger.columns:
            self._add_regime_shading(figure, ledger, dates)

        figure.add_trace(go.Scatter(x=dates, y=ledger["position"], name="Position", mode="lines"), row=2, col=1)
        if "confidence" in ledger.columns:
            figure.add_trace(go.Scatter(x=dates, y=ledger["confidence"], name="Confidence", mode="lines"), row=3, col=1)
        if "predicted_return" in ledger.columns:
            figure.add_trace(
                go.Scatter(x=dates, y=ledger["predicted_return"], name="Predicted Return", mode="lines"),
                row=3,
                col=1,
            )
        figure.add_trace(go.Scatter(x=dates, y=ledger["equity"], name="Equity", mode="lines"), row=4, col=1)
        figure.add_trace(
            go.Scatter(x=dates, y=ledger["drawdown"], name="Drawdown", mode="lines", fill="tozeroy"),
            row=5,
            col=1,
        )

        table_columns = [
            column
            for column in ["date", "close", "trade_action", "reason", "position", "confidence", "predicted_return", "equity"]
            if column in ledger.columns
        ]
        table = ledger[table_columns].tail(20).copy()
        if "date" in table.columns:
            table["date"] = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")
        figure.add_trace(
            go.Table(
                header=dict(values=table_columns, fill_color="#f2f2f2", align="left"),
                cells=dict(values=[table[column] for column in table_columns], align="left"),
                name="Decision Table",
            ),
            row=6,
            col=1,
        )
        figure.update_layout(
            title=title or self.title,
            template="plotly_white",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            height=1_200,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        figure.update_yaxes(tickformat=".0%", row=5, col=1)
        return figure

    def price_regime_figure(
        self,
        ohlcv: pd.DataFrame,
        decision_ledger: pd.DataFrame | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Show OHLC candles, volume, and optional regime shading."""
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25],
            subplot_titles=("Price", "Volume"),
        )
        figure.add_trace(
            go.Candlestick(
                x=ohlcv.index,
                open=ohlcv["open"],
                high=ohlcv["high"],
                low=ohlcv["low"],
                close=ohlcv["close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["volume"], name="Volume", marker_color="#98a2b3"), row=2, col=1)
        if decision_ledger is not None and "regime" in decision_ledger.columns:
            dates = pd.to_datetime(decision_ledger["date"]) if "date" in decision_ledger.columns else pd.to_datetime(decision_ledger.index)
            self._add_regime_shading(figure, decision_ledger, dates)
        figure.update_layout(
            title=title or self.title,
            template="plotly_white",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            height=760,
        )
        return figure

    def equity_drawdown_figure(self, decision_ledger: pd.DataFrame, title: str | None = None) -> go.Figure:
        """Show strategy equity, drawdown, and per-bar returns."""
        ledger = decision_ledger.copy()
        dates = pd.to_datetime(ledger["date"]) if "date" in ledger.columns else pd.to_datetime(ledger.index)
        figure = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.45, 0.30, 0.25],
            subplot_titles=("Equity", "Drawdown", "Strategy Return"),
        )
        figure.add_trace(go.Scatter(x=dates, y=ledger["equity"], name="Equity", mode="lines"), row=1, col=1)
        figure.add_trace(
            go.Scatter(x=dates, y=ledger["drawdown"], name="Drawdown", mode="lines", fill="tozeroy"),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Bar(x=dates, y=ledger["strategy_return"], name="Strategy Return", marker_color="#175cd3"),
            row=3,
            col=1,
        )
        figure.update_layout(title=title or self.title, template="plotly_white", hovermode="x unified", height=780)
        figure.update_yaxes(tickformat=".0%", row=2, col=1)
        figure.update_yaxes(tickformat=".1%", row=3, col=1)
        return figure

    def signal_diagnostics_figure(self, decision_ledger: pd.DataFrame, title: str | None = None) -> go.Figure:
        """Show raw/executable signals, position, turnover, and costs."""
        ledger = decision_ledger.copy()
        dates = pd.to_datetime(ledger["date"]) if "date" in ledger.columns else pd.to_datetime(ledger.index)
        figure = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=("Signals", "Position", "Turnover", "Cost"),
        )
        figure.add_trace(go.Scatter(x=dates, y=ledger["raw_signal"], name="Raw Signal", mode="lines"), row=1, col=1)
        figure.add_trace(
            go.Scatter(x=dates, y=ledger["executable_signal"], name="Executable Signal", mode="lines"),
            row=1,
            col=1,
        )
        figure.add_trace(go.Scatter(x=dates, y=ledger["position"], name="Position", mode="lines"), row=2, col=1)
        figure.add_trace(go.Bar(x=dates, y=ledger["turnover"], name="Turnover", marker_color="#667085"), row=3, col=1)
        figure.add_trace(go.Bar(x=dates, y=ledger["cost"], name="Cost", marker_color="#b42318"), row=4, col=1)
        figure.update_layout(title=title or self.title, template="plotly_white", hovermode="x unified", height=860)
        return figure

    def model_diagnostics_figure(self, decision_ledger: pd.DataFrame, title: str | None = None) -> go.Figure:
        """Show model probability, confidence, prediction, and decision table."""
        ledger = decision_ledger.copy()
        dates = pd.to_datetime(ledger["date"]) if "date" in ledger.columns else pd.to_datetime(ledger.index)
        figure = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.28, 0.28, 0.24, 0.20],
            specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
            subplot_titles=("Probability Up", "Confidence", "Predicted Return", "Recent Decisions"),
        )
        if "probability_up" in ledger.columns:
            figure.add_trace(go.Scatter(x=dates, y=ledger["probability_up"], name="Probability Up", mode="lines"), row=1, col=1)
            figure.add_hline(y=0.5, line_dash="dot", line_color="#667085", row=1, col=1)
        if "confidence" in ledger.columns:
            figure.add_trace(go.Scatter(x=dates, y=ledger["confidence"], name="Confidence", mode="lines"), row=2, col=1)
        if "predicted_return" in ledger.columns:
            figure.add_trace(
                go.Scatter(x=dates, y=ledger["predicted_return"], name="Predicted Return", mode="lines"),
                row=3,
                col=1,
            )
            figure.add_hline(y=0.0, line_dash="dot", line_color="#667085", row=3, col=1)

        table_columns = [
            column
            for column in ["date", "trade_action", "reason", "probability_up", "confidence", "predicted_return", "position"]
            if column in ledger.columns
        ]
        table = ledger[table_columns].tail(25).copy()
        if "date" in table.columns:
            table["date"] = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")
        figure.add_trace(
            go.Table(
                header=dict(values=table_columns, fill_color="#f2f2f2", align="left"),
                cells=dict(values=[table[column] for column in table_columns], align="left"),
            ),
            row=4,
            col=1,
        )
        figure.update_layout(title=title or self.title, template="plotly_white", hovermode="x unified", height=900)
        figure.update_yaxes(tickformat=".0%", row=1, col=1)
        figure.update_yaxes(tickformat=".0%", row=2, col=1)
        figure.update_yaxes(tickformat=".1%", row=3, col=1)
        return figure

    @staticmethod
    def _add_decision_markers(figure: go.Figure, ledger: pd.DataFrame, dates: pd.Series) -> None:
        marker_specs = {
            "buy": ("triangle-up", "#12805c", "Buy"),
            "sell_short": ("triangle-down", "#b42318", "Short"),
            "exit": ("x", "#344054", "Exit"),
            "cover_and_buy": ("triangle-up", "#175cd3", "Cover/Buy"),
            "sell_and_short": ("triangle-down", "#93370d", "Sell/Short"),
        }
        for action, (symbol, color, name) in marker_specs.items():
            mask = ledger["trade_action"].eq(action)
            if mask.any():
                figure.add_trace(
                    go.Scatter(
                        x=dates[mask],
                        y=ledger.loc[mask, "close"],
                        mode="markers",
                        marker=dict(symbol=symbol, size=11, color=color),
                        name=name,
                    ),
                    row=1,
                    col=1,
                )

    @staticmethod
    def _add_regime_shading(figure: go.Figure, ledger: pd.DataFrame, dates: pd.Series) -> None:
        colors = {"Bull Trend": "rgba(18,128,92,0.08)", "High Volatility": "rgba(180,35,24,0.08)", "Sideways": "rgba(102,112,133,0.06)"}
        regimes = ledger["regime"].fillna("Unknown").to_numpy()
        if len(regimes) == 0:
            return
        start = 0
        for idx in range(1, len(regimes) + 1):
            if idx == len(regimes) or regimes[idx] != regimes[start]:
                regime = regimes[start]
                figure.add_vrect(
                    x0=dates.iloc[start],
                    x1=dates.iloc[idx - 1],
                    fillcolor=colors.get(regime, "rgba(102,112,133,0.04)"),
                    line_width=0,
                    layer="below",
                    row=1,
                    col=1,
                )
                start = idx
