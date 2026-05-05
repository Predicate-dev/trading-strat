"""Visualization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


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

