"""Portfolio risk metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RiskManager:
    """Calculate market and portfolio risk diagnostics."""

    annualization_factor: int = 252
    risk_free_rate: float = 0.0

    def value_at_risk(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> float:
        """Calculate positive VaR as the loss threshold at a confidence level."""
        clean_returns = returns.dropna()
        if clean_returns.empty:
            raise ValueError("Returns are empty; cannot calculate VaR.")

        if method == "historical":
            return float(-np.percentile(clean_returns, (1.0 - confidence_level) * 100.0))
        if method == "parametric":
            z_score = self._normal_z_score(1.0 - confidence_level)
            return float(-(clean_returns.mean() + z_score * clean_returns.std(ddof=0)))
        raise ValueError(f"Unsupported VaR method: {method}")

    def maximum_drawdown(self, equity_curve: pd.Series) -> float:
        if equity_curve.empty:
            raise ValueError("Equity curve is empty; cannot calculate drawdown.")
        drawdown = equity_curve / equity_curve.cummax() - 1.0
        return float(drawdown.min())

    def sharpe_ratio(self, returns: pd.Series) -> float:
        excess_returns = returns.dropna() - self.risk_free_rate / self.annualization_factor
        std = excess_returns.std(ddof=0)
        if std == 0.0:
            return float("nan")
        return float(excess_returns.mean() / std * np.sqrt(self.annualization_factor))

    def sortino_ratio(self, returns: pd.Series) -> float:
        excess_returns = returns.dropna() - self.risk_free_rate / self.annualization_factor
        downside = excess_returns[excess_returns < 0.0].std(ddof=0)
        if downside == 0.0:
            return float("nan")
        return float(excess_returns.mean() / downside * np.sqrt(self.annualization_factor))

    def calmar_ratio(self, returns: pd.Series, equity_curve: pd.Series) -> float:
        if equity_curve.empty:
            raise ValueError("Equity curve is empty; cannot calculate Calmar ratio.")
        years = max(len(returns.dropna()) / self.annualization_factor, 1 / self.annualization_factor)
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / years) - 1.0
        max_drawdown = abs(self.maximum_drawdown(equity_curve))
        if max_drawdown == 0.0:
            return float("nan")
        return float(cagr / max_drawdown)

    def summary(self, returns: pd.Series, equity_curve: pd.Series) -> dict[str, float]:
        return {
            "var_95": self.value_at_risk(returns, confidence_level=0.95),
            "max_drawdown": self.maximum_drawdown(equity_curve),
            "sharpe": self.sharpe_ratio(returns),
            "sortino": self.sortino_ratio(returns),
            "calmar": self.calmar_ratio(returns, equity_curve),
        }

    @staticmethod
    def _normal_z_score(probability: float) -> float:
        statistics_normal = getattr(__import__("statistics"), "NormalDist")
        return float(statistics_normal().inv_cdf(probability))
