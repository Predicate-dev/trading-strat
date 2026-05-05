"""Market regime detection using Gaussian Mixture clustering."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

RegimeLabel = Literal["Bull Trend", "High Volatility", "Sideways"]

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RegimeFilter:
    """Classify market states into bull, high-volatility, and sideways regimes."""

    n_components: int = 3
    lookback: int = 63
    random_state: int = 42
    covariance_type: str = "full"
    label_column: str = "regime"
    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _model: GaussianMixture | None = field(default=None, init=False, repr=False)
    _cluster_labels: dict[int, RegimeLabel] = field(default_factory=dict, init=False, repr=False)

    def fit(self, data: pd.DataFrame) -> RegimeFilter:
        """Fit the scaler, GMM, and semantic cluster mapping on training data."""
        features = self._build_feature_matrix(data).dropna()
        if len(features) < max(self.lookback, self.n_components * 20):
            LOGGER.warning("Not enough observations for robust regime detection; using Sideways fallback.")
            self._scaler = None
            self._model = None
            self._cluster_labels = {}
            return self

        self._scaler = StandardScaler()
        matrix = self._scaler.fit_transform(features)
        self._model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=10,
        )
        clusters = pd.Series(self._model.fit_predict(matrix), index=features.index, name="cluster")
        self._cluster_labels = self._map_clusters_to_regimes(features, clusters)
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict market regimes with the previously fitted model."""
        if self._scaler is None or self._model is None or not self._cluster_labels:
            return pd.Series("Sideways", index=data.index, name=self.label_column)

        features = self._build_feature_matrix(data)
        valid_features = features.dropna()
        regimes = pd.Series("Sideways", index=data.index, name=self.label_column)
        if valid_features.empty:
            return regimes

        clusters = pd.Series(
            self._model.predict(self._scaler.transform(valid_features)),
            index=valid_features.index,
            name="cluster",
        )
        regimes.loc[valid_features.index] = clusters.map(self._cluster_labels)
        return regimes.ffill().fillna("Sideways").rename(self.label_column)

    def fit_predict(self, data: pd.DataFrame) -> pd.Series:
        """Fit a GMM and return regime labels aligned to the input index."""
        return self.fit(data).predict(data)

    def add_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result[self.label_column] = self.fit_predict(result)
        return result

    def _build_feature_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        returns = data["close"].pct_change()
        trend = data["close"].pct_change(self.lookback)
        volatility = returns.rolling(self.lookback).std(ddof=0) * np.sqrt(252)
        moving_average_slope = data["close"].rolling(self.lookback).mean().pct_change(self.lookback)
        drawdown = data["close"] / data["close"].rolling(self.lookback).max() - 1.0
        return pd.DataFrame(
            {
                "trend": trend,
                "volatility": volatility,
                "moving_average_slope": moving_average_slope,
                "drawdown": drawdown,
            },
            index=data.index,
        )

    @staticmethod
    def _map_clusters_to_regimes(features: pd.DataFrame, clusters: pd.Series) -> dict[int, RegimeLabel]:
        summary = features.groupby(clusters).agg(
            trend=("trend", "mean"),
            volatility=("volatility", "mean"),
            slope=("moving_average_slope", "mean"),
            drawdown=("drawdown", "mean"),
        )
        high_vol_cluster = int(summary["volatility"].idxmax())
        remaining = summary.drop(index=high_vol_cluster)
        bull_score = remaining["trend"] + remaining["slope"] + remaining["drawdown"].clip(upper=0.0)
        bull_cluster = int(bull_score.idxmax()) if not remaining.empty else high_vol_cluster

        labels: dict[int, RegimeLabel] = {}
        for cluster in summary.index:
            cluster_id = int(cluster)
            if cluster_id == high_vol_cluster:
                labels[cluster_id] = "High Volatility"
            elif cluster_id == bull_cluster:
                labels[cluster_id] = "Bull Trend"
            else:
                labels[cluster_id] = "Sideways"
        return labels
