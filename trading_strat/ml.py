"""Machine-learning datasets, training, walk-forward prediction, and persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelType = Literal[
    "logistic_regression",
    "random_forest_classifier",
    "random_forest_regressor",
    "hist_gradient_boosting_classifier",
    "hist_gradient_boosting_regressor",
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for sklearn model construction and walk-forward training."""

    model_type: ModelType = "random_forest_classifier"
    target_column: str = "target_up_5"
    prediction_horizon: int = 5
    train_size: int = 504
    test_size: int = 126
    embargo: int = 5
    random_state: int = 42
    model_name: str = "ml_model"
    scale_features: bool = True
    model_params: dict[str, object] = field(default_factory=dict)

    @property
    def is_classifier(self) -> bool:
        return "classifier" in self.model_type or self.model_type == "logistic_regression"


@dataclass(frozen=True, slots=True)
class Dataset:
    features: pd.DataFrame
    target: pd.Series
    feature_columns: list[str]


@dataclass(slots=True)
class TrainedModel:
    """A fitted sklearn pipeline plus metadata."""

    pipeline: Pipeline
    config: ModelConfig
    feature_columns: list[str]
    feature_importance_: pd.Series | None = None

    def predict_frame(self, data: pd.DataFrame, fold: int = -1) -> pd.DataFrame:
        features = data.reindex(columns=self.feature_columns)
        output = pd.DataFrame(index=data.index)
        if self.config.is_classifier:
            if hasattr(self.pipeline, "predict_proba"):
                probability_up = self.pipeline.predict_proba(features)[:, 1]
            else:
                probability_up = self.pipeline.predict(features)
            output["probability_up"] = probability_up
            output["confidence"] = np.abs(probability_up - 0.5) * 2.0
            output["predicted_return"] = probability_up - 0.5
        else:
            predicted_return = self.pipeline.predict(features)
            output["predicted_return"] = predicted_return
            scale = np.nanstd(predicted_return)
            output["confidence"] = 0.0 if scale == 0.0 else np.clip(np.abs(predicted_return) / (2.0 * scale), 0.0, 1.0)
            output["probability_up"] = np.nan
        output["model_name"] = self.config.model_name
        output["fold"] = fold
        return output

    def save(self, path: str | Path) -> Path:
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Install joblib to persist trained models.") from exc

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path

    @staticmethod
    def load(path: str | Path) -> TrainedModel:
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Install joblib to load persisted models.") from exc
        return joblib.load(path)


@dataclass(slots=True)
class DatasetBuilder:
    """Build ML matrices from engineered feature frames."""

    target_column: str = "target_up_5"
    feature_columns: list[str] | None = None
    exclude_columns: tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "regime",
    )

    def build(self, data: pd.DataFrame) -> Dataset:
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' is missing.")
        feature_columns = self.feature_columns or self.infer_feature_columns(data)
        matrix = data[feature_columns].replace([np.inf, -np.inf], np.nan)
        target = data[self.target_column]
        valid = target.notna()
        return Dataset(features=matrix.loc[valid], target=target.loc[valid], feature_columns=feature_columns)

    def infer_feature_columns(self, data: pd.DataFrame) -> list[str]:
        excluded_prefixes = ("target_", "forward_return_")
        columns: list[str] = []
        for column in data.columns:
            if column in self.exclude_columns or str(column).startswith(excluded_prefixes):
                continue
            if pd.api.types.is_numeric_dtype(data[column]):
                columns.append(str(column))
        if not columns:
            raise ValueError("No numeric feature columns inferred.")
        return columns


class TimeSeriesModelTrainer:
    """Train sklearn models with embargoed walk-forward validation."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def fit(self, data: pd.DataFrame, feature_columns: list[str] | None = None) -> TrainedModel:
        dataset = DatasetBuilder(self.config.target_column, feature_columns).build(data)
        pipeline = self._build_pipeline()
        pipeline.fit(dataset.features, dataset.target)
        return TrainedModel(
            pipeline=pipeline,
            config=self.config,
            feature_columns=dataset.feature_columns,
            feature_importance_=self._feature_importance(pipeline, dataset.feature_columns),
        )

    def walk_forward_predict(self, data: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
        dataset_builder = DatasetBuilder(self.config.target_column, feature_columns)
        dataset = dataset_builder.build(data)
        date_index = self._date_index(dataset.features.index)
        unique_dates = pd.Index(pd.unique(date_index)).sort_values()
        predictions: list[pd.DataFrame] = []
        fold = 0
        start = 0

        while start + self.config.train_size + self.config.embargo + self.config.test_size <= len(unique_dates):
            train_dates = unique_dates[start : start + self.config.train_size]
            test_start = start + self.config.train_size + self.config.embargo
            test_dates = unique_dates[test_start : test_start + self.config.test_size]
            train_mask = date_index.isin(train_dates)
            test_mask = date_index.isin(test_dates)

            pipeline = self._build_pipeline()
            pipeline.fit(dataset.features.loc[train_mask], dataset.target.loc[train_mask])
            trained = TrainedModel(
                pipeline=pipeline,
                config=self.config,
                feature_columns=dataset.feature_columns,
                feature_importance_=self._feature_importance(pipeline, dataset.feature_columns),
            )
            fold_predictions = trained.predict_frame(dataset.features.loc[test_mask], fold=fold)
            predictions.append(fold_predictions)
            fold += 1
            start += self.config.test_size

        if not predictions:
            raise ValueError("No walk-forward folds produced. Increase data length or reduce train/test/embargo sizes.")
        return pd.concat(predictions).sort_index()

    def _build_pipeline(self) -> Pipeline:
        estimator = self._build_estimator()
        steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if self.config.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        return Pipeline(steps)

    def _build_estimator(self) -> object:
        params = dict(self.config.model_params)
        params.setdefault("random_state", self.config.random_state)
        if self.config.model_type == "logistic_regression":
            params.setdefault("max_iter", 1_000)
            return LogisticRegression(**params)
        if self.config.model_type == "random_forest_classifier":
            params.setdefault("n_estimators", 200)
            params.setdefault("min_samples_leaf", 5)
            return RandomForestClassifier(**params)
        if self.config.model_type == "random_forest_regressor":
            params.setdefault("n_estimators", 200)
            params.setdefault("min_samples_leaf", 5)
            return RandomForestRegressor(**params)
        if self.config.model_type == "hist_gradient_boosting_classifier":
            return HistGradientBoostingClassifier(**params)
        if self.config.model_type == "hist_gradient_boosting_regressor":
            return HistGradientBoostingRegressor(**params)
        raise ValueError(f"Unsupported model_type: {self.config.model_type}")

    @staticmethod
    def _feature_importance(pipeline: Pipeline, feature_columns: list[str]) -> pd.Series | None:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.ravel(np.abs(model.coef_))
        else:
            return None
        index = feature_columns
        if len(values) != len(index):
            try:
                index = list(pipeline[:-1].get_feature_names_out(feature_columns))
            except Exception:
                index = feature_columns[: len(values)]
        return pd.Series(values, index=index, name="feature_importance").sort_values(ascending=False)

    @staticmethod
    def _date_index(index: pd.Index) -> pd.Index:
        if isinstance(index, pd.MultiIndex):
            return pd.Index(index.get_level_values("date"))
        return pd.Index(index)
