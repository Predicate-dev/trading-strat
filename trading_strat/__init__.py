"""Modular algorithmic trading research toolkit."""

from trading_strat.backtester import BacktestConfig, BacktestResult, VectorizedBacktester
from trading_strat.data import DataHandler, DataRequest, DataValidationReport, ProviderConfig
from trading_strat.features import FeatureGenerator, TargetBuilder
from trading_strat.ml import DatasetBuilder, ModelConfig, TimeSeriesModelTrainer, TrainedModel
from trading_strat.plotter import Plotter
from trading_strat.regime import RegimeFilter
from trading_strat.risk import RiskManager
from trading_strat.slippage import SlippageModel
from trading_strat.strategies import (
    AdaptiveTrendStrategy,
    BaseStrategy,
    EnsembleStrategy,
    MLSignalStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    RegimeAwareMomentumStrategy,
)
from trading_strat.walk_forward import WalkForwardResult, WalkForwardSplit, WalkForwardValidator

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "AdaptiveTrendStrategy",
    "BaseStrategy",
    "DataHandler",
    "DataRequest",
    "DataValidationReport",
    "DatasetBuilder",
    "EnsembleStrategy",
    "FeatureGenerator",
    "MLSignalStrategy",
    "MeanReversionStrategy",
    "ModelConfig",
    "MomentumStrategy",
    "Plotter",
    "ProviderConfig",
    "RegimeAwareMomentumStrategy",
    "RegimeFilter",
    "RiskManager",
    "SlippageModel",
    "TargetBuilder",
    "TimeSeriesModelTrainer",
    "TrainedModel",
    "VectorizedBacktester",
    "WalkForwardResult",
    "WalkForwardSplit",
    "WalkForwardValidator",
]
