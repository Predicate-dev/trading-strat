"""Modular algorithmic trading research toolkit."""

from trading_strat.backtester import BacktestConfig, BacktestResult, VectorizedBacktester
from trading_strat.data import DataHandler
from trading_strat.features import FeatureGenerator
from trading_strat.plotter import Plotter
from trading_strat.regime import RegimeFilter
from trading_strat.risk import RiskManager
from trading_strat.slippage import SlippageModel
from trading_strat.strategies import BaseStrategy, MeanReversionStrategy, MomentumStrategy, RegimeAwareMomentumStrategy
from trading_strat.walk_forward import WalkForwardResult, WalkForwardSplit, WalkForwardValidator

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BaseStrategy",
    "DataHandler",
    "FeatureGenerator",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "Plotter",
    "RegimeAwareMomentumStrategy",
    "RegimeFilter",
    "RiskManager",
    "SlippageModel",
    "VectorizedBacktester",
    "WalkForwardResult",
    "WalkForwardSplit",
    "WalkForwardValidator",
]
