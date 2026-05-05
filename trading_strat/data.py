"""Data ingestion and OHLCV preparation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

Provider = Literal["yfinance", "ccxt"]

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DataHandler:
    """Fetch, clean, impute, and resample OHLCV data.

    The class keeps provider-specific API details at the boundary of the
    system so feature, strategy, and backtest modules only depend on clean
    Pandas data frames.
    """

    provider: Provider = "yfinance"
    exchange_id: str = "binance"
    max_retries: int = 3
    retry_sleep_seconds: float = 1.0
    cache_dir: Path = Path(".cache/yfinance")

    def fetch_historical(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars from the configured provider."""
        try:
            if self.provider == "yfinance":
                return self._fetch_yfinance(symbol, start, end, interval)
            if self.provider == "ccxt":
                return self._fetch_ccxt(symbol, start, end, interval)
        except Exception as exc:
            LOGGER.exception("Failed to fetch historical data for %s via %s", symbol, self.provider)
            raise RuntimeError(f"Data fetch failed for {symbol} using {self.provider}") from exc

        raise ValueError(f"Unsupported provider: {self.provider}")

    def clean_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize columns, remove invalid rows, and sort by timestamp."""
        if data.empty:
            raise ValueError("Cannot clean an empty OHLCV data frame.")

        clean = data.copy()
        clean.columns = [str(column).lower().replace(" ", "_") for column in clean.columns]

        rename_map = {
            "adj_close": "adj_close",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        clean = clean.rename(columns=rename_map)

        required = ["open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in clean.columns]
        if missing:
            raise ValueError(f"OHLCV data is missing required columns: {missing}")

        clean = clean.loc[~clean.index.duplicated(keep="last")].sort_index()
        clean = clean[required + [column for column in clean.columns if column not in required]]
        clean[required] = clean[required].apply(pd.to_numeric, errors="coerce")
        clean = clean[(clean["close"] > 0) & (clean["high"] >= clean["low"])]
        clean = clean.dropna(how="all")
        return clean

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing OHLCV values with market-data-aware defaults."""
        clean = self.clean_ohlcv(data)
        price_columns = ["open", "high", "low", "close"]
        clean[price_columns] = clean[price_columns].ffill()
        clean["volume"] = clean["volume"].fillna(0.0)
        return clean.dropna(subset=price_columns)

    def resample(self, data: pd.DataFrame, frequency: str = "W-FRI") -> pd.DataFrame:
        """Resample OHLCV bars to a coarser frequency."""
        clean = self.handle_missing_values(data)
        aggregated = clean.resample(frequency).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        return self.handle_missing_values(aggregated)

    def _fetch_yfinance(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None,
        interval: str,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("Install yfinance to use provider='yfinance'.") from exc

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(self.cache_dir))
        except Exception as exc:
            LOGGER.debug("Unable to set yfinance cache location to %s: %s", self.cache_dir, exc)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                data = yf.download(
                    symbol,
                    start=pd.Timestamp(start).date().isoformat(),
                    end=None if end is None else pd.Timestamp(end).date().isoformat(),
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if data.empty:
                    raise ValueError(f"No data returned for {symbol}.")

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data.index = pd.to_datetime(data.index)
                data.index.name = "date"
                return self.handle_missing_values(data)
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "yfinance attempt %s/%s failed for %s: %s",
                    attempt,
                    self.max_retries,
                    symbol,
                    exc,
                )
                time.sleep(self.retry_sleep_seconds * attempt)

        raise RuntimeError(f"yfinance failed after {self.max_retries} attempts for {symbol}") from last_error

    def _fetch_ccxt(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None,
        interval: str,
    ) -> pd.DataFrame:
        try:
            import ccxt
        except ImportError as exc:
            raise ImportError("Install ccxt to use provider='ccxt'.") from exc

        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class({"enableRateLimit": True})
        since = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = None if end is None else int(pd.Timestamp(end).timestamp() * 1000)
        rows: list[list[float]] = []

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                while True:
                    batch = exchange.fetch_ohlcv(symbol, timeframe=interval, since=since, limit=1000)
                    if not batch:
                        break
                    rows.extend(batch)
                    since = int(batch[-1][0]) + 1
                    if end_ms is not None and since >= end_ms:
                        break
                    if len(batch) < 1000:
                        break
                break
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "ccxt attempt %s/%s failed for %s on %s: %s",
                    attempt,
                    self.max_retries,
                    symbol,
                    self.exchange_id,
                    exc,
                )
                time.sleep(self.retry_sleep_seconds * attempt)
        else:
            raise RuntimeError(f"ccxt failed after {self.max_retries} attempts for {symbol}") from last_error

        data = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        if data.empty:
            raise ValueError(f"No data returned for {symbol}.")

        data["date"] = pd.to_datetime(data["date"], unit="ms")
        data = data.set_index("date")
        if end_ms is not None:
            data = data.loc[data.index <= pd.Timestamp(end)]
        return self.handle_missing_values(data)
