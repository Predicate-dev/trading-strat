"""Data ingestion, validation, and durable OHLCV caching."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

Provider = Literal["yfinance", "ccxt"]
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Market data provider settings."""

    provider: Provider = "yfinance"
    exchange_id: str = "binance"
    max_retries: int = 3
    retry_sleep_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class DataRequest:
    """OHLCV request and cache settings."""

    symbols: tuple[str, ...]
    start: str | pd.Timestamp
    end: str | pd.Timestamp | None = None
    interval: str = "1d"
    cache_dir: Path = Path(".cache/data")
    use_cache: bool = True
    refresh: bool = False


@dataclass(frozen=True, slots=True)
class DataValidationReport:
    """Validation diagnostics for an OHLCV data frame."""

    rows: int
    duplicate_rows: int
    missing_values: int
    bad_price_rows: int
    bad_volume_rows: int
    monotonic_index: bool
    has_required_columns: bool
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(slots=True)
class DataHandler:
    """Fetch, validate, clean, cache, refresh, and resample OHLCV data."""

    provider: Provider = "yfinance"
    exchange_id: str = "binance"
    max_retries: int = 3
    retry_sleep_seconds: float = 1.0
    cache_dir: Path = Path(".cache/data")

    @classmethod
    def from_config(cls, provider_config: ProviderConfig, cache_dir: Path = Path(".cache/data")) -> DataHandler:
        return cls(
            provider=provider_config.provider,
            exchange_id=provider_config.exchange_id,
            max_retries=provider_config.max_retries,
            retry_sleep_seconds=provider_config.retry_sleep_seconds,
            cache_dir=cache_dir,
        )

    def fetch_historical(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        use_cache: bool = True,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch one symbol, using parquet cache and incremental refresh when possible."""
        request = DataRequest(
            symbols=(symbol,),
            start=start,
            end=end,
            interval=interval,
            cache_dir=self.cache_dir,
            use_cache=use_cache,
            refresh=refresh,
        )
        return self.fetch_one(request, symbol)

    def fetch_one(self, request: DataRequest, symbol: str) -> pd.DataFrame:
        """Fetch one symbol for a request, returning a single-symbol OHLCV frame."""
        cache_path = self._cache_path(request, symbol)
        cached = self._read_cache(cache_path) if request.use_cache else pd.DataFrame()
        requested_start = pd.Timestamp(request.start)
        requested_end = pd.Timestamp(request.end) if request.end is not None else pd.Timestamp.utcnow().tz_localize(None)

        if request.use_cache and not cached.empty and not request.refresh:
            sliced = self._slice_dates(cached, requested_start, requested_end)
            if not sliced.empty and sliced.index.min() <= requested_start and sliced.index.max() >= requested_end.normalize():
                return sliced

        fetch_start = requested_start
        if request.use_cache and request.refresh and not cached.empty:
            last_cached = pd.Timestamp(cached.index.max())
            fetch_start = max(requested_start, last_cached + self._interval_offset(request.interval))

        fresh = pd.DataFrame()
        if cached.empty or fetch_start <= requested_end:
            try:
                fresh = self._fetch_provider(symbol, fetch_start, request.end, request.interval)
            except Exception:
                if cached.empty:
                    raise
                LOGGER.warning("Incremental refresh returned no new data for %s; using existing cache.", symbol)

        combined = self._merge_cached_and_fresh(cached, fresh)
        if request.use_cache and not combined.empty:
            self._write_cache(cache_path, combined)
        return self._slice_dates(combined, requested_start, requested_end)

    def fetch_many(self, request: DataRequest) -> pd.DataFrame:
        """Fetch multiple symbols and return a MultiIndex frame indexed by date and symbol."""
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            symbol_data = self.fetch_one(request, symbol).copy()
            symbol_data["symbol"] = symbol
            frames.append(symbol_data)
        if not frames:
            raise ValueError("DataRequest must include at least one symbol.")
        data = pd.concat(frames).set_index("symbol", append=True).sort_index()
        data.index = data.index.set_names(["date", "symbol"])
        return data

    def validate_ohlcv(self, data: pd.DataFrame) -> DataValidationReport:
        """Validate OHLCV columns, index ordering, duplicate rows, values, and missingness."""
        issues: list[str] = []
        if data.empty:
            issues.append("data is empty")
            return DataValidationReport(0, 0, 0, 0, 0, True, False, tuple(issues))

        columns = {str(column).lower().replace(" ", "_") for column in data.columns}
        has_required = set(OHLCV_COLUMNS).issubset(columns)
        if not has_required:
            issues.append("missing required OHLCV columns")

        index = data.index.get_level_values(0) if isinstance(data.index, pd.MultiIndex) else data.index
        monotonic = bool(index.is_monotonic_increasing)
        if not monotonic:
            issues.append("timestamps are not monotonic increasing")

        duplicate_rows = int(data.index.duplicated().sum())
        if duplicate_rows:
            issues.append("duplicate index rows detected")

        normalized = data.copy()
        normalized.columns = [str(column).lower().replace(" ", "_") for column in normalized.columns]
        missing_values = int(normalized[list(columns.intersection(OHLCV_COLUMNS))].isna().sum().sum())
        if missing_values:
            issues.append("missing OHLCV values detected")

        bad_price_rows = 0
        bad_volume_rows = 0
        if has_required:
            prices = normalized[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
            volume = pd.to_numeric(normalized["volume"], errors="coerce")
            bad_price_mask = (
                prices.isna().any(axis=1)
                | prices.le(0.0).any(axis=1)
                | prices["high"].lt(prices["low"])
                | prices["high"].lt(prices[["open", "close"]].max(axis=1))
                | prices["low"].gt(prices[["open", "close"]].min(axis=1))
            )
            bad_volume_mask = volume.isna() | volume.lt(0.0)
            bad_price_rows = int(bad_price_mask.sum())
            bad_volume_rows = int(bad_volume_mask.sum())
            if bad_price_rows:
                issues.append("bad price rows detected")
            if bad_volume_rows:
                issues.append("bad volume rows detected")

        return DataValidationReport(
            rows=len(data),
            duplicate_rows=duplicate_rows,
            missing_values=missing_values,
            bad_price_rows=bad_price_rows,
            bad_volume_rows=bad_volume_rows,
            monotonic_index=monotonic,
            has_required_columns=has_required,
            issues=tuple(issues),
        )

    def clean_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize columns, sort, remove duplicates, and drop impossible OHLCV rows."""
        if data.empty:
            raise ValueError("Cannot clean an empty OHLCV data frame.")

        clean = data.copy()
        clean.columns = [str(column).lower().replace(" ", "_") for column in clean.columns]
        missing = [column for column in OHLCV_COLUMNS if column not in clean.columns]
        if missing:
            raise ValueError(f"OHLCV data is missing required columns: {missing}")

        clean.index = pd.to_datetime(clean.index)
        clean = clean.loc[~clean.index.duplicated(keep="last")].sort_index()
        clean[OHLCV_COLUMNS] = clean[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
        clean = clean.dropna(how="all")
        price_positive_or_missing = clean[["open", "high", "low", "close"]].gt(0.0) | clean[
            ["open", "high", "low", "close"]
        ].isna()
        high_low_ok = clean["high"].ge(clean["low"]) | clean[["high", "low"]].isna().any(axis=1)
        high_contains_open_close = clean["high"].ge(clean[["open", "close"]].max(axis=1)) | clean[
            ["high", "open", "close"]
        ].isna().any(axis=1)
        low_contains_open_close = clean["low"].le(clean[["open", "close"]].min(axis=1)) | clean[
            ["low", "open", "close"]
        ].isna().any(axis=1)
        volume_ok = clean["volume"].ge(0.0) | clean["volume"].isna()
        clean = clean[
            price_positive_or_missing.all(axis=1)
            & high_low_ok
            & high_contains_open_close
            & low_contains_open_close
            & volume_ok
        ]
        clean = clean.dropna(subset=["open", "high", "low", "close"], how="all")
        return clean[OHLCV_COLUMNS + [column for column in clean.columns if column not in OHLCV_COLUMNS]]

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing OHLCV values with market-data-aware defaults."""
        clean = self.clean_ohlcv(data)
        price_columns = ["open", "high", "low", "close"]
        clean[price_columns] = clean[price_columns].ffill()
        clean["volume"] = clean["volume"].fillna(0.0)
        return clean.dropna(subset=price_columns)

    def resample(self, data: pd.DataFrame, frequency: str = "W-FRI") -> pd.DataFrame:
        """Resample OHLCV bars to a coarser frequency."""
        if isinstance(data.index, pd.MultiIndex):
            frames = []
            for symbol, frame in data.groupby(level="symbol", sort=False):
                resampled = self.resample(frame.droplevel("symbol"), frequency)
                resampled["symbol"] = symbol
                frames.append(resampled)
            result = pd.concat(frames).set_index("symbol", append=True).sort_index()
            result.index = result.index.set_names(["date", "symbol"])
            return result

        clean = self.handle_missing_values(data)
        aggregated = clean.resample(frequency).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        return self.handle_missing_values(aggregated)

    def _fetch_provider(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None,
        interval: str,
    ) -> pd.DataFrame:
        try:
            if self.provider == "yfinance":
                return self._fetch_yfinance(symbol, start, end, interval)
            if self.provider == "ccxt":
                return self._fetch_ccxt(symbol, start, end, interval)
        except Exception as exc:
            LOGGER.exception("Failed to fetch historical data for %s via %s", symbol, self.provider)
            raise RuntimeError(f"Data fetch failed for {symbol} using {self.provider}") from exc
        raise ValueError(f"Unsupported provider: {self.provider}")

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
            yf.set_tz_cache_location(str(self.cache_dir / "yfinance_tz"))
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
                data.index = pd.to_datetime(data.index).tz_localize(None)
                data.index.name = "date"
                return self.handle_missing_values(data)
            except Exception as exc:
                last_error = exc
                LOGGER.warning("yfinance attempt %s/%s failed for %s: %s", attempt, self.max_retries, symbol, exc)
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
                LOGGER.warning("ccxt attempt %s/%s failed for %s on %s: %s", attempt, self.max_retries, symbol, self.exchange_id, exc)
                time.sleep(self.retry_sleep_seconds * attempt)
        else:
            raise RuntimeError(f"ccxt failed after {self.max_retries} attempts for {symbol}") from last_error

        data = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        if data.empty:
            raise ValueError(f"No data returned for {symbol}.")
        data["date"] = pd.to_datetime(data["date"], unit="ms").dt.tz_localize(None)
        data = data.set_index("date")
        if end_ms is not None:
            data = data.loc[data.index <= pd.Timestamp(end)]
        return self.handle_missing_values(data)

    def _cache_path(self, request: DataRequest, symbol: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        return Path(request.cache_dir) / self.provider / request.interval / f"{safe_symbol}.parquet"

    @staticmethod
    def _read_cache(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            data = pd.read_parquet(path)
            data.index = pd.to_datetime(data.index)
            data.index.name = "date"
            return data.sort_index()
        except Exception as exc:
            LOGGER.warning("Failed reading parquet cache %s: %s", path, exc)
            return pd.DataFrame()

    @staticmethod
    def _write_cache(path: Path, data: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.sort_index().to_parquet(path)

    @staticmethod
    def _merge_cached_and_fresh(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        if cached.empty:
            return fresh
        if fresh.empty:
            return cached
        combined = pd.concat([cached, fresh])
        return combined.loc[~combined.index.duplicated(keep="last")].sort_index()

    @staticmethod
    def _slice_dates(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if data.empty:
            return data
        start = start.tz_localize(None) if start.tzinfo else start
        end = end.tz_localize(None) if end.tzinfo else end
        return data.loc[(data.index >= start) & (data.index <= end)]

    @staticmethod
    def _interval_offset(interval: str) -> pd.Timedelta:
        if interval.endswith("m"):
            return pd.Timedelta(minutes=int(interval[:-1]))
        if interval.endswith("h"):
            return pd.Timedelta(hours=int(interval[:-1]))
        if interval.endswith("d"):
            return pd.Timedelta(days=int(interval[:-1]))
        if interval.endswith("wk"):
            return pd.Timedelta(weeks=int(interval[:-2]))
        return pd.Timedelta(days=1)
