import datetime as dt
from typing import Any

from puppy.common.schemas import MarketStep
from .duckdb_store import DuckDBMarketStore


class MultiAssetEnvironment:
    def __init__(
        self,
        market_store: DuckDBMarketStore,
        symbols: list[str],
        dates: list[dt.date] | None = None,
    ) -> None:
        self.market_store = market_store
        self.symbols = symbols
        self.current_index = 0
        
        if dates is not None:
            self.dates = sorted(dates)
        else:
            self.dates = self._fetch_dates_from_store()

    def _fetch_dates_from_store(self) -> list[dt.date]:
        self.market_store.connect()
        if not self.symbols:
            return []
        
        symbols_str = ", ".join(f"'{s}'" for s in self.symbols)
        query = f"SELECT DISTINCT date FROM prices WHERE symbol IN ({symbols_str}) ORDER BY date"
        rows = self.market_store.conn.execute(query).fetchall()
        return [row[0] if isinstance(row[0], dt.date) else row[0].date() for row in rows]

    def get_market_step(self, date: dt.date, as_of: dt.datetime | None = None) -> MarketStep:
        """Retrieve the market step for a specific date."""
        return self.market_store.get_market_step(date, self.symbols, as_of=as_of)

    def step(self, date: dt.date | None = None, as_of: dt.datetime | None = None) -> MarketStep:
        """Advance the environment by one step or query a specific date."""
        if date is not None:
            # If an explicit date is provided, find it in our list and set the index
            if date in self.dates:
                self.current_index = self.dates.index(date)
            return self.get_market_step(date, as_of=as_of)
            
        if self.current_index >= len(self.dates):
            raise IndexError("Environment simulation has ended. Please reset().")
            
        current_date = self.dates[self.current_index]
        step_data = self.get_market_step(current_date, as_of=as_of)
        self.current_index += 1
        return step_data

    def reset(self) -> None:
        """Reset the environment iteration to the beginning."""
        self.current_index = 0

    @property
    def simulation_length(self) -> int:
        return len(self.dates)

    @property
    def is_done(self) -> bool:
        return self.current_index >= len(self.dates)
