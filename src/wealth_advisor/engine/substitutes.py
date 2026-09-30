from datetime import date, timedelta


_DEFAULT_SUBSTITUTES: dict[str, list[str]] = {
    "SPY": ["SCHX", "IWB"],
    "VOO": ["SCHX", "IWB"],
    "QQQ": ["VGT", "XLK"],
    "VTI": ["ITOT"],
    "BND": ["AGG"],
    "VEA": ["IEFA"],
}


class SubstituteRegistry:
    """Maps tickers to wash-sale-safe substitutes (different benchmark index families)."""

    def __init__(self, substitutes: dict[str, list[str]] | None = None) -> None:
        self._map: dict[str, list[str]] = (
            substitutes if substitutes is not None else dict(_DEFAULT_SUBSTITUTES)
        )

    def get_substitute(self, symbol: str, active_locks: set[str]) -> str:
        """Return the first unlocked substitute, or 'CASH' when none are available."""
        for candidate in self._map.get(symbol, []):
            if candidate not in active_locks:
                return candidate
        return "CASH"


class WashSaleTracker:
    """Tracks per-ticker lock windows after tax-loss harvests."""

    def __init__(self) -> None:
        self._locks: dict[str, date] = {}

    def record_harvest(
        self, symbol: str, harvest_date: date, lock_days: int = 30
    ) -> None:
        """Lock *symbol* for *lock_days* days starting from *harvest_date*."""
        expiry = harvest_date + timedelta(days=lock_days)
        if symbol not in self._locks or self._locks[symbol] < expiry:
            self._locks[symbol] = expiry

    def is_locked(self, symbol: str, as_of: date) -> bool:
        """True when *symbol* is still inside its lock window as of *as_of*."""
        expiry = self._locks.get(symbol)
        return expiry is not None and as_of <= expiry

    def lock_expires_on(self, symbol: str) -> date | None:
        """Expiry date for *symbol*, or None if not locked."""
        return self._locks.get(symbol)

    def get_active_locks(self, as_of: date) -> set[str]:
        """All symbols whose lock window has not yet expired."""
        return {sym for sym, expiry in self._locks.items() if as_of <= expiry}
