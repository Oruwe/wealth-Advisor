from collections.abc import Mapping
from datetime import date
from decimal import Decimal

from wealth_advisor.domain.primitives import DomainModel, Price, Symbol


class MissingPriceError(LookupError):
    """A valuation needed a price that the snapshot does not have."""


class PriceSnapshot(DomainModel):
    """Market prices for a single date."""

    as_of: date
    prices: Mapping[Symbol, Price]

    def price_of(self, symbol: str) -> Decimal:
        try:
            return self.prices[symbol]
        except KeyError:
            raise MissingPriceError(f"no price for {symbol} as of {self.as_of}") from None
