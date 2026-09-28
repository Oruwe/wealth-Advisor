from collections import Counter
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Self

from pydantic import Field, field_validator, model_validator

from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.primitives import (
    ClientId,
    DomainModel,
    Money,
    Quantity,
    Symbol,
    Weight,
)


class AssetClass(StrEnum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CASH = "cash"


class Security(DomainModel):
    symbol: Symbol
    asset_class: AssetClass


class TargetAllocation(DomainModel):
    """Target weight for each asset class. The weights must add up to exactly 1."""

    equity: Weight
    fixed_income: Weight
    commodity: Weight
    cash: Weight

    @property
    def weights(self) -> dict[AssetClass, Decimal]:
        return {asset_class: getattr(self, asset_class.value) for asset_class in AssetClass}

    @model_validator(mode="after")
    def _weights_add_up_to_one(self) -> Self:
        total = sum(self.weights.values(), start=Decimal(0))
        if total != 1:
            raise ValueError(f"weights must add up to exactly 1, got {total}")
        return self


class TaxLot(DomainModel):
    quantity: Quantity
    cost_basis: Money
    acquired_on: date


class Holding(DomainModel):
    security: Security
    lots: tuple[TaxLot, ...] = Field(min_length=1)

    @field_validator("lots")
    @classmethod
    def _oldest_lot_first(cls, lots: tuple[TaxLot, ...]) -> tuple[TaxLot, ...]:
        return tuple(sorted(lots, key=lambda lot: (lot.acquired_on, lot.quantity, lot.cost_basis)))

    @property
    def quantity(self) -> Decimal:
        return sum((lot.quantity for lot in self.lots), start=Decimal(0))


class Portfolio(DomainModel):
    """What a client owns on one date, in a canonical order so equal portfolios compare equal."""

    client_id: ClientId
    as_of: date
    cash: Money
    holdings: tuple[Holding, ...] = ()

    @field_validator("holdings")
    @classmethod
    def _sorted_by_symbol(cls, holdings: tuple[Holding, ...]) -> tuple[Holding, ...]:
        return tuple(sorted(holdings, key=lambda holding: holding.security.symbol))

    @model_validator(mode="after")
    def _one_holding_per_symbol(self) -> Self:
        counts = Counter(holding.security.symbol for holding in self.holdings)
        duplicates = sorted(symbol for symbol, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"each symbol must be in only one holding: {', '.join(duplicates)}")
        return self

    @model_validator(mode="after")
    def _no_lots_after_as_of(self) -> Self:
        late = sorted(
            {
                holding.security.symbol
                for holding in self.holdings
                for lot in holding.lots
                if lot.acquired_on > self.as_of
            }
        )
        if late:
            raise ValueError(f"lots acquired after {self.as_of}: {', '.join(late)}")
        return self

    def value_by_asset_class(self, prices: PriceSnapshot) -> dict[AssetClass, Decimal]:
        if prices.as_of != self.as_of:
            raise ValueError(f"prices are as of {prices.as_of}, portfolio is as of {self.as_of}")
        values = dict.fromkeys(AssetClass, Decimal(0))
        values[AssetClass.CASH] += self.cash
        for holding in self.holdings:
            price = prices.price_of(holding.security.symbol)
            values[holding.security.asset_class] += holding.quantity * price
        return values

    def market_value(self, prices: PriceSnapshot) -> Decimal:
        return sum(self.value_by_asset_class(prices).values(), start=Decimal(0))
