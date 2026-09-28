from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import Field

from wealth_advisor.domain.portfolio import Security
from wealth_advisor.domain.primitives import (
    ClientId,
    DomainModel,
    Money,
    Quantity,
    Symbol,
    TaxRate,
)


class Term(StrEnum):
    SHORT = "short_term"
    LONG = "long_term"


class LotSale(DomainModel):
    """The tax lot, or part of one, a sale relieves; doubles as the broker's lot instruction."""

    security: Security
    acquired_on: date
    quantity: Quantity
    cost_basis: Money
    proceeds: Decimal = Field(ge=0)
    term: Term

    @property
    def gain(self) -> Decimal:
        return self.proceeds - self.cost_basis


class WashSaleWarning(DomainModel):
    symbol: Symbol
    loss_at_risk: Decimal = Field(gt=0)
    reason: str


class TaxImpact(DomainModel):
    """Estimated federal income tax on the gains a set of orders realises."""

    client_id: ClientId
    as_of: date
    lot_sales: tuple[LotSale, ...]
    short_term_gain: Decimal
    long_term_gain: Decimal
    short_term_rate: TaxRate
    long_term_rate: TaxRate
    estimated_tax: Money
    net_capital_loss: Decimal = Field(ge=0)
    wash_sales: tuple[WashSaleWarning, ...]
