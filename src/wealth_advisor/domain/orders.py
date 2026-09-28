from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import Field

from wealth_advisor.domain.portfolio import AssetClass, Security, TargetAllocation
from wealth_advisor.domain.primitives import ClientId, DomainModel, Price, Quantity


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Order(DomainModel):
    """A proposed trade, sized at the snapshot price."""

    side: Side
    security: Security
    quantity: Quantity
    price: Price


class RebalanceProposal(DomainModel):
    """Orders that move a portfolio to its target, with the dollar targets and resulting values."""

    client_id: ClientId
    as_of: date
    target: TargetAllocation
    target_values: dict[AssetClass, Decimal]
    orders: tuple[Order, ...]
    value_after: dict[AssetClass, Decimal]
    cash_after: Decimal = Field(ge=0)
