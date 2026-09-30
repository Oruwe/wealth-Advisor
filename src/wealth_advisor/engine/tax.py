from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum


class TradeAction(StrEnum):
    BUY = "buy"
    SELL = "sell"


class TaxJurisdiction(StrEnum):
    US = "US"
    INDIA = "INDIA"


@dataclass(frozen=True)
class TaxLot:
    symbol: str
    shares: Decimal
    cost_basis: Decimal
    purchase_date: date
    asset_jurisdiction: TaxJurisdiction = field(default=TaxJurisdiction.US)
    investor_jurisdiction: TaxJurisdiction = field(default=TaxJurisdiction.US)


@dataclass(frozen=True)
class Trade:
    symbol: str
    action: TradeAction
    shares: Decimal
    estimated_tax_impact: Decimal
    routing_reason: str | None = None
