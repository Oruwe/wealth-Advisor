"""Helpers around the worked example in `wealth_advisor.demo`: client C-1001's $100k account."""

from datetime import date
from decimal import Decimal

from wealth_advisor.demo import AS_OF as AS_OF
from wealth_advisor.domain.client import ClientProfile
from wealth_advisor.domain.orders import Order, Side
from wealth_advisor.domain.portfolio import AssetClass, Holding, Security, TaxLot
from wealth_advisor.policy.model_portfolio import MODEL_SECURITIES, TARGET_BY_RISK

CONSERVATIVE = TARGET_BY_RISK[3]  # 20% equity, 65% fixed income, 5% commodity, 10% cash
VTI = MODEL_SECURITIES[AssetClass.EQUITY]
BND = MODEL_SECURITIES[AssetClass.FIXED_INCOME]
GLD = MODEL_SECURITIES[AssetClass.COMMODITY]
TSLA = Security(symbol="TSLA", asset_class=AssetClass.EQUITY)


def lot(quantity: str, cost_basis: str, acquired_on: date) -> TaxLot:
    return TaxLot(
        quantity=Decimal(quantity), cost_basis=Decimal(cost_basis), acquired_on=acquired_on
    )


def holding(security: Security, quantity: str, acquired_on: date = AS_OF) -> Holding:
    return Holding(security=security, lots=(lot(quantity, "0", acquired_on),))


def order(side: Side, security: Security, quantity: str, price: str) -> Order:
    return Order(side=side, security=security, quantity=Decimal(quantity), price=Decimal(price))


def profile(client_id: str = "C-1001", cash_reserve: str = "2000.00") -> ClientProfile:
    """The facts `wealth_advisor.demo.DEMO_IPS` states: risk 3, 10 years, 24% bracket."""
    return ClientProfile(
        client_id=client_id,
        risk_tolerance=3,
        time_horizon_years=10,
        cash_reserve=Decimal(cash_reserve),
        marginal_tax_rate=Decimal("0.24"),
    )
