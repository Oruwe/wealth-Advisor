"""The worked example shared by the tests: a $100k account for a conservative client, C-1001."""

from datetime import date
from decimal import Decimal

from wealth_advisor.domain.client import ClientProfile
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.orders import Order, Side
from wealth_advisor.domain.portfolio import AssetClass, Holding, Portfolio, Security, TaxLot
from wealth_advisor.policy.model_portfolio import MODEL_SECURITIES, TARGET_BY_RISK

AS_OF = date(2026, 9, 25)
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
    return ClientProfile(
        client_id=client_id,
        risk_tolerance=3,
        time_horizon_years=10,
        cash_reserve=Decimal(cash_reserve),
        marginal_tax_rate=Decimal("0.24"),
    )


def example_portfolio() -> Portfolio:
    """$5k cash, $25k TSLA (outside the model), $30k VTI, $35k BND, $5k GLD, bought over years."""
    return Portfolio(
        client_id="C-1001",
        as_of=AS_OF,
        cash=Decimal("5000.00"),
        holdings=(
            Holding(
                security=TSLA,
                lots=(
                    lot("60", "10800.00", date(2024, 3, 15)),
                    lot("40", "12000.00", date(2026, 5, 1)),
                ),
            ),
            Holding(
                security=VTI,
                lots=(
                    lot("70", "14000.00", date(2023, 6, 1)),
                    lot("30", "9300.00", date(2026, 9, 1)),
                ),
            ),
            Holding(security=BND, lots=(lot("500", "36500.00", date(2022, 1, 10)),)),
            Holding(security=GLD, lots=(lot("25", "4000.00", date(2023, 11, 20)),)),
        ),
    )


def example_prices() -> PriceSnapshot:
    return PriceSnapshot(
        as_of=AS_OF,
        prices={
            "TSLA": Decimal("250"),
            "VTI": Decimal("300"),
            "BND": Decimal("70"),
            "GLD": Decimal("200"),
        },
    )
