"""The worked example: client C-1001, a conservative investor with a $100k taxable account."""

from datetime import date
from decimal import Decimal

from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.portfolio import AssetClass, Holding, Portfolio, Security, TaxLot
from wealth_advisor.policy.model_portfolio import MODEL_SECURITIES

AS_OF = date(2026, 9, 25)

DEMO_IPS = """\
Investment Policy Statement for client C-1001, prepared on 20 September 2026.

Objective: preserve capital first, then grow it modestly. The client plans to retire in about
ten years.
Risk tolerance: 3 on a scale of 1 to 10. A loss of more than 10% in one year would be
unacceptable.
Time horizon: 10 years.
Liquidity: keep at least $2,000 in cash at all times for emergencies.
Taxes: the client is in the 24% federal income-tax bracket, and the account is taxable.
Restrictions: no cryptocurrencies, options or other speculative instruments.
"""


def _lot(quantity: str, cost_basis: str, acquired_on: date) -> TaxLot:
    return TaxLot(
        quantity=Decimal(quantity), cost_basis=Decimal(cost_basis), acquired_on=acquired_on
    )


def demo_portfolio() -> Portfolio:
    """$5k cash, $25k TSLA (outside the model), $30k VTI, $35k BND, $5k GLD, bought over years."""
    return Portfolio(
        client_id="C-1001",
        as_of=AS_OF,
        cash=Decimal("5000.00"),
        holdings=(
            Holding(
                security=Security(symbol="TSLA", asset_class=AssetClass.EQUITY),
                lots=(
                    _lot("60", "10800.00", date(2024, 3, 15)),
                    _lot("40", "12000.00", date(2026, 5, 1)),
                ),
            ),
            Holding(
                security=MODEL_SECURITIES[AssetClass.EQUITY],
                lots=(
                    _lot("70", "14000.00", date(2023, 6, 1)),
                    _lot("30", "9300.00", date(2026, 9, 1)),
                ),
            ),
            Holding(
                security=MODEL_SECURITIES[AssetClass.FIXED_INCOME],
                lots=(_lot("500", "36500.00", date(2022, 1, 10)),),
            ),
            Holding(
                security=MODEL_SECURITIES[AssetClass.COMMODITY],
                lots=(_lot("25", "4000.00", date(2023, 11, 20)),),
            ),
        ),
    )


def demo_prices() -> PriceSnapshot:
    return PriceSnapshot(
        as_of=AS_OF,
        prices={
            "TSLA": Decimal("250"),
            "VTI": Decimal("300"),
            "BND": Decimal("70"),
            "GLD": Decimal("200"),
        },
    )
