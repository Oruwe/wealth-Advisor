from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from strategies import portfolios

from wealth_advisor.domain.market import MissingPriceError, PriceSnapshot
from wealth_advisor.domain.portfolio import (
    AssetClass,
    Holding,
    Portfolio,
    Security,
    TargetAllocation,
    TaxLot,
)

AS_OF = date(2026, 9, 25)


def lot(quantity: str, cost_basis: str, acquired_on: date = date(2024, 1, 2)) -> TaxLot:
    return TaxLot(
        quantity=Decimal(quantity), cost_basis=Decimal(cost_basis), acquired_on=acquired_on
    )


def holding(symbol: str, asset_class: AssetClass, *lots: TaxLot) -> Holding:
    return Holding(security=Security(symbol=symbol, asset_class=asset_class), lots=lots)


@pytest.fixture
def portfolio() -> Portfolio:
    return Portfolio(
        client_id="C-1001",
        as_of=AS_OF,
        cash=Decimal("5000.00"),
        holdings=(
            holding(
                "VTI",
                AssetClass.EQUITY,
                lot("10.5", "2100.00"),
                lot("4", "1000.00", date(2025, 6, 2)),
            ),
            holding("BND", AssetClass.FIXED_INCOME, lot("100", "7300.00")),
            holding("SGOV", AssetClass.CASH, lot("20", "2000.00")),
        ),
    )


@pytest.fixture
def prices() -> PriceSnapshot:
    return PriceSnapshot(
        as_of=AS_OF,
        prices={"VTI": Decimal("312.4567"), "BND": Decimal("72.10"), "SGOV": Decimal("100.52")},
    )


def test_holding_quantity_is_the_sum_of_its_lots(portfolio: Portfolio) -> None:
    vti = next(h for h in portfolio.holdings if h.security.symbol == "VTI")

    assert vti.quantity == Decimal("14.5")


def test_values_every_asset_class_exactly(portfolio: Portfolio, prices: PriceSnapshot) -> None:
    assert portfolio.value_by_asset_class(prices) == {
        AssetClass.EQUITY: Decimal("4530.62215"),
        AssetClass.FIXED_INCOME: Decimal("7210.00"),
        AssetClass.COMMODITY: Decimal("0"),
        AssetClass.CASH: Decimal("7010.40"),
    }


def test_market_value_is_exact_not_rounded(portfolio: Portfolio, prices: PriceSnapshot) -> None:
    assert portfolio.market_value(prices) == Decimal("18751.02215")


def test_valuation_fails_loudly_when_a_price_is_missing(portfolio: Portfolio) -> None:
    prices = PriceSnapshot(
        as_of=AS_OF, prices={"VTI": Decimal("312.4567"), "SGOV": Decimal("100.52")}
    )

    with pytest.raises(MissingPriceError, match="no price for BND"):
        portfolio.market_value(prices)


def test_valuation_rejects_prices_from_another_date(
    portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    stale = PriceSnapshot(as_of=date(2026, 9, 24), prices=prices.prices)

    with pytest.raises(ValueError, match="prices are as of 2026-09-24"):
        portfolio.market_value(stale)


def test_rejects_the_same_symbol_in_two_holdings() -> None:
    with pytest.raises(ValidationError, match="only one holding: VTI"):
        Portfolio(
            client_id="C-1001",
            as_of=AS_OF,
            cash=Decimal(0),
            holdings=(
                holding("VTI", AssetClass.EQUITY, lot("1", "100.00")),
                holding("vti", AssetClass.EQUITY, lot("2", "200.00")),
            ),
        )


def test_rejects_lots_acquired_after_the_portfolio_date() -> None:
    with pytest.raises(ValidationError, match="lots acquired after 2026-09-25: VTI"):
        Portfolio(
            client_id="C-1001",
            as_of=AS_OF,
            cash=Decimal(0),
            holdings=(holding("VTI", AssetClass.EQUITY, lot("1", "100.00", date(2026, 9, 26))),),
        )


def test_a_holding_needs_at_least_one_lot() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        Holding(security=Security(symbol="VTI", asset_class=AssetClass.EQUITY), lots=())


def test_target_allocation_maps_each_asset_class_to_its_weight() -> None:
    allocation = TargetAllocation(
        equity=Decimal("0.20"),
        fixed_income=Decimal("0.60"),
        commodity=Decimal("0.05"),
        cash=Decimal("0.15"),
    )

    assert allocation.weights == {
        AssetClass.EQUITY: Decimal("0.20"),
        AssetClass.FIXED_INCOME: Decimal("0.60"),
        AssetClass.COMMODITY: Decimal("0.05"),
        AssetClass.CASH: Decimal("0.15"),
    }


def test_target_allocation_rejects_weights_that_do_not_add_up_to_one() -> None:
    with pytest.raises(ValidationError, match=r"add up to exactly 1, got 0\.95"):
        TargetAllocation(
            equity=Decimal("0.20"),
            fixed_income=Decimal("0.60"),
            commodity=Decimal("0.05"),
            cash=Decimal("0.10"),
        )


@given(portfolios())
def test_json_round_trip_preserves_every_value(portfolio: Portfolio) -> None:
    assert Portfolio.model_validate_json(portfolio.model_dump_json()) == portfolio


@given(portfolios(), st.data())
def test_input_order_never_changes_the_portfolio(portfolio: Portfolio, data: st.DataObject) -> None:
    shuffled = tuple(
        Holding(security=holding.security, lots=tuple(data.draw(st.permutations(holding.lots))))
        for holding in data.draw(st.permutations(portfolio.holdings))
    )

    rebuilt = Portfolio(
        client_id=portfolio.client_id, as_of=portfolio.as_of, cash=portfolio.cash, holdings=shuffled
    )

    assert rebuilt == portfolio
