from datetime import date
from decimal import Decimal, localcontext

import pytest
from hypothesis import given
from strategies import MODEL_SYMBOLS, RebalanceCase, rebalance_cases

from wealth_advisor.domain.client import ClientProfile
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.orders import Order, RebalanceProposal, Side
from wealth_advisor.domain.portfolio import (
    AssetClass,
    Holding,
    Portfolio,
    Security,
    TargetAllocation,
    TaxLot,
)
from wealth_advisor.policy.model_portfolio import MODEL_SECURITIES, TARGET_BY_RISK
from wealth_advisor.rebalancer import RebalanceError, propose_rebalance, rebalance

AS_OF = date(2026, 9, 25)
CONSERVATIVE = TARGET_BY_RISK[3]  # 20% equity, 65% fixed income, 5% commodity, 10% cash
VTI = MODEL_SECURITIES[AssetClass.EQUITY]
BND = MODEL_SECURITIES[AssetClass.FIXED_INCOME]
GLD = MODEL_SECURITIES[AssetClass.COMMODITY]
TSLA = Security(symbol="TSLA", asset_class=AssetClass.EQUITY)


def holding(security: Security, quantity: str, acquired_on: date = AS_OF) -> Holding:
    lot = TaxLot(quantity=Decimal(quantity), cost_basis=Decimal(0), acquired_on=acquired_on)
    return Holding(security=security, lots=(lot,))


def order(side: Side, security: Security, quantity: str, price: str) -> Order:
    return Order(side=side, security=security, quantity=Decimal(quantity), price=Decimal(price))


def profile(client_id: str = "C-1001") -> ClientProfile:
    return ClientProfile(
        client_id=client_id,
        risk_tolerance=3,
        time_horizon_years=10,
        cash_reserve=Decimal("2000.00"),
        marginal_tax_rate=Decimal("0.24"),
    )


@pytest.fixture
def portfolio() -> Portfolio:
    """$100k: $5k cash, $25k TSLA (outside the model), $30k VTI, $35k BND, $5k GLD."""
    return Portfolio(
        client_id="C-1001",
        as_of=AS_OF,
        cash=Decimal("5000.00"),
        holdings=(
            holding(TSLA, "100"),
            holding(VTI, "100"),
            holding(BND, "500"),
            holding(GLD, "25"),
        ),
    )


@pytest.fixture
def prices() -> PriceSnapshot:
    return PriceSnapshot(
        as_of=AS_OF,
        prices={
            "TSLA": Decimal("250"),
            "VTI": Decimal("300"),
            "BND": Decimal("70"),
            "GLD": Decimal("200"),
        },
    )


def test_exits_holdings_outside_the_model_and_trades_the_rest_to_target(
    portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    proposal = rebalance(portfolio, prices, CONSERVATIVE, cash_reserve=Decimal("2000.00"))

    assert proposal.orders == (
        order(Side.SELL, TSLA, "100", "250"),
        order(Side.SELL, VTI, "33", "300"),
        order(Side.BUY, BND, "429", "70"),
    )
    assert proposal.value_after == {
        AssetClass.EQUITY: Decimal("20100"),
        AssetClass.FIXED_INCOME: Decimal("65030"),
        AssetClass.COMMODITY: Decimal("5000"),
        AssetClass.CASH: Decimal("9870.00"),
    }


def test_keeps_the_cash_reserve_even_when_nearest_rounding_would_breach_it(
    portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    proposal = rebalance(portfolio, prices, CONSERVATIVE, cash_reserve=Decimal("15000.00"))

    # Rounding every trade to the nearest share would sell 1 GLD and leave $14,910 in cash.
    assert proposal.orders == (
        order(Side.SELL, GLD, "2", "200"),
        order(Side.SELL, TSLA, "100", "250"),
        order(Side.SELL, VTI, "37", "300"),
        order(Side.BUY, BND, "377", "70"),
    )
    assert proposal.cash_after == Decimal("15110.00")


def test_prefers_the_smaller_trade_when_two_roundings_drift_equally() -> None:
    at_100 = {"VTI": Decimal(100), "BND": Decimal(100), "GLD": Decimal(100)}
    portfolio = Portfolio(
        client_id="C-1001",
        as_of=AS_OF,
        cash=Decimal(0),
        holdings=(holding(VTI, "10"), holding(BND, "10")),
    )
    target = TargetAllocation(
        equity=Decimal("0.425"),
        fixed_income=Decimal("0.575"),
        commodity=Decimal(0),
        cash=Decimal(0),
    )

    proposal = rebalance(portfolio, PriceSnapshot(as_of=AS_OF, prices=at_100), target, Decimal(0))

    # Both ideal trades are 1.5 shares, so every affordable rounding drifts equally.
    assert proposal.orders == (
        order(Side.SELL, VTI, "1", "100"),
        order(Side.BUY, BND, "1", "100"),
    )


def test_refuses_a_reserve_larger_than_the_portfolio(
    portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    with pytest.raises(RebalanceError, match="exceeds portfolio value"):
        rebalance(portfolio, prices, CONSERVATIVE, cash_reserve=Decimal("100000.01"))


def test_refuses_a_holding_classified_differently_from_the_model(prices: PriceSnapshot) -> None:
    mislabelled = Portfolio(
        client_id="C-1001",
        as_of=AS_OF,
        cash=Decimal(0),
        holdings=(holding(Security(symbol="VTI", asset_class=AssetClass.FIXED_INCOME), "1"),),
    )

    with pytest.raises(RebalanceError, match="VTI is held as fixed_income"):
        rebalance(mislabelled, prices, CONSERVATIVE, cash_reserve=Decimal(0))


def test_proposal_uses_the_target_for_the_clients_risk_score(
    portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    assert propose_rebalance(profile(), portfolio, prices) == rebalance(
        portfolio, prices, CONSERVATIVE, cash_reserve=Decimal("2000.00")
    )


def test_refuses_a_profile_for_a_different_client(
    portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    with pytest.raises(RebalanceError, match="profile is for C-2002"):
        propose_rebalance(profile("C-2002"), portfolio, prices)


def test_result_does_not_depend_on_the_callers_decimal_precision(
    portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    expected = rebalance(portfolio, prices, CONSERVATIVE, cash_reserve=Decimal("15000.00"))

    with localcontext(prec=6):
        assert rebalance(portfolio, prices, CONSERVATIVE, Decimal("15000.00")) == expected


@given(rebalance_cases())
def test_cash_never_drops_below_the_reserve(case: RebalanceCase) -> None:
    proposal = rebalance(*case)

    assert proposal.target_values[AssetClass.CASH] >= case.cash_reserve
    assert proposal.cash_after >= case.cash_reserve


@given(rebalance_cases())
def test_trading_at_snapshot_prices_conserves_value(case: RebalanceCase) -> None:
    proposal = rebalance(*case)
    total = case.portfolio.market_value(case.prices)

    assert sum(proposal.value_after.values()) == total
    assert sum(proposal.target_values.values()) == total


@given(rebalance_cases())
def test_trades_whole_shares_except_full_exits_and_never_oversells(case: RebalanceCase) -> None:
    held = {h.security.symbol: h.quantity for h in case.portfolio.holdings}

    for placed in rebalance(*case).orders:
        if placed.side is Side.SELL:
            assert placed.quantity <= held[placed.security.symbol]
        assert placed.quantity % 1 == 0 or placed.quantity == held.get(placed.security.symbol)


@given(rebalance_cases())
def test_sells_every_holding_outside_the_model_in_full(case: RebalanceCase) -> None:
    sold = {o.security.symbol: o.quantity for o in rebalance(*case).orders if o.side is Side.SELL}

    for outside in case.portfolio.holdings:
        if outside.security.symbol not in MODEL_SYMBOLS:
            assert sold[outside.security.symbol] == outside.quantity


@given(rebalance_cases())
def test_each_asset_class_lands_within_one_share_of_its_target(case: RebalanceCase) -> None:
    proposal = rebalance(*case)

    for asset_class, security in MODEL_SECURITIES.items():
        miss = abs(proposal.value_after[asset_class] - proposal.target_values[asset_class])
        assert miss < case.prices.price_of(security.symbol)


def apply(portfolio: Portfolio, proposal: RebalanceProposal) -> Portfolio:
    quantities = {h.security: h.quantity for h in portfolio.holdings}
    for placed in proposal.orders:
        change = placed.quantity if placed.side is Side.BUY else -placed.quantity
        quantities[placed.security] = quantities.get(placed.security, Decimal(0)) + change
    return Portfolio(
        client_id=portfolio.client_id,
        as_of=portfolio.as_of,
        cash=proposal.cash_after,
        holdings=tuple(holding(s, str(q), portfolio.as_of) for s, q in quantities.items() if q),
    )


@given(rebalance_cases(whole_shares=True))
def test_rebalancing_twice_changes_nothing(case: RebalanceCase) -> None:
    first = rebalance(*case)

    second = rebalance(apply(case.portfolio, first), case.prices, case.target, case.cash_reserve)

    assert second.orders == ()
