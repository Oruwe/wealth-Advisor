from datetime import date
from decimal import Decimal

import pytest
from examples import AS_OF, TSLA, VTI, lot, order, profile
from hypothesis import given
from strategies import RebalanceCase, rebalance_cases

from wealth_advisor.domain.client import ClientProfile
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.orders import Side
from wealth_advisor.domain.portfolio import Holding, Portfolio
from wealth_advisor.domain.tax import LotSale, Term, WashSaleWarning
from wealth_advisor.rebalancer import propose_rebalance, rebalance
from wealth_advisor.tax import NettedGains, TaxError, estimate_tax, net_capital_gains


def vti_account(*lots: tuple[str, str, date], as_of: date = AS_OF) -> Portfolio:
    return Portfolio(
        client_id="C-1001",
        as_of=as_of,
        cash=Decimal(0),
        holdings=(Holding(security=VTI, lots=tuple(lot(*spec) for spec in lots)),),
    )


def test_sells_the_highest_cost_lots_first_and_nets_the_result(
    portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    proposal = propose_rebalance(profile(), portfolio, prices)

    impact = estimate_tax(profile(), portfolio, proposal.orders)

    assert impact.lot_sales == (
        LotSale(
            security=TSLA,
            acquired_on=date(2026, 5, 1),
            quantity=Decimal(40),
            cost_basis=Decimal("12000.00"),
            proceeds=Decimal(10000),
            term=Term.SHORT,
        ),
        LotSale(
            security=TSLA,
            acquired_on=date(2024, 3, 15),
            quantity=Decimal(60),
            cost_basis=Decimal("10800.00"),
            proceeds=Decimal(15000),
            term=Term.LONG,
        ),
        LotSale(
            security=VTI,
            acquired_on=date(2026, 9, 1),
            quantity=Decimal(30),
            cost_basis=Decimal("9300.00"),
            proceeds=Decimal(9000),
            term=Term.SHORT,
        ),
        LotSale(
            security=VTI,
            acquired_on=date(2023, 6, 1),
            quantity=Decimal(4),
            cost_basis=Decimal("800.00"),
            proceeds=Decimal(1200),
            term=Term.LONG,
        ),
    )
    assert (impact.short_term_gain, impact.long_term_gain) == (Decimal(-2300), Decimal(4600))
    # The $2,300 short-term loss absorbs half the long-term gain; the rest is taxed at 15%.
    assert impact.estimated_tax == Decimal("345.00")
    assert impact.net_capital_loss == 0
    assert impact.wash_sales == ()


def test_flags_a_loss_while_recently_bought_shares_are_still_held() -> None:
    account = vti_account(
        ("50", "17000.00", date(2025, 1, 10)), ("50", "15500.00", date(2026, 9, 10))
    )

    impact = estimate_tax(profile(), account, [order(Side.SELL, VTI, "50", "300")])

    assert impact.wash_sales == (
        WashSaleWarning(
            symbol="VTI",
            loss_at_risk=Decimal(2000),
            reason="VTI bought on 2026-09-10 is still held",
        ),
    )


def test_flags_a_loss_on_a_security_the_same_orders_buy() -> None:
    account = vti_account(("50", "17000.00", date(2025, 1, 10)))
    orders = [order(Side.SELL, VTI, "50", "300"), order(Side.BUY, VTI, "10", "300")]

    impact = estimate_tax(profile(), account, orders)

    assert impact.wash_sales == (
        WashSaleWarning(
            symbol="VTI", loss_at_risk=Decimal(2000), reason="the same orders also buy VTI"
        ),
    )


def test_selling_the_recent_purchase_itself_is_no_wash_sale() -> None:
    account = vti_account(("50", "15500.00", date(2026, 9, 10)))

    impact = estimate_tax(profile(), account, [order(Side.SELL, VTI, "50", "300")])

    assert impact.wash_sales == ()
    assert impact.net_capital_loss == Decimal(500)
    assert impact.estimated_tax == 0


@pytest.mark.parametrize(
    ("acquired_on", "sold_on", "term"),
    [
        pytest.param(date(2025, 9, 25), date(2026, 9, 25), Term.SHORT, id="exactly-one-year"),
        pytest.param(date(2025, 9, 24), date(2026, 9, 25), Term.LONG, id="a-year-and-a-day"),
        pytest.param(date(2024, 2, 29), date(2025, 2, 28), Term.SHORT, id="leap-day-anniversary"),
        pytest.param(date(2024, 2, 29), date(2025, 3, 1), Term.LONG, id="after-leap-anniversary"),
    ],
)
def test_a_lot_is_long_term_only_after_its_first_anniversary(
    acquired_on: date, sold_on: date, term: Term
) -> None:
    account = vti_account(("1", "100.00", acquired_on), as_of=sold_on)

    impact = estimate_tax(profile(), account, [order(Side.SELL, VTI, "1", "300")])

    assert impact.lot_sales[0].term is term


def test_splits_a_partly_sold_lots_basis_pro_rata_to_the_nearest_cent() -> None:
    account = vti_account(("3", "100.00", date(2024, 1, 2)))

    impact = estimate_tax(profile(), account, [order(Side.SELL, VTI, "2", "300")])

    assert impact.lot_sales[0].cost_basis == Decimal("66.67")


def test_sells_the_older_lot_first_when_costs_per_share_tie() -> None:
    account = vti_account(("10", "1000.00", date(2026, 6, 1)), ("10", "1000.00", date(2023, 6, 1)))

    impact = estimate_tax(profile(), account, [order(Side.SELL, VTI, "10", "150")])

    assert [(sale.acquired_on, sale.term) for sale in impact.lot_sales] == [
        (date(2023, 6, 1), Term.LONG)
    ]


@pytest.mark.parametrize(
    ("short_term", "long_term", "netted"),
    [
        pytest.param("1000", "500", ("1000", "500", "0"), id="both-gains"),
        pytest.param("1000", "-400", ("600", "0", "0"), id="long-loss-absorbed"),
        pytest.param("-300", "1000", ("0", "700", "0"), id="short-loss-absorbed"),
        pytest.param("-1000", "400", ("0", "0", "600"), id="short-loss-too-big"),
        pytest.param("400", "-1000", ("0", "0", "600"), id="long-loss-too-big"),
        pytest.param("-200", "-300", ("0", "0", "500"), id="both-losses"),
    ],
)
def test_nets_gains_and_losses_like_schedule_d(
    short_term: str, long_term: str, netted: tuple[str, str, str]
) -> None:
    result = net_capital_gains(Decimal(short_term), Decimal(long_term))

    assert result == NettedGains(*(Decimal(amount) for amount in netted))


def test_refuses_to_sell_more_than_the_lots_hold() -> None:
    account = vti_account(("10", "1000.00", date(2024, 1, 2)))

    with pytest.raises(TaxError, match="sells 11 VTI, but its lots hold only 10"):
        estimate_tax(profile(), account, [order(Side.SELL, VTI, "11", "300")])


def test_refuses_a_profile_for_another_client(portfolio: Portfolio) -> None:
    with pytest.raises(TaxError, match="profile is for C-2002"):
        estimate_tax(profile("C-2002"), portfolio, [])


def client_for(case: RebalanceCase) -> ClientProfile:
    return ClientProfile(
        client_id=case.portfolio.client_id,
        risk_tolerance=5,
        time_horizon_years=10,
        cash_reserve=case.cash_reserve,
        marginal_tax_rate=Decimal("0.32"),
    )


@given(rebalance_cases())
def test_relieves_exactly_the_quantity_each_sale_sells(case: RebalanceCase) -> None:
    orders = rebalance(*case).orders

    impact = estimate_tax(client_for(case), case.portfolio, orders)

    for placed in orders:
        if placed.side is Side.SELL:
            sold = [s.quantity for s in impact.lot_sales if s.security == placed.security]
            assert sum(sold, start=Decimal(0)) == placed.quantity


@given(rebalance_cases())
def test_a_full_exit_realises_exactly_the_recorded_cost_basis(case: RebalanceCase) -> None:
    impact = estimate_tax(client_for(case), case.portfolio, rebalance(*case).orders)

    for held in case.portfolio.holdings:
        sold = [s for s in impact.lot_sales if s.security == held.security]
        if sum((s.quantity for s in sold), start=Decimal(0)) == held.quantity:
            realised = sum((s.cost_basis for s in sold), start=Decimal(0))
            assert realised == sum((lot.cost_basis for lot in held.lots), start=Decimal(0))


@given(rebalance_cases())
def test_gains_and_tax_follow_from_the_lot_sales(case: RebalanceCase) -> None:
    impact = estimate_tax(client_for(case), case.portfolio, rebalance(*case).orders)
    total = sum((sale.gain for sale in impact.lot_sales), start=Decimal(0))

    assert impact.short_term_gain + impact.long_term_gain == total
    assert impact.net_capital_loss == max(-total, Decimal(0))
    assert impact.estimated_tax >= 0
