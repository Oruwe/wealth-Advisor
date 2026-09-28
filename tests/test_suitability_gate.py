from datetime import date
from decimal import Decimal

import pytest
from examples import AS_OF, BND, CONSERVATIVE, TSLA, VTI, order, profile
from hypothesis import given
from hypothesis import strategies as st
from strategies import RebalanceCase, rebalance_cases

from wealth_advisor.domain.client import ClientProfile
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.orders import Order, RebalanceProposal, Side
from wealth_advisor.domain.portfolio import AssetClass, Portfolio, Security
from wealth_advisor.domain.suitability import Rule, SuitabilityReport, Violation
from wealth_advisor.policy.suitability import parse_policy
from wealth_advisor.rebalancer import propose_rebalance, rebalance
from wealth_advisor.suitability_gate import check_suitability

DOGE = Security(symbol="DOGE-USD", asset_class=AssetClass.EQUITY)


@pytest.fixture
def proposal(portfolio: Portfolio, prices: PriceSnapshot) -> RebalanceProposal:
    """Sells TSLA 100 and VTI 34, buys BND 429: 19.8% equity and $10,170 cash afterwards."""
    return propose_rebalance(profile(), portfolio, prices)


def with_orders(proposal: RebalanceProposal, *orders: Order) -> RebalanceProposal:
    return proposal.model_copy(update={"orders": orders})


def rules(report: SuitabilityReport) -> set[Rule]:
    return {violation.rule for violation in report.violations}


def test_approves_the_rebalancers_proposal(
    portfolio: Portfolio, prices: PriceSnapshot, proposal: RebalanceProposal
) -> None:
    report = check_suitability(profile(), portfolio, prices, proposal)

    assert report.approved
    assert report.violations == ()


def test_catches_equity_one_share_over_the_conservative_cap(
    portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    uncapped = rebalance(portfolio, prices, CONSERVATIVE, cash_reserve=Decimal("2000.00"))

    report = check_suitability(profile(), portfolio, prices, uncapped)

    assert not report.approved
    assert report.violations == (
        Violation(
            rule=Rule.MAX_WEIGHT,
            detail="equity would be 20.10% of the portfolio; "
            "conservative clients may hold at most 20.00%",
        ),
    )


def test_enforces_whatever_caps_the_policy_file_sets(
    portfolio: Portfolio, prices: PriceSnapshot, proposal: RebalanceProposal
) -> None:
    stricter = parse_policy(
        "approved_securities: [VTI, BND, GLD]\nrisk_bands:\n"
        "  - {name: cautious, risk_scores: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],"
        " max_weight: {equity: 0.15}}"
    )

    report = check_suitability(profile(), portfolio, prices, proposal, policy=stricter)

    assert report.violations == (
        Violation(
            rule=Rule.MAX_WEIGHT,
            detail="equity would be 19.80% of the portfolio; "
            "cautious clients may hold at most 15.00%",
        ),
    )


def test_blocks_buying_a_security_outside_the_approved_list(
    portfolio: Portfolio, prices: PriceSnapshot, proposal: RebalanceProposal
) -> None:
    with_doge = PriceSnapshot(as_of=AS_OF, prices={**prices.prices, "DOGE-USD": Decimal("0.25")})
    buys_doge = with_orders(proposal, *proposal.orders, order(Side.BUY, DOGE, "1000", "0.25"))

    report = check_suitability(profile(), portfolio, with_doge, buys_doge)

    assert (
        Violation(
            rule=Rule.APPROVED_SECURITIES,
            detail="buys DOGE-USD, which is not on the approved list",
        )
        in report.violations
    )


def test_reports_every_rule_a_proposal_breaks(
    portfolio: Portfolio, prices: PriceSnapshot, proposal: RebalanceProposal
) -> None:
    keeps_tsla = with_orders(proposal, *(o for o in proposal.orders if o.security != TSLA))

    report = check_suitability(profile(), portfolio, prices, keeps_tsla)

    assert rules(report) == {
        Rule.MATCHING_FIGURES,
        Rule.APPROVED_SECURITIES,
        Rule.MAX_WEIGHT,
        Rule.CASH_RESERVE,
    }
    assert (
        Violation(
            rule=Rule.APPROVED_SECURITIES,
            detail="keeps TSLA, which is not on the approved list",
        )
        in report.violations
    )


def test_blocks_a_proposal_that_spends_the_cash_reserve(
    portfolio: Portfolio, prices: PriceSnapshot, proposal: RebalanceProposal
) -> None:
    report = check_suitability(profile(cash_reserve="15000.00"), portfolio, prices, proposal)

    assert report.violations == (
        Violation(
            rule=Rule.CASH_RESERVE,
            detail="cash would be $10,170.00 after the trades; "
            "the IPS requires at least $15,000.00",
        ),
    )


def test_blocks_selling_more_than_is_held(
    portfolio: Portfolio, prices: PriceSnapshot, proposal: RebalanceProposal
) -> None:
    oversells = with_orders(
        proposal,
        order(Side.SELL, TSLA, "100", "250"),
        order(Side.SELL, VTI, "101", "300"),
        order(Side.BUY, BND, "429", "70"),
    )

    report = check_suitability(profile(), portfolio, prices, oversells)

    assert (
        Violation(rule=Rule.NO_OVERSELLING, detail="sells 101 VTI but only 100 are held")
        in report.violations
    )


def test_blocks_orders_priced_away_from_the_snapshot(
    portfolio: Portfolio, prices: PriceSnapshot, proposal: RebalanceProposal
) -> None:
    repriced = with_orders(
        proposal,
        order(Side.SELL, TSLA, "100", "250"),
        order(Side.SELL, VTI, "34", "300"),
        order(Side.BUY, BND, "429", "65"),
    )

    report = check_suitability(profile(), portfolio, prices, repriced)

    assert (
        Violation(rule=Rule.SNAPSHOT_PRICES, detail="buy BND at 65, but the snapshot price is 70")
        in report.violations
    )


def test_blocks_an_order_the_snapshot_cannot_price(
    portfolio: Portfolio, prices: PriceSnapshot, proposal: RebalanceProposal
) -> None:
    buys_doge = with_orders(proposal, *proposal.orders, order(Side.BUY, DOGE, "1000", "0.25"))

    report = check_suitability(profile(), portfolio, prices, buys_doge)

    assert (
        Violation(
            rule=Rule.SNAPSHOT_PRICES,
            detail="buy DOGE-USD at 0.25, but the snapshot has no price",
        )
        in report.violations
    )


def test_verifies_the_figures_a_proposal_reports_about_itself(
    portfolio: Portfolio, prices: PriceSnapshot, proposal: RebalanceProposal
) -> None:
    inflated = proposal.model_copy(update={"cash_after": proposal.cash_after + 1000})

    report = check_suitability(profile(), portfolio, prices, inflated)

    assert rules(report) == {Rule.MATCHING_FIGURES}


def test_blocks_mixing_clients_or_dates(
    portfolio: Portfolio, prices: PriceSnapshot, proposal: RebalanceProposal
) -> None:
    other_day = PriceSnapshot(as_of=date(2026, 9, 24), prices=prices.prices)

    report = check_suitability(profile("C-2002"), portfolio, other_day, proposal)

    assert rules(report) == {Rule.SAME_CLIENT, Rule.SAME_DATE}


@given(rebalance_cases(), st.integers(1, 10))
def test_every_proposal_the_rebalancer_makes_passes_the_gate(
    case: RebalanceCase, risk_score: int
) -> None:
    client = ClientProfile(
        client_id=case.portfolio.client_id,
        risk_tolerance=risk_score,
        time_horizon_years=10,
        cash_reserve=case.cash_reserve,
        marginal_tax_rate=Decimal("0.24"),
    )

    proposal = propose_rebalance(client, case.portfolio, case.prices)

    assert check_suitability(client, case.portfolio, case.prices, proposal).violations == ()
