import json
from decimal import Decimal
from typing import NamedTuple

import pytest
from examples import AS_OF, profile
from fakes import GROUNDED_BRIEFING, FakeWriter

from wealth_advisor.agents.briefing import BriefingError, briefing_facts, write_briefing
from wealth_advisor.domain.client import ClientProfile
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.orders import RebalanceProposal
from wealth_advisor.domain.portfolio import Portfolio
from wealth_advisor.domain.suitability import Rule, SuitabilityReport, Violation
from wealth_advisor.domain.tax import TaxImpact, WashSaleWarning
from wealth_advisor.rebalancer import propose_rebalance
from wealth_advisor.suitability_gate import check_suitability
from wealth_advisor.tax import estimate_tax

DEMO_TAX: dict[str, object] = {
    "estimated_tax": "$345.00",
    "short_term_gain": "-$2,300.00",
    "long_term_gain": "$4,600.00",
    "net_capital_loss": "$0.00",
    "short_term_rate": "24.00%",
    "long_term_rate": "15.00%",
    "wash_sales": [],
}


class Analysis(NamedTuple):
    profile: ClientProfile
    proposal: RebalanceProposal
    suitability: SuitabilityReport
    tax: TaxImpact


def analyse(client: ClientProfile, portfolio: Portfolio, prices: PriceSnapshot) -> Analysis:
    proposal = propose_rebalance(client, portfolio, prices)
    return Analysis(
        client,
        proposal,
        check_suitability(client, portfolio, prices, proposal),
        estimate_tax(client, portfolio, proposal.orders),
    )


@pytest.fixture
def analysis(portfolio: Portfolio, prices: PriceSnapshot) -> Analysis:
    return analyse(profile(), portfolio, prices)


@pytest.fixture
def facts(analysis: Analysis) -> dict[str, object]:
    return briefing_facts(*analysis)


def facts_message(facts: dict[str, object]) -> str:
    return f"<facts>\n{json.dumps(facts, indent=2)}\n</facts>"


def test_facts_hand_the_writer_every_figure_already_formatted(facts: dict[str, object]) -> None:
    assert facts == {
        "client_id": "C-1001",
        "as_of": "2026-09-25",
        "risk_tolerance": 3,
        "risk_band": "conservative",
        "time_horizon_years": 10,
        "portfolio_value": "$100,000.00",
        "suitability": {"approved": True, "violations": []},
        "orders": [
            {
                "side": "sell",
                "symbol": "TSLA",
                "shares": "100",
                "price": "$250.00",
                "amount": "$25,000.00",
                "reason": "not in the model portfolio",
            },
            {
                "side": "sell",
                "symbol": "VTI",
                "shares": "34",
                "price": "$300.00",
                "amount": "$10,200.00",
                "reason": "toward the target allocation",
            },
            {
                "side": "buy",
                "symbol": "BND",
                "shares": "429",
                "price": "$70.00",
                "amount": "$30,030.00",
                "reason": "toward the target allocation",
            },
        ],
        "allocation_after": [
            {"asset_class": "equity", "target": "20.00%", "after": "19.80%", "cap": "20.00%"},
            {"asset_class": "fixed_income", "target": "65.00%", "after": "65.03%", "cap": "none"},
            {"asset_class": "commodity", "target": "5.00%", "after": "5.00%", "cap": "10.00%"},
            {"asset_class": "cash", "target": "10.00%", "after": "10.17%", "cap": "none"},
        ],
        "cash_after": "$10,170.00",
        "cash_reserve": "$2,000.00",
        "tax": DEMO_TAX,
    }


def test_facts_name_every_broken_rule_and_wash_sale(analysis: Analysis) -> None:
    broken = Violation(
        rule=Rule.CASH_RESERVE,
        detail="cash would be $500.00 after the trades; the IPS requires at least $2,000.00",
    )
    warning = WashSaleWarning(
        symbol="VTI",
        loss_at_risk=Decimal("300.00"),
        reason="VTI bought on 2026-09-01 is still held",
    )

    facts = briefing_facts(
        analysis.profile,
        analysis.proposal,
        analysis.suitability.model_copy(update={"violations": (broken,)}),
        analysis.tax.model_copy(update={"wash_sales": (warning,)}),
    )

    assert facts["suitability"] == {
        "approved": False,
        "violations": [
            "cash_reserve: cash would be $500.00 after the trades; "
            "the IPS requires at least $2,000.00"
        ],
    }
    assert facts["tax"] == DEMO_TAX | {
        "wash_sales": ["VTI: $300.00 of losses at risk; VTI bought on 2026-09-01 is still held"]
    }


def test_an_empty_account_has_no_weights_to_report(prices: PriceSnapshot) -> None:
    empty = Portfolio(client_id="C-1001", as_of=AS_OF, holdings=(), cash=Decimal("0.00"))

    facts = briefing_facts(*analyse(profile(cash_reserve="0.00"), empty, prices))

    assert facts["portfolio_value"] == "$0.00"
    assert facts["allocation_after"] == [
        {"asset_class": "equity", "target": "20.00%", "after": "0.00%", "cap": "20.00%"},
        {"asset_class": "fixed_income", "target": "65.00%", "after": "0.00%", "cap": "none"},
        {"asset_class": "commodity", "target": "5.00%", "after": "0.00%", "cap": "10.00%"},
        {"asset_class": "cash", "target": "10.00%", "after": "0.00%", "cap": "none"},
    ]


def test_accepts_a_briefing_that_copies_its_numbers_from_the_facts(
    facts: dict[str, object],
) -> None:
    writer = FakeWriter(GROUNDED_BRIEFING)

    assert write_briefing(facts, writer) == GROUNDED_BRIEFING
    assert writer.messages == [facts_message(facts)]


def test_asks_once_more_naming_the_numbers_the_writer_made_up(facts: dict[str, object]) -> None:
    writer = FakeWriter(
        "Equity ends 0.2 points under its cap, saving the client about $1.5k.", GROUNDED_BRIEFING
    )

    assert write_briefing(facts, writer) == GROUNDED_BRIEFING
    assert writer.messages[1] == (
        f"{facts_message(facts)}\n\nYour last draft used numbers that are not in the facts: "
        "0.2, 1.5k. Rewrite it using only numbers copied from the facts."
    )


def test_gives_up_when_the_rewrite_still_makes_numbers_up(facts: dict[str, object]) -> None:
    writer = FakeWriter("The tax is about $340.", "The tax is roughly $350.")

    with pytest.raises(BriefingError, match=r"numbers that are not in the facts: 350$"):
        write_briefing(facts, writer)
    assert len(writer.messages) == 2


def test_ignores_the_markers_of_a_numbered_list(facts: dict[str, object]) -> None:
    listed = "1. Verdict: approved.\n2) Tax: an estimated $345.\n"

    assert write_briefing(facts, FakeWriter(listed)) == listed
