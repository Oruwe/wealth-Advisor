from decimal import Decimal

import pytest
from examples import BND, TSLA, VTI, order, profile
from fakes import GROUNDED_BRIEFING, FakeReader, FakeWriter, faithful_reading

from wealth_advisor.advisor import AdviceDossier, Agents, advise
from wealth_advisor.agents.ips_reader import IpsReadingError
from wealth_advisor.demo import DEMO_IPS
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.orders import Side
from wealth_advisor.domain.portfolio import Portfolio


def faithful_agents() -> Agents:
    return Agents(reader=FakeReader(faithful_reading()), writer=FakeWriter(GROUNDED_BRIEFING))


def test_advises_the_demo_client_from_ips_to_briefing(
    portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    dossier = advise(DEMO_IPS, portfolio, prices, faithful_agents())

    assert dossier.profile == profile()
    assert dossier.ips_quotes["risk_tolerance"] == "Risk tolerance: 3 on a scale of 1 to 10."
    assert dossier.proposal.orders == (
        order(Side.SELL, TSLA, "100", "250"),
        order(Side.SELL, VTI, "34", "300"),
        order(Side.BUY, BND, "429", "70"),
    )
    assert dossier.suitability.approved
    assert dossier.tax.estimated_tax == Decimal("345.00")
    assert dossier.briefing == GROUNDED_BRIEFING


def test_stops_before_the_briefing_when_the_ips_reading_is_not_grounded(
    portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    writer = FakeWriter()
    agents = Agents(reader=FakeReader(faithful_reading(risk_tolerance=7)), writer=writer)

    with pytest.raises(IpsReadingError):
        advise(DEMO_IPS, portfolio, prices, agents)
    assert writer.messages == []


def test_the_dossier_survives_a_json_round_trip(
    portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    dossier = advise(DEMO_IPS, portfolio, prices, faithful_agents())

    assert AdviceDossier.model_validate_json(dossier.model_dump_json()) == dossier
