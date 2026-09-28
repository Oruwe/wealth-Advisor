from typing import NamedTuple

from wealth_advisor.agents.briefing import WriterAgent, briefing_facts, write_briefing
from wealth_advisor.agents.ips_reader import ReaderAgent, read_ips
from wealth_advisor.domain.client import ClientProfile
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.orders import RebalanceProposal
from wealth_advisor.domain.portfolio import Portfolio
from wealth_advisor.domain.primitives import DomainModel
from wealth_advisor.domain.suitability import SuitabilityReport
from wealth_advisor.domain.tax import TaxImpact
from wealth_advisor.rebalancer import propose_rebalance
from wealth_advisor.suitability_gate import check_suitability
from wealth_advisor.tax import estimate_tax


class Agents(NamedTuple):
    reader: ReaderAgent
    writer: WriterAgent


class AdviceDossier(DomainModel):
    """Everything an adviser reviews before approving a rebalance."""

    profile: ClientProfile
    ips_quotes: dict[str, str]  # the IPS passage behind each profile fact
    proposal: RebalanceProposal
    suitability: SuitabilityReport
    tax: TaxImpact
    briefing: str


def advise(
    ips_text: str, portfolio: Portfolio, prices: PriceSnapshot, agents: Agents
) -> AdviceDossier:
    """Agents read the IPS and write the briefing; every number in between comes from code."""
    profile, ips_quotes = read_ips(ips_text, portfolio.client_id, agents.reader)
    proposal = propose_rebalance(profile, portfolio, prices)
    suitability = check_suitability(profile, portfolio, prices, proposal)
    tax = estimate_tax(profile, portfolio, proposal.orders)
    briefing = write_briefing(briefing_facts(profile, proposal, suitability, tax), agents.writer)
    return AdviceDossier(
        profile=profile,
        ips_quotes=ips_quotes,
        proposal=proposal,
        suitability=suitability,
        tax=tax,
        briefing=briefing,
    )
