import json
from decimal import Decimal
from fractions import Fraction
from typing import Protocol

from wealth_advisor.agents.grounding import numbers_in, unsupported_numbers
from wealth_advisor.domain.client import ClientProfile
from wealth_advisor.domain.orders import RebalanceProposal
from wealth_advisor.domain.portfolio import AssetClass
from wealth_advisor.domain.suitability import SuitabilityReport
from wealth_advisor.domain.tax import TaxImpact
from wealth_advisor.formatting import dollars, percent
from wealth_advisor.policy.model_portfolio import MODEL_SECURITIES
from wealth_advisor.policy.suitability import SUITABILITY_POLICY, SuitabilityPolicy

ROLE = "Adviser briefing writer"
GOAL = "Explain a rebalancing proposal to a registered investment adviser from the facts given."
INSTRUCTIONS = """\
You write a short briefing for a registered investment adviser about a rebalancing proposal.
The facts are JSON between <facts> and </facts>. Use only those facts.

Every number you write must be copied from the facts exactly as it appears there. Never
calculate, round, estimate or introduce a number, and do not use numbered lists.

Cover, in this order:
- Verdict: whether the suitability gate approved the proposal, and any rule it broke.
- Trades: what is sold and bought, and why.
- Suitability: how the result fits the client's IPS, comparing weights with their caps and
  cash with the reserve.
- Tax: the estimated tax, the gains and losses behind it, and any wash-sale warning.
Write plain, factual sentences for a professional reader, and never promise returns.
"""


class WriterAgent(Protocol):
    def run(self, message: str) -> str: ...


class BriefingError(ValueError):
    """The writer kept using numbers that are not in the facts."""


def briefing_facts(
    profile: ClientProfile,
    proposal: RebalanceProposal,
    suitability: SuitabilityReport,
    tax: TaxImpact,
    policy: SuitabilityPolicy = SUITABILITY_POLICY,
) -> dict[str, object]:
    """Everything the writer may say, with every number already formatted."""
    total = sum(proposal.value_after.values(), start=Decimal(0))
    band = policy.band_for(profile.risk_tolerance)
    model = set(MODEL_SECURITIES.values())
    return {
        "client_id": profile.client_id,
        "as_of": proposal.as_of.isoformat(),
        "risk_tolerance": profile.risk_tolerance,
        "risk_band": band.name,
        "time_horizon_years": profile.time_horizon_years,
        "portfolio_value": dollars(total),
        "suitability": {
            "approved": suitability.approved,
            "violations": [f"{v.rule}: {v.detail}" for v in suitability.violations],
        },
        "orders": [
            {
                "side": order.side.value,
                "symbol": order.security.symbol,
                "shares": format(order.quantity.normalize(), "f"),
                "price": dollars(order.price),
                "amount": dollars(order.quantity * order.price),
                "reason": "toward the target allocation"
                if order.security in model
                else "not in the model portfolio",
            }
            for order in proposal.orders
        ],
        "allocation_after": [
            {
                "asset_class": asset_class.value,
                "target": percent(Fraction(proposal.target.weights[asset_class])),
                "after": percent(_share(proposal.value_after[asset_class], total)),
                "cap": percent(Fraction(band.max_weight[asset_class]))
                if asset_class in band.max_weight
                else "none",
            }
            for asset_class in AssetClass
        ],
        "cash_after": dollars(proposal.cash_after),
        "cash_reserve": dollars(profile.cash_reserve),
        "tax": {
            "estimated_tax": dollars(tax.estimated_tax),
            "short_term_gain": dollars(tax.short_term_gain),
            "long_term_gain": dollars(tax.long_term_gain),
            "net_capital_loss": dollars(tax.net_capital_loss),
            "short_term_rate": percent(Fraction(tax.short_term_rate)),
            "long_term_rate": percent(Fraction(tax.long_term_rate)),
            "wash_sales": [
                f"{w.symbol}: {dollars(w.loss_at_risk)} of losses at risk; {w.reason}"
                for w in tax.wash_sales
            ],
        },
    }


def write_briefing(facts: dict[str, object], writer: WriterAgent, attempts: int = 2) -> str:
    """Have the writer brief the adviser, rejecting any draft with a number that is not in the
    facts. A rejected draft gets one retry that names the offending numbers."""
    facts_json = json.dumps(facts, indent=2)
    allowed = numbers_in(facts_json)
    message = f"<facts>\n{facts_json}\n</facts>"
    stray: list[str] = []
    for _ in range(attempts):
        draft = writer.run(message)
        stray = unsupported_numbers(draft, allowed)
        if not stray:
            return draft
        message = (
            f"<facts>\n{facts_json}\n</facts>\n\nYour last draft used numbers that are not in "
            f"the facts: {', '.join(stray)}. Rewrite it using only numbers copied from the facts."
        )
    raise BriefingError(f"the briefing uses numbers that are not in the facts: {', '.join(stray)}")


def _share(value: Decimal, total: Decimal) -> Fraction:
    return Fraction(value) / Fraction(total) if total else Fraction(0)
