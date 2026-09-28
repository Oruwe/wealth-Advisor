"""Stand-ins for the Lyzr agents, so no test ever calls a model."""

from typing import Any

from wealth_advisor.agents.ips_reader import IpsReading

# A briefing for the demo client that copies every number from the facts.
GROUNDED_BRIEFING = """\
Verdict: the suitability gate approved the proposal for client C-1001.
Trades: sell 100 TSLA, which is not in the model portfolio, for $25,000.00; sell 34 VTI; and
buy 429 BND.
Suitability: equity lands at 19.8% against a 20% cap, and cash of $10,170 stays above the
$2,000 reserve.
Tax: the estimated tax is $345, from a $4,600 long-term gain and a $2,300 short-term loss.
"""


def faithful_reading(**changes: Any) -> IpsReading:
    """What a faithful reader returns for `DEMO_IPS`, with any fields changed."""
    return IpsReading(
        risk_tolerance=3,
        risk_tolerance_quote="Risk tolerance: 3 on a scale of 1 to 10.",
        time_horizon_years=10,
        time_horizon_quote="Time horizon: 10 years.",
        cash_reserve="2000",
        cash_reserve_quote="keep at least $2,000 in cash at all times",
        marginal_tax_rate="0.24",
        marginal_tax_rate_quote="the client is in the 24% federal income-tax bracket",
    ).model_copy(update=changes)


class FakeReader:
    """Returns its readings in order, then keeps repeating the last one; records every message."""

    def __init__(self, *readings: IpsReading) -> None:
        self.readings = list(readings)
        self.messages: list[str] = []

    def run(self, message: str) -> IpsReading:
        self.messages.append(message)
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]


class FakeWriter:
    """Returns its drafts in order, and records every message it was sent."""

    def __init__(self, *drafts: str) -> None:
        self.drafts = list(drafts)
        self.messages: list[str] = []

    def run(self, message: str) -> str:
        self.messages.append(message)
        return self.drafts.pop(0)
