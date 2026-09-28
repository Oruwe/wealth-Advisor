"""Stand-ins for the Lyzr agents, so no test ever calls a model."""

from types import SimpleNamespace
from typing import Any, cast

from lyzr import Studio
from lyzr.providers import ModelResolver

from wealth_advisor.agents.ips_reader import IpsReading

# A briefing for the demo client that copies every number from the facts.
GROUNDED_BRIEFING = """\
Verdict: the suitability gate approved the proposal for client C-1001.
Trades: sell 100 TSLA, which is not in the model portfolio, for $25,000.00; sell 34 VTI; and
buy 429 BND.
Suitability: equity lands at 19.8% against a 20% cap, and cash of $10,170 stays above the
$2,000 reserve.
Tax: the $2,300 short-term loss offsets half of the $4,600 long-term gain, leaving $2,300 of
net long-term gain taxed at 15%: an estimated $345.
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


class FakeAgent:
    """A Lyzr agent as Studio stores it: the SDK resolves the model and strips the prompts."""

    def __init__(self, agent_id: str, config: dict[str, Any], reply: object) -> None:
        provider, model, _ = ModelResolver.parse(config["provider"])
        self.id, self.name, self.reply = agent_id, config["name"], reply
        self.provider_id, self.model = provider.value, model
        self.temperature = config["temperature"]
        self.agent_role = config["role"].strip()
        self.agent_goal = config["goal"].strip()
        self.agent_instructions = config["instructions"].strip()
        self.calls: list[dict[str, Any]] = []

    def run(self, message: str, **options: Any) -> object:
        self.calls.append({"message": message, **options})
        return self.reply


class FakeStudio:
    """Lyzr Studio in memory. Policies and agents outlive each run, as they do in Lyzr."""

    def __init__(self, replies: dict[str, object] | None = None) -> None:
        self.replies = replies or {}
        self.opened_with: list[dict[str, Any]] = []
        self.policies: list[SimpleNamespace] = []
        self.policy_configs: list[dict[str, Any]] = []
        self.agents: dict[str, FakeAgent] = {}
        self.created: list[dict[str, Any]] = []
        self.fetched: list[tuple[str, object]] = []
        self.closed = 0

    def open(self, **settings: Any) -> Studio:
        self.opened_with.append(settings)
        return cast(Studio, self)

    def list_rai_policies(self) -> SimpleNamespace:
        return SimpleNamespace(policies=list(self.policies))

    def create_rai_policy(self, **config: Any) -> SimpleNamespace:
        self.policy_configs.append(config)
        self.policies.append(
            SimpleNamespace(id=f"policy-{len(self.policies) + 1}", name=config["name"])
        )
        return self.policies[-1]

    def list_agents(self) -> SimpleNamespace:
        return SimpleNamespace(agents=list(self.agents.values()))

    def create_agent(self, **config: Any) -> FakeAgent:
        self.created.append(config)
        base = config["name"].rsplit("-", 1)[0]
        agent = FakeAgent(f"agent-{len(self.created)}", config, self.replies.get(base))
        self.agents[agent.id] = agent
        return agent

    def get_agent(self, agent_id: str, response_model: object = None) -> FakeAgent:
        self.fetched.append((agent_id, response_model))
        return self.agents[agent_id]

    def close(self) -> None:
        self.closed += 1
