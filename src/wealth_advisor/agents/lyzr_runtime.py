from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager

from lyzr import AgentResponse, PIIAction, PIIType, SecretsAction, Studio
from lyzr.models import Agent

from wealth_advisor.advisor import Agents
from wealth_advisor.agents import briefing, ips_reader
from wealth_advisor.agents.ips_reader import IpsReading
from wealth_advisor.settings import Settings

# An IPS is untrusted text, and client PII should never reach the model or the briefing.
REDACTED_PII = {
    PIIType.PERSON: PIIAction.REDACT,
    PIIType.EMAIL: PIIAction.REDACT,
    PIIType.PHONE: PIIAction.REDACT,
    PIIType.SSN: PIIAction.REDACT,
    PIIType.CREDIT_CARD: PIIAction.REDACT,
}


class AgentOutputError(RuntimeError):
    """A Lyzr agent returned something other than what it was set up to return."""


class LyzrReader:
    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def run(self, message: str) -> IpsReading:
        output = self._agent.run(message)
        if not isinstance(output, IpsReading):
            raise AgentOutputError(f"the IPS reader returned {type(output).__name__}")
        return output


class LyzrWriter:
    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def run(self, message: str) -> str:
        output = self._agent.run(message)
        if not isinstance(output, AgentResponse):
            raise AgentOutputError(f"the briefing writer returned {type(output).__name__}")
        return output.response


@contextmanager
def lyzr_agents(
    settings: Settings, studio_factory: Callable[..., Studio] = Studio
) -> Iterator[Agents]:
    """Create the Safe AI policy and both agents in Lyzr Studio, and delete them afterwards."""
    with ExitStack() as cleanup:
        studio = studio_factory(api_key=settings.lyzr_api_key.get_secret_value())
        cleanup.callback(studio.close)
        policy = studio.create_rai_policy(
            name="wealth-advisor-safe-ai",
            description="Guards the wealth advisor's agents against injection, PII and secrets.",
            prompt_injection=True,
            secrets_detection=SecretsAction.MASK,
            pii_detection=REDACTED_PII,
            toxicity_threshold=0.4,
        )
        cleanup.callback(studio.rai.delete_policy, policy.id)
        reader = studio.create_agent(
            name="wealth-advisor-ips-reader",
            provider=settings.lyzr_model,
            role=ips_reader.ROLE,
            goal=ips_reader.GOAL,
            instructions=ips_reader.INSTRUCTIONS,
            temperature=0.0,
            response_model=IpsReading,
            rai_policy=policy,
        )
        cleanup.callback(studio.delete_agent, reader.id)
        writer = studio.create_agent(
            name="wealth-advisor-briefing-writer",
            provider=settings.lyzr_model,
            role=briefing.ROLE,
            goal=briefing.GOAL,
            instructions=briefing.INSTRUCTIONS,
            temperature=0.0,
            rai_policy=policy,
        )
        cleanup.callback(studio.delete_agent, writer.id)
        yield Agents(reader=LyzrReader(reader), writer=LyzrWriter(writer))
