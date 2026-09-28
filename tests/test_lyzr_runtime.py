from inspect import signature
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fakes import GROUNDED_BRIEFING, faithful_reading
from lyzr import AgentResponse, PIIAction, PIIType, SecretsAction, Studio
from lyzr.agents import AgentModule
from lyzr.models import Agent
from lyzr.models.config import AgentConfig
from lyzr.rai import RAIModule
from pydantic import SecretStr

from wealth_advisor.advisor import advise
from wealth_advisor.agents import briefing, ips_reader
from wealth_advisor.agents.ips_reader import IpsReading
from wealth_advisor.agents.lyzr_runtime import (
    AgentOutputError,
    LyzrReader,
    LyzrWriter,
    lyzr_agents,
)
from wealth_advisor.demo import DEMO_IPS
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.portfolio import Portfolio
from wealth_advisor.settings import Settings

READER = "wealth-advisor-ips-reader"
WRITER = "wealth-advisor-briefing-writer"


class FakeAgent:
    def __init__(self, agent_id: str, reply: object = None) -> None:
        self.id = agent_id
        self.reply = reply
        self.messages: list[str] = []

    def run(self, message: str) -> object:
        self.messages.append(message)
        return self.reply


class FakeStudio:
    """Records what `lyzr_agents` creates in Lyzr Studio, and what it cleans up."""

    def __init__(self, replies: dict[str, object] | None = None, fail_creating: str = "") -> None:
        self.replies = replies or {}
        self.fail_creating = fail_creating
        self.opened_with: dict[str, Any] = {}
        self.policy_config: dict[str, Any] = {}
        self.policy = SimpleNamespace(id="policy-1")
        self.agent_configs: dict[str, dict[str, Any]] = {}
        self.cleanup: list[str] = []
        self.rai = self

    def open(self, **settings: Any) -> Studio:
        self.opened_with = settings
        return cast(Studio, self)

    def create_rai_policy(self, **config: Any) -> SimpleNamespace:
        self.policy_config = config
        return self.policy

    def create_agent(self, **config: Any) -> FakeAgent:
        name = config["name"]
        if name == self.fail_creating:
            raise RuntimeError("Lyzr Studio is unavailable")
        self.agent_configs[name] = config
        return FakeAgent(f"agent-{len(self.agent_configs)}", self.replies.get(name))

    def delete_agent(self, agent_id: str) -> bool:
        self.cleanup.append(f"delete {agent_id}")
        return True

    def delete_policy(self, policy_id: str) -> bool:
        self.cleanup.append(f"delete {policy_id}")
        return True

    def close(self) -> None:
        self.cleanup.append("close")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        lyzr_api_key=SecretStr("sk-test-not-a-real-key"), lyzr_model="anthropic/claude-sonnet-4-5"
    )


def test_opens_studio_with_the_api_key_from_settings(settings: Settings) -> None:
    studio = FakeStudio()

    with lyzr_agents(settings, studio_factory=studio.open):
        assert studio.opened_with == {"api_key": "sk-test-not-a-real-key"}


def test_guards_the_agents_with_a_safe_ai_policy(settings: Settings) -> None:
    studio = FakeStudio()

    with lyzr_agents(settings, studio_factory=studio.open):
        assert studio.policy_config == {
            "name": "wealth-advisor-safe-ai",
            "description": "Guards the wealth advisor's agents against injection, PII and secrets.",
            "prompt_injection": True,
            "secrets_detection": SecretsAction.MASK,
            "pii_detection": dict.fromkeys(
                [PIIType.PERSON, PIIType.EMAIL, PIIType.PHONE, PIIType.SSN, PIIType.CREDIT_CARD],
                PIIAction.REDACT,
            ),
            "toxicity_threshold": 0.4,
        }


def test_sets_up_both_agents_deterministic_on_the_configured_model(settings: Settings) -> None:
    studio = FakeStudio()

    with lyzr_agents(settings, studio_factory=studio.open):
        assert studio.agent_configs == {
            READER: {
                "name": READER,
                "provider": "anthropic/claude-sonnet-4-5",
                "role": ips_reader.ROLE,
                "goal": ips_reader.GOAL,
                "instructions": ips_reader.INSTRUCTIONS,
                "temperature": 0.0,
                "response_model": IpsReading,
                "rai_policy": studio.policy,
            },
            WRITER: {
                "name": WRITER,
                "provider": "anthropic/claude-sonnet-4-5",
                "role": briefing.ROLE,
                "goal": briefing.GOAL,
                "instructions": briefing.INSTRUCTIONS,
                "temperature": 0.0,
                "rai_policy": studio.policy,
            },
        }


@pytest.mark.parametrize(
    "model", [Settings.model_fields["lyzr_model"].default, "anthropic/claude-sonnet-4-5"]
)
def test_every_call_fits_the_lyzr_sdk(model: str) -> None:
    """Checks the calls against the SDK's own signatures and agent validation, offline."""
    settings = Settings(lyzr_api_key=SecretStr("sk-test-not-a-real-key"), lyzr_model=model)
    studio = FakeStudio()

    with lyzr_agents(settings, studio.open):
        signature(RAIModule.create_policy).bind(None, **studio.policy_config)
        for config in studio.agent_configs.values():
            signature(AgentModule.create).bind(None, **config)
            AgentConfig(**{key: value for key, value in config.items() if key != "rai_policy"})


def test_the_fake_cleans_up_through_the_same_calls_as_lyzr() -> None:
    assert signature(FakeStudio.delete_agent) == signature(AgentModule.delete)
    assert signature(FakeStudio.delete_policy) == signature(RAIModule.delete_policy)


def test_deletes_everything_it_created_even_when_advising_fails(settings: Settings) -> None:
    studio = FakeStudio()

    with pytest.raises(ValueError, match="advising failed"), lyzr_agents(settings, studio.open):
        raise ValueError("advising failed")
    assert studio.cleanup == ["delete agent-2", "delete agent-1", "delete policy-1", "close"]


def test_deletes_what_it_created_when_studio_fails_midway(settings: Settings) -> None:
    studio = FakeStudio(fail_creating=WRITER)
    agents = lyzr_agents(settings, studio.open)

    with pytest.raises(RuntimeError, match="Lyzr Studio is unavailable"):
        agents.__enter__()
    assert studio.cleanup == ["delete agent-1", "delete policy-1", "close"]


def test_advises_through_the_lyzr_agents(
    settings: Settings, portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    studio = FakeStudio(
        replies={
            READER: faithful_reading(),
            WRITER: AgentResponse(response=GROUNDED_BRIEFING, session_id="session-1"),
        }
    )

    with lyzr_agents(settings, studio.open) as agents:
        dossier = advise(DEMO_IPS, portfolio, prices, agents)

    assert dossier.suitability.approved
    assert dossier.briefing == GROUNDED_BRIEFING


def test_the_reader_rejects_a_reply_that_is_not_a_reading() -> None:
    text = AgentResponse(response="Risk tolerance: 3", session_id="session-1")
    reader = LyzrReader(cast(Agent, FakeAgent("agent-1", text)))

    with pytest.raises(AgentOutputError, match="the IPS reader returned AgentResponse"):
        reader.run("<ips>...</ips>")


def test_the_writer_rejects_a_reply_that_is_not_text() -> None:
    writer = LyzrWriter(cast(Agent, FakeAgent("agent-2", faithful_reading())))

    with pytest.raises(AgentOutputError, match="the briefing writer returned IpsReading"):
        writer.run("<facts>...</facts>")
