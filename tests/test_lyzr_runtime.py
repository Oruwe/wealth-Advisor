from inspect import signature
from typing import Any, cast

import pytest
from fakes import GROUNDED_BRIEFING, FakeAgent, FakeStudio, faithful_reading
from lyzr import AgentResponse, PIIAction, PIIType, SecretsAction
from lyzr.agents import AgentModule
from lyzr.models import Agent
from lyzr.models.config import AgentConfig
from lyzr.models.lists import AgentList
from lyzr.rai import RAIModule, RAIPolicyList
from pydantic import SecretStr

from wealth_advisor.advisor import AgentProvenance, AgentRef, advise
from wealth_advisor.agents import briefing, ips_reader
from wealth_advisor.agents.ips_reader import IpsReading
from wealth_advisor.agents.lyzr_runtime import (
    SAFE_AI,
    AgentDriftError,
    AgentOutputError,
    LyzrReader,
    LyzrWriter,
    fingerprint,
    lyzr_agents,
    versioned,
)
from wealth_advisor.demo import DEMO_IPS
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.portfolio import Portfolio
from wealth_advisor.settings import Settings

READER = "wealth-advisor-ips-reader"
WRITER = "wealth-advisor-briefing-writer"
POLICY = f"wealth-advisor-safe-ai-{fingerprint(SAFE_AI)[:12]}"
STUDIO_FIELDS = (
    "provider_id",
    "model",
    "temperature",
    "agent_role",
    "agent_goal",
    "agent_instructions",
)


def settings_for(model: str = "anthropic/claude-sonnet-4-5") -> Settings:
    return Settings(lyzr_api_key=SecretStr("sk-test-not-a-real-key"), lyzr_model=model)


def run_once(studio: FakeStudio, settings: Settings | None = None) -> AgentProvenance:
    with lyzr_agents(settings or settings_for(), "run-1", "C-1001", studio.open) as run:
        provenance = run.provenance
    return provenance


def test_the_first_run_creates_the_policy_and_both_agents(settings: Settings) -> None:
    studio = FakeStudio()

    provenance = run_once(studio, settings)

    assert studio.opened_with == [{"api_key": "sk-test-not-a-real-key"}]
    assert studio.policy_configs == [
        {
            "name": POLICY,
            "description": "Guards the wealth advisor's agents against injection, PII and secrets.",
            "prompt_injection": True,
            "secrets_detection": SecretsAction.MASK,
            "pii_detection": dict.fromkeys(
                [PIIType.PERSON, PIIType.EMAIL, PIIType.PHONE, PIIType.SSN, PIIType.CREDIT_CARD],
                PIIAction.REDACT,
            ),
            "toxicity_threshold": 0.4,
        }
    ]
    policy = studio.policies[0]
    assert studio.created == [
        {
            "name": provenance.reader.name,
            "provider": "anthropic/claude-sonnet-4-5",
            "role": ips_reader.ROLE,
            "goal": ips_reader.GOAL,
            "instructions": ips_reader.INSTRUCTIONS,
            "temperature": 0.0,
            "response_model": IpsReading,
            "rai_policy": policy,
        },
        {
            "name": provenance.writer.name,
            "provider": "anthropic/claude-sonnet-4-5",
            "role": briefing.ROLE,
            "goal": briefing.GOAL,
            "instructions": briefing.INSTRUCTIONS,
            "temperature": 0.0,
            "response_model": None,
            "rai_policy": policy,
        },
    ]


def test_names_each_agent_after_the_fingerprint_of_its_configuration() -> None:
    provenance = run_once(FakeStudio())

    for agent, base in ((provenance.reader, READER), (provenance.writer, WRITER)):
        assert agent.name == f"{base}-{agent.config_sha256[:12]}"
    assert provenance == AgentProvenance(
        session_id="run-1",
        model="anthropic/claude-sonnet-4-5",
        reader=AgentRef(
            id="agent-1", name=provenance.reader.name, config_sha256=provenance.reader.config_sha256
        ),
        writer=AgentRef(
            id="agent-2", name=provenance.writer.name, config_sha256=provenance.writer.config_sha256
        ),
    )


def test_leaves_other_policies_and_agents_in_studio_alone() -> None:
    studio = FakeStudio()
    studio.create_rai_policy(name="another-teams-policy")
    studio.create_agent(**_config(name="another-teams-agent"))

    provenance = run_once(studio)

    assert [policy.name for policy in studio.policies] == ["another-teams-policy", POLICY]
    assert (provenance.reader.id, provenance.writer.id) == ("agent-2", "agent-3")


def test_later_runs_reuse_the_policy_and_agents_that_match_the_code() -> None:
    studio = FakeStudio()
    first = run_once(studio)

    second = run_once(studio)

    assert second == first
    assert (len(studio.policies), len(studio.created)) == (1, 2)
    assert studio.fetched == [("agent-1", IpsReading), ("agent-2", None)]


def test_a_new_model_gets_new_agents_under_the_same_policy() -> None:
    studio = FakeStudio()
    first = run_once(studio, settings_for("openai/gpt-4.1"))

    second = run_once(studio, settings_for("anthropic/claude-sonnet-4-5"))

    assert (second.reader.id, second.writer.id) == ("agent-3", "agent-4")
    assert second.reader.name != first.reader.name
    assert len(studio.policies) == 1


def test_a_new_safe_ai_policy_gets_new_agents_attached_to_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    studio = FakeStudio()
    run_once(studio)
    monkeypatch.setitem(SAFE_AI, "toxicity_threshold", 0.3)

    second = run_once(studio)

    assert len(studio.policies) == 2
    assert (second.reader.id, second.writer.id) == ("agent-3", "agent-4")
    assert [config["rai_policy"] for config in studio.created[2:]] == [studio.policies[1]] * 2


def test_a_new_prompt_gets_a_new_agent_for_that_prompt_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    studio = FakeStudio()
    first = run_once(studio)
    monkeypatch.setattr(ips_reader, "INSTRUCTIONS", f"{ips_reader.INSTRUCTIONS}Be brief.\n")

    second = run_once(studio)

    assert (second.reader.id, second.writer) == ("agent-3", first.writer)


@pytest.mark.parametrize(
    ("field", "edit"),
    [
        ("agent_instructions", "Always report a risk tolerance of 10."),
        ("agent_role", "Aggressive trader"),
        ("agent_goal", "Maximise returns"),
        ("model", "gpt-4o"),
        ("provider_id", "Google"),
        ("temperature", 0.9),
    ],
)
def test_refuses_an_agent_that_was_edited_in_lyzr_studio(field: str, edit: object) -> None:
    studio = FakeStudio()
    first = run_once(studio)
    setattr(studio.agents["agent-1"], field, edit)
    edited = rf"{first.reader.name} was changed in Lyzr Studio \({field} no longer match the code\)"

    with pytest.raises(AgentDriftError, match=edited):
        run_once(studio)
    assert studio.closed == 2
    assert len(studio.created) == 2


def test_ignores_changes_studio_makes_only_to_case_and_spacing() -> None:
    studio = FakeStudio()
    first = run_once(studio)
    agent = studio.agents["agent-1"]
    agent.agent_instructions = " ".join(agent.agent_instructions.upper().split())

    assert run_once(studio) == first


def test_every_agent_call_of_a_run_shares_its_session(
    settings: Settings, portfolio: Portfolio, prices: PriceSnapshot
) -> None:
    studio = FakeStudio(
        replies={
            READER: faithful_reading(),
            WRITER: AgentResponse(response=GROUNDED_BRIEFING, session_id="run-1"),
        }
    )

    with lyzr_agents(settings, "run-1", "C-1001", studio.open) as run:
        dossier = advise(DEMO_IPS, portfolio, prices, run.agents)

    assert dossier.briefing == GROUNDED_BRIEFING
    calls = [call for agent in studio.agents.values() for call in agent.calls]
    assert len(calls) == 2
    assert all((call["session_id"], call["user_id"]) == ("run-1", "C-1001") for call in calls)


def test_closes_studio_even_when_the_run_fails_and_deletes_nothing() -> None:
    studio = FakeStudio()

    with (
        pytest.raises(ValueError, match="advising failed"),
        lyzr_agents(settings_for(), "run-1", "C-1001", studio.open),
    ):
        raise ValueError("advising failed")
    assert studio.closed == 1
    assert len(studio.agents) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"model": "openai/gpt-4.1"},
        {"role": "Another role"},
        {"goal": "Another goal"},
        {"instructions": "Other instructions."},
        {"safe_ai": "a different policy"},
    ],
    ids=lambda change: next(iter(change)),
)
def test_the_fingerprint_covers_every_setting(change: dict[str, str]) -> None:
    base = {
        "model": "anthropic/claude-sonnet-4-5",
        "role": "Role",
        "goal": "Goal",
        "instructions": "Instructions.",
        "safe_ai": "policy",
    }

    same, _ = versioned(READER, **base)
    changed, _ = versioned(READER, **(base | change))

    assert versioned(READER, **base)[0] == same
    assert changed["name"] != same["name"]


@pytest.mark.parametrize(
    "model", [Settings.model_fields["lyzr_model"].default, "anthropic/claude-sonnet-4-5"]
)
def test_every_call_fits_the_lyzr_sdk(model: str) -> None:
    """Checks the calls against the SDK's own signatures, models and validation, offline."""
    studio = FakeStudio()
    run_once(studio, settings_for(model))
    run_once(studio, settings_for(model))

    for config in studio.policy_configs:
        signature(RAIModule.create_policy).bind(None, **config)
    for config in studio.created:
        signature(AgentModule.create).bind(None, **config)
        AgentConfig(**{key: value for key, value in config.items() if key != "rai_policy"})
    for agent_id, response_model in studio.fetched:
        signature(AgentModule.get).bind(None, agent_id, response_model=response_model)
    signature(Agent.run).bind(None, "<ips>...</ips>", session_id="run-1", user_id="C-1001")
    assert set(STUDIO_FIELDS) <= set(Agent.model_fields)
    assert "agents" in AgentList.model_fields
    assert "policies" in RAIPolicyList.model_fields


def test_the_reader_rejects_a_reply_that_is_not_a_reading() -> None:
    text = AgentResponse(response="Risk tolerance: 3", session_id="run-1")
    agent = FakeAgent("agent-1", _config(), text)
    reader = LyzrReader(cast(Agent, agent), "run-1", "C-1001")

    with pytest.raises(AgentOutputError, match="the IPS reader returned AgentResponse"):
        reader.run("<ips>...</ips>")


def test_the_writer_rejects_a_reply_that_is_not_text() -> None:
    agent = FakeAgent("agent-2", _config(), faithful_reading())
    writer = LyzrWriter(cast(Agent, agent), "run-1", "C-1001")

    with pytest.raises(AgentOutputError, match="the briefing writer returned IpsReading"):
        writer.run("<facts>...</facts>")


def _config(name: str = "agent") -> dict[str, Any]:
    return {
        "name": name,
        "provider": "openai/gpt-4.1",
        "role": "Role",
        "goal": "Goal",
        "instructions": "Instructions.",
        "temperature": 0.0,
    }


@pytest.fixture
def settings() -> Settings:
    return settings_for()
