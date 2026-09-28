"""The advisor's Lyzr agents, kept in Lyzr Studio between runs so AIMS keeps their history.

Every agent's name ends with a fingerprint of its whole configuration: model, instructions,
output format and Safe AI policy. A configuration therefore never changes under a name.
Changing any of it creates a new agent, and the old one keeps its history. A run reuses the
agents that match the code, refuses any that were edited in Lyzr Studio, and makes all of its
agent calls in one Lyzr session named after the run.
"""

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from typing import NamedTuple, TypedDict

from lyzr import AgentResponse, PIIAction, PIIType, SecretsAction, Studio
from lyzr.models import Agent
from lyzr.providers import ModelResolver
from lyzr.rai import RAIPolicy
from pydantic import BaseModel

from wealth_advisor.advisor import AgentProvenance, AgentRef, Agents
from wealth_advisor.agents import briefing, ips_reader
from wealth_advisor.agents.grounding import normalise
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


class SafeAiSettings(TypedDict):
    description: str
    prompt_injection: bool
    secrets_detection: SecretsAction
    pii_detection: dict[PIIType, PIIAction]
    toxicity_threshold: float


SAFE_AI: SafeAiSettings = {
    "description": "Guards the wealth advisor's agents against injection, PII and secrets.",
    "prompt_injection": True,
    "secrets_detection": SecretsAction.MASK,
    "pii_detection": REDACTED_PII,
    "toxicity_threshold": 0.4,
}


class AgentSettings(TypedDict):
    name: str
    provider: str
    role: str
    goal: str
    instructions: str
    temperature: float


class AgentOutputError(RuntimeError):
    """A Lyzr agent returned something other than what it was set up to return."""


class AgentDriftError(RuntimeError):
    """An agent in Lyzr Studio no longer matches the configuration its name stands for."""


class LyzrRun(NamedTuple):
    agents: Agents
    provenance: AgentProvenance


class LyzrReader:
    def __init__(self, agent: Agent, session_id: str, user_id: str) -> None:
        self._agent, self._session_id, self._user_id = agent, session_id, user_id

    def run(self, message: str) -> IpsReading:
        output = self._agent.run(message, session_id=self._session_id, user_id=self._user_id)
        if not isinstance(output, IpsReading):
            raise AgentOutputError(f"the IPS reader returned {type(output).__name__}")
        return output


class LyzrWriter:
    def __init__(self, agent: Agent, session_id: str, user_id: str) -> None:
        self._agent, self._session_id, self._user_id = agent, session_id, user_id

    def run(self, message: str) -> str:
        output = self._agent.run(message, session_id=self._session_id, user_id=self._user_id)
        if not isinstance(output, AgentResponse):
            raise AgentOutputError(f"the briefing writer returned {type(output).__name__}")
        return output.response


@contextmanager
def lyzr_agents(
    settings: Settings,
    session_id: str,
    user_id: str,
    studio_factory: Callable[..., Studio] = Studio,
) -> Iterator[LyzrRun]:
    """Find or create the Safe AI policy and both agents, and make every agent call of the run
    in Lyzr session `session_id` on behalf of `user_id`."""
    with ExitStack() as cleanup:
        studio = studio_factory(api_key=settings.lyzr_api_key.get_secret_value())
        cleanup.callback(studio.close)
        policy_fingerprint = fingerprint(SAFE_AI)
        policy = _policy(studio, f"wealth-advisor-safe-ai-{policy_fingerprint[:12]}")
        existing = {agent.name: agent for agent in studio.list_agents().agents}

        reader_settings, reader_fingerprint = versioned(
            "wealth-advisor-ips-reader",
            settings.lyzr_model,
            ips_reader.ROLE,
            ips_reader.GOAL,
            ips_reader.INSTRUCTIONS,
            output=IpsReading.model_json_schema(),
            safe_ai=policy_fingerprint,
        )
        writer_settings, writer_fingerprint = versioned(
            "wealth-advisor-briefing-writer",
            settings.lyzr_model,
            briefing.ROLE,
            briefing.GOAL,
            briefing.INSTRUCTIONS,
            safe_ai=policy_fingerprint,
        )
        reader = _agent(studio, existing, reader_settings, policy, IpsReading)
        writer = _agent(studio, existing, writer_settings, policy)
        yield LyzrRun(
            agents=Agents(
                reader=LyzrReader(reader, session_id, user_id),
                writer=LyzrWriter(writer, session_id, user_id),
            ),
            provenance=AgentProvenance(
                session_id=session_id,
                model=settings.lyzr_model,
                reader=AgentRef(id=reader.id, name=reader.name, config_sha256=reader_fingerprint),
                writer=AgentRef(id=writer.id, name=writer.name, config_sha256=writer_fingerprint),
            ),
        )


def versioned(
    name: str, model: str, role: str, goal: str, instructions: str, **extra: object
) -> tuple[AgentSettings, str]:
    """An agent's settings, named after the fingerprint of everything that shapes its output."""
    settings: AgentSettings = {
        "name": name,
        "provider": model,
        "role": role,
        "goal": goal,
        "instructions": instructions,
        "temperature": 0.0,
    }
    digest = fingerprint({**settings, **extra})
    return {**settings, "name": f"{name}-{digest[:12]}"}, digest


def fingerprint(configuration: Mapping[str, object]) -> str:
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _policy(studio: Studio, name: str) -> RAIPolicy:
    for policy in studio.list_rai_policies().policies:
        if policy.name == name:
            return policy
    return studio.create_rai_policy(name=name, **SAFE_AI)


def _agent(
    studio: Studio,
    existing: Mapping[str, Agent],
    settings: AgentSettings,
    policy: RAIPolicy,
    response_model: type[BaseModel] | None = None,
) -> Agent:
    found = existing.get(settings["name"])
    if found is None:
        return studio.create_agent(**settings, response_model=response_model, rai_policy=policy)
    if changed := _changed(found, settings):
        raise AgentDriftError(
            f"{found.name} was changed in Lyzr Studio ({', '.join(changed)} no longer match the "
            "code); delete it there and run again to recreate it"
        )
    return studio.get_agent(found.id, response_model=response_model)


def _changed(agent: Agent, settings: AgentSettings) -> list[str]:
    provider, model, _ = ModelResolver.parse(settings["provider"])
    expected: dict[str, object] = {
        "provider_id": provider.value,
        "model": model,
        "temperature": settings["temperature"],
        "agent_role": settings["role"],
        "agent_goal": settings["goal"],
        "agent_instructions": settings["instructions"],
    }
    return [field for field, value in expected.items() if not _same(getattr(agent, field), value)]


def _same(actual: object, expected: object) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return normalise(actual) == normalise(expected)
    return actual == expected
