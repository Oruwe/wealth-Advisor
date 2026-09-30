"""Unit tests for the engine layer: optimizer, strategic agent, event bus, pipeline.

No test touches Lyzr Studio or a real Redis instance. Every external boundary is faked in
memory so the suite runs offline and deterministically.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import SecretStr, ValidationError
from sklearn.covariance import LedoitWolf

from wealth_advisor.engine.event_bus import STREAM_NAME, WealthEventBus
from wealth_advisor.engine.market_data import MarketDataFeed, RiskEstimator
from wealth_advisor.engine.optimizer import FiduciaryOptimizer, OptimizerError
from wealth_advisor.engine.pipeline import FiduciaryPipeline, PipelineError
from wealth_advisor.engine.strategic_agent import (
    AGENT_NAME,
    FiduciaryBrain,
    RiskParameters,
)
from wealth_advisor.settings import Settings


# ------------------------------------------------------------------------- fakes


class _FakeAgent:
    """Lyzr agent stand-in: records the messages it was sent and returns a scripted reply."""

    def __init__(self, reply: object, agent_id: str = "agent-1") -> None:
        self.reply = reply
        self.id = agent_id
        self.name = AGENT_NAME
        self.messages: list[str] = []

    def run(self, message: str, **_: Any) -> object:
        self.messages.append(message)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class _FakeStudio:
    """Studio in memory: no policies, one agent by name."""

    def __init__(self, reply: object, existing: list[_FakeAgent] | None = None) -> None:
        self._reply = reply
        self._existing: list[_FakeAgent] = list(existing or [])
        self.created: list[dict[str, Any]] = []
        self.fetched: list[str] = []

    def list_agents(self) -> SimpleNamespace:
        return SimpleNamespace(agents=list(self._existing))

    def create_agent(self, **config: Any) -> _FakeAgent:
        self.created.append(config)
        agent = _FakeAgent(self._reply, agent_id=f"agent-{len(self.created)}")
        agent.name = config["name"]
        self._existing.append(agent)
        return agent

    def get_agent(self, agent_id: str, response_model: object = None) -> _FakeAgent:
        self.fetched.append(agent_id)
        for agent in self._existing:
            if agent.id == agent_id:
                return agent
        raise LookupError(agent_id)


class _Drained(Exception):
    """Sentinel raised by `_FakeRedis.xread` once every queued batch has been consumed.

    `WealthEventBus.listen_for_triggers` runs `while True`; in production it blocks on Redis.
    In tests we replace that block with a deterministic exit signal: the fake raises `_Drained`,
    which propagates out of the generator so `_drain()` can collect the results so far.
    """


class _FakeRedis:
    """Redis stand-in supporting xadd and xread. Batches are drained in FIFO order."""

    def __init__(self) -> None:
        self._counter = 0
        self.published: list[tuple[str, dict[str, str]]] = []
        self._batches: list[list[tuple[str, dict[str, str]]]] = []

    def enqueue(self, fields_list: list[dict[str, str]]) -> None:
        """Test helper: schedule a batch for the next xread call."""
        batch: list[tuple[str, dict[str, str]]] = []
        for fields in fields_list:
            self._counter += 1
            batch.append((f"{self._counter:010d}-0", fields))
        self._batches.append(batch)

    def xadd(self, stream: str, fields: dict[str, str]) -> str:
        self._counter += 1
        entry_id = f"{self._counter:010d}-0"
        self.published.append((entry_id, fields))
        return entry_id

    def xread(
        self,
        streams: dict[str, str],
        count: int = 10,
        block: int | None = None,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        stream_name = next(iter(streams.keys()))
        if not self._batches:
            raise _Drained
        return [(stream_name, self._batches.pop(0))]


def _drain(bus: WealthEventBus, start_id: str = "0") -> list[dict[str, Any]]:
    """Consume `bus.listen_for_triggers` until the fake redis raises `_Drained`."""
    results: list[dict[str, Any]] = []
    try:
        for item in bus.listen_for_triggers(block_ms=0, start_id=start_id):
            results.append(item)
    except _Drained:
        pass
    return results


class _FakeBrain:
    """Strategic-agent stand-in: returns a scripted dict or raises a scripted exception."""

    def __init__(self, reply: dict[str, float] | Exception) -> None:
        self.reply = reply
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def evaluate_event(
        self,
        client_profile: dict[str, Any],
        event_payload: dict[str, Any],
    ) -> dict[str, float]:
        self.calls.append((client_profile, event_payload))
        if isinstance(self.reply, Exception):
            raise self.reply
        return dict(self.reply)


class _FakeOptimizer:
    """Optimizer stand-in: returns a scripted weight vector."""

    def __init__(self, weights: list[float] | Exception) -> None:
        self.weights = weights
        self.calls: list[dict[str, Any]] = []

    def optimize_allocation(self, **kwargs: Any) -> list[float]:
        self.calls.append(kwargs)
        if isinstance(self.weights, Exception):
            raise self.weights
        return list(self.weights)


# ---------------------------------------------------------------------- fixtures


@pytest.fixture
def settings() -> Settings:
    return Settings(lyzr_api_key=SecretStr("sk-test"), lyzr_model="openai/gpt-4.1")


@pytest.fixture
def market_data() -> dict[str, Any]:
    return {
        "asset_names": ["VTI", "VEA", "BND", "CASH"],
        "expected_returns": [0.08, 0.07, 0.03, 0.01],
        "covariance_matrix": [
            [0.04, 0.02, 0.00, 0.00],
            [0.02, 0.05, 0.00, 0.00],
            [0.00, 0.00, 0.01, 0.00],
            [0.00, 0.00, 0.00, 0.0001],
        ],
        "equity_indices": [0, 1],
    }


# --------------------------------------------------------------------- optimizer


def _default_optimizer_inputs() -> dict[str, Any]:
    return {
        "expected_returns": [0.08, 0.07, 0.03, 0.01],
        "covariance_matrix": [
            [0.04, 0.02, 0.00, 0.00],
            [0.02, 0.05, 0.00, 0.00],
            [0.00, 0.00, 0.01, 0.00],
            [0.00, 0.00, 0.00, 0.0001],
        ],
        "equity_indices": [0, 1],
    }


def test_optimizer_weights_sum_to_one() -> None:
    weights = FiduciaryOptimizer().optimize_allocation(
        **_default_optimizer_inputs(),
        target_risk_aversion=3.0,
        max_equity_exposure=0.8,
    )
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)


def test_optimizer_is_long_only() -> None:
    weights = FiduciaryOptimizer().optimize_allocation(
        **_default_optimizer_inputs(),
        target_risk_aversion=3.0,
        max_equity_exposure=0.8,
    )
    assert all(w >= -1e-9 for w in weights)


def test_optimizer_respects_equity_cap_moderate() -> None:
    weights = FiduciaryOptimizer().optimize_allocation(
        **_default_optimizer_inputs(),
        target_risk_aversion=2.0,
        max_equity_exposure=0.8,
    )
    equity_weight = weights[0] + weights[1]
    assert equity_weight <= 0.8 + 1e-6


def test_optimizer_respects_equity_cap_conservative() -> None:
    weights = FiduciaryOptimizer().optimize_allocation(
        **_default_optimizer_inputs(),
        target_risk_aversion=8.0,
        max_equity_exposure=0.2,
    )
    equity_weight = weights[0] + weights[1]
    assert equity_weight <= 0.2 + 1e-6


def test_optimizer_higher_risk_aversion_reduces_equity() -> None:
    inputs = _default_optimizer_inputs()
    optimizer = FiduciaryOptimizer()
    low = optimizer.optimize_allocation(
        **inputs, target_risk_aversion=1.5, max_equity_exposure=0.8
    )
    high = optimizer.optimize_allocation(
        **inputs, target_risk_aversion=9.0, max_equity_exposure=0.8
    )
    assert (high[0] + high[1]) < (low[0] + low[1])


def test_optimizer_rejects_shape_mismatch() -> None:
    inputs = _default_optimizer_inputs()
    inputs["expected_returns"] = [0.08, 0.07, 0.03]
    with pytest.raises(OptimizerError, match="does not match"):
        FiduciaryOptimizer().optimize_allocation(
            **inputs, target_risk_aversion=3.0, max_equity_exposure=0.5
        )


def test_optimizer_rejects_non_symmetric_covariance() -> None:
    inputs = _default_optimizer_inputs()
    inputs["covariance_matrix"] = [
        [0.04, 0.10, 0.00, 0.00],
        [0.02, 0.05, 0.00, 0.00],
        [0.00, 0.00, 0.01, 0.00],
        [0.00, 0.00, 0.00, 0.0001],
    ]
    with pytest.raises(OptimizerError, match="symmetric"):
        FiduciaryOptimizer().optimize_allocation(
            **inputs, target_risk_aversion=3.0, max_equity_exposure=0.5
        )


def test_optimizer_rejects_empty_returns() -> None:
    with pytest.raises(OptimizerError, match="empty"):
        FiduciaryOptimizer().optimize_allocation(
            expected_returns=[],
            covariance_matrix=[[]],
            equity_indices=[],
            target_risk_aversion=3.0,
            max_equity_exposure=0.5,
        )


def test_optimizer_rejects_out_of_range_risk_aversion() -> None:
    with pytest.raises(OptimizerError, match="target_risk_aversion"):
        FiduciaryOptimizer().optimize_allocation(
            **_default_optimizer_inputs(),
            target_risk_aversion=11.0,
            max_equity_exposure=0.5,
        )


def test_optimizer_rejects_out_of_range_equity_cap() -> None:
    with pytest.raises(OptimizerError, match="max_equity_exposure"):
        FiduciaryOptimizer().optimize_allocation(
            **_default_optimizer_inputs(),
            target_risk_aversion=3.0,
            max_equity_exposure=1.5,
        )


def test_optimizer_rejects_out_of_range_equity_index() -> None:
    inputs = _default_optimizer_inputs()
    inputs["equity_indices"] = [0, 4]
    with pytest.raises(OptimizerError, match="out of range"):
        FiduciaryOptimizer().optimize_allocation(
            **inputs, target_risk_aversion=3.0, max_equity_exposure=0.5
        )


def test_optimizer_rejects_multidimensional_returns() -> None:
    inputs = _default_optimizer_inputs()
    inputs["expected_returns"] = [[0.08, 0.07], [0.03, 0.01]]
    with pytest.raises(OptimizerError, match="1-D"):
        FiduciaryOptimizer().optimize_allocation(
            **inputs, target_risk_aversion=3.0, max_equity_exposure=0.5
        )


# ---------------------------------------------------------------- strategic agent


def _valid_parameters() -> RiskParameters:
    return RiskParameters(target_risk_aversion=4.0, max_equity_exposure=0.6)


def test_risk_parameters_reject_out_of_range() -> None:
    with pytest.raises(ValidationError):
        RiskParameters(target_risk_aversion=0.5, max_equity_exposure=0.5)
    with pytest.raises(ValidationError):
        RiskParameters(target_risk_aversion=10.5, max_equity_exposure=0.5)
    with pytest.raises(ValidationError):
        RiskParameters(target_risk_aversion=5.0, max_equity_exposure=1.2)


def test_risk_parameters_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RiskParameters.model_validate(
            {"target_risk_aversion": 5.0, "max_equity_exposure": 0.5, "sneaky": 42}
        )


def test_brain_creates_agent_when_missing(settings: Settings) -> None:
    studio = _FakeStudio(reply=_valid_parameters())
    brain = FiduciaryBrain(settings=settings, studio=studio)  # type: ignore[arg-type]
    result = brain.evaluate_event(
        {"client_id": "C-1001", "profile": "moderate"},
        {"event_type": "job_loss"},
    )
    assert result == {"target_risk_aversion": 4.0, "max_equity_exposure": 0.6}
    assert len(studio.created) == 1
    assert studio.created[0]["name"] == AGENT_NAME


def test_brain_reuses_existing_agent(settings: Settings) -> None:
    existing = _FakeAgent(reply=_valid_parameters())
    existing.name = AGENT_NAME
    studio = _FakeStudio(reply=_valid_parameters(), existing=[existing])
    brain = FiduciaryBrain(settings=settings, studio=studio)  # type: ignore[arg-type]
    brain.evaluate_event({"client_id": "C-1"}, {"event_type": "layoff"})
    assert studio.created == []
    assert studio.fetched == [existing.id]


def test_brain_fences_inputs_in_tags(settings: Settings) -> None:
    studio = _FakeStudio(reply=_valid_parameters())
    brain = FiduciaryBrain(settings=settings, studio=studio)  # type: ignore[arg-type]
    brain.evaluate_event(
        {"client_id": "C-1001", "profile": "moderate"},
        {"event_type": "diagnosis"},
    )
    agent = studio._existing[-1]
    (message,) = agent.messages
    assert "<client_profile>" in message and "</client_profile>" in message
    assert "<event>" in message and "</event>" in message
    assert "moderate" in message
    assert "diagnosis" in message


def test_brain_rejects_wrong_return_type(settings: Settings) -> None:
    studio = _FakeStudio(reply={"not": "risk-parameters"})
    brain = FiduciaryBrain(settings=settings, studio=studio)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected RiskParameters"):
        brain.evaluate_event({}, {})


def test_brain_returns_bounded_dict(settings: Settings) -> None:
    studio = _FakeStudio(
        reply=RiskParameters(target_risk_aversion=9.5, max_equity_exposure=0.15)
    )
    brain = FiduciaryBrain(settings=settings, studio=studio)  # type: ignore[arg-type]
    result = brain.evaluate_event({}, {})
    assert 1.0 <= result["target_risk_aversion"] <= 10.0
    assert 0.0 <= result["max_equity_exposure"] <= 1.0


# ------------------------------------------------------------------- event bus


def _good_fields(client_id: str = "C-1") -> dict[str, str]:
    return {
        "event_type": "layoff",
        "client_id": client_id,
        "payload": json.dumps({"client_profile": {"risk": "moderate"}, "event": {"delta": -1}}),
    }


def test_publish_event_writes_to_stream() -> None:
    fake_redis = _FakeRedis()
    brain = _FakeBrain({"target_risk_aversion": 5.0, "max_equity_exposure": 0.5})
    bus = WealthEventBus(brain=brain, client=fake_redis)  # type: ignore[arg-type]

    entry_id = bus.publish_event(
        event_type="layoff",
        client_id="C-1001",
        payload={"impact": "moderate"},
    )

    assert entry_id.endswith("-0")
    (stored_id, stored_fields) = fake_redis.published[0]
    assert stored_id == entry_id
    assert stored_fields["event_type"] == "layoff"
    assert stored_fields["client_id"] == "C-1001"
    assert json.loads(stored_fields["payload"]) == {"impact": "moderate"}


def test_listen_dispatches_events_to_brain() -> None:
    fake_redis = _FakeRedis()
    fake_redis.enqueue([_good_fields("C-1"), _good_fields("C-2")])
    brain = _FakeBrain({"target_risk_aversion": 5.0, "max_equity_exposure": 0.4})
    bus = WealthEventBus(brain=brain, client=fake_redis)  # type: ignore[arg-type]

    results = _drain(bus)

    assert len(results) == 2
    assert [r["client_id"] for r in results] == ["C-1", "C-2"]
    assert [r["parameters"] for r in results] == [
        {"target_risk_aversion": 5.0, "max_equity_exposure": 0.4},
        {"target_risk_aversion": 5.0, "max_equity_exposure": 0.4},
    ]
    assert len(brain.calls) == 2


def test_listen_drops_malformed_payload_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_redis = _FakeRedis()
    fake_redis.enqueue(
        [
            {"event_type": "bad", "client_id": "C-1", "payload": "not-json"},
            _good_fields("C-2"),
        ]
    )
    brain = _FakeBrain({"target_risk_aversion": 5.0, "max_equity_exposure": 0.4})
    bus = WealthEventBus(brain=brain, client=fake_redis)  # type: ignore[arg-type]

    caplog.set_level(logging.ERROR, logger="wealth_advisor.engine.event_bus")
    results = _drain(bus)

    assert len(results) == 1
    assert results[0]["client_id"] == "C-2"
    assert len(brain.calls) == 1
    assert any("malformed payload" in record.message for record in caplog.records)


def test_listen_skips_events_when_brain_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_redis = _FakeRedis()
    fake_redis.enqueue([_good_fields("C-1"), _good_fields("C-2")])
    brain = _FakeBrain(RuntimeError("boom"))
    bus = WealthEventBus(brain=brain, client=fake_redis)  # type: ignore[arg-type]

    caplog.set_level(logging.ERROR, logger="wealth_advisor.engine.event_bus")
    results = _drain(bus)

    assert results == []
    assert len(brain.calls) == 2
    assert any("brain evaluation failed" in r.message for r in caplog.records)


def test_listen_recovers_across_multiple_xread_batches() -> None:
    fake_redis = _FakeRedis()
    fake_redis.enqueue([_good_fields("C-1")])
    fake_redis.enqueue([_good_fields("C-2")])
    brain = _FakeBrain({"target_risk_aversion": 5.0, "max_equity_exposure": 0.4})
    bus = WealthEventBus(brain=brain, client=fake_redis)  # type: ignore[arg-type]

    results = _drain(bus)

    assert [r["client_id"] for r in results] == ["C-1", "C-2"]


def test_stream_name_is_client_events() -> None:
    assert STREAM_NAME == "client_events"


# --------------------------------------------------------------------- pipeline


def _make_pipeline(
    brain_reply: dict[str, float] | Exception,
    weights: list[float] | Exception,
    settings: Settings,
) -> tuple[FiduciaryPipeline, _FakeBrain, _FakeOptimizer]:
    brain = _FakeBrain(brain_reply)
    optimizer = _FakeOptimizer(weights)
    event_bus = WealthEventBus(brain=brain, client=_FakeRedis())  # type: ignore[arg-type]
    pipeline = FiduciaryPipeline(brain=brain, optimizer=optimizer, event_bus=event_bus)  # type: ignore[arg-type]
    _ = settings  # kept for test signature parity
    return pipeline, brain, optimizer


def test_pipeline_returns_full_result_dict(
    settings: Settings, market_data: dict[str, Any]
) -> None:
    pipeline, brain, optimizer = _make_pipeline(
        brain_reply={"target_risk_aversion": 4.0, "max_equity_exposure": 0.6},
        weights=[0.3, 0.2, 0.4, 0.1],
        settings=settings,
    )

    result = pipeline.process_event(
        client_profile={"client_id": "C-1001", "profile": "moderate"},
        event_payload={"event_type": "job_loss"},
        market_data=market_data,
    )

    assert result["client_id"] == "C-1001"
    assert result["event_type"] == "job_loss"
    assert result["risk_parameters"] == {
        "target_risk_aversion": 4.0,
        "max_equity_exposure": 0.6,
    }
    assert result["target_weights"] == {
        "VTI": 0.3,
        "VEA": 0.2,
        "BND": 0.4,
        "CASH": 0.1,
    }
    assert "timestamp" in result
    assert len(brain.calls) == 1
    assert optimizer.calls[0]["target_risk_aversion"] == 4.0
    assert optimizer.calls[0]["max_equity_exposure"] == 0.6


def test_pipeline_rejects_missing_market_data_keys(
    settings: Settings, market_data: dict[str, Any]
) -> None:
    pipeline, _, _ = _make_pipeline(
        brain_reply={"target_risk_aversion": 4.0, "max_equity_exposure": 0.6},
        weights=[0.25, 0.25, 0.25, 0.25],
        settings=settings,
    )
    market_data.pop("covariance_matrix")

    with pytest.raises(PipelineError, match="missing keys"):
        pipeline.process_event(
            client_profile={"client_id": "C-1"},
            event_payload={"event_type": "diag"},
            market_data=market_data,
        )


def test_pipeline_rejects_weight_length_mismatch(
    settings: Settings, market_data: dict[str, Any]
) -> None:
    pipeline, _, _ = _make_pipeline(
        brain_reply={"target_risk_aversion": 4.0, "max_equity_exposure": 0.6},
        weights=[0.5, 0.5],
        settings=settings,
    )

    with pytest.raises(PipelineError, match="but market_data lists"):
        pipeline.process_event(
            client_profile={"client_id": "C-1"},
            event_payload={"event_type": "diag"},
            market_data=market_data,
        )


def test_pipeline_wraps_brain_errors(
    settings: Settings, market_data: dict[str, Any]
) -> None:
    pipeline, _, _ = _make_pipeline(
        brain_reply=RuntimeError("agent down"),
        weights=[0.25, 0.25, 0.25, 0.25],
        settings=settings,
    )

    with pytest.raises(PipelineError, match="strategic evaluation failed"):
        pipeline.process_event(
            client_profile={"client_id": "C-1"},
            event_payload={"event_type": "diag"},
            market_data=market_data,
        )


def test_pipeline_wraps_optimizer_errors(
    settings: Settings, market_data: dict[str, Any]
) -> None:
    pipeline, _, _ = _make_pipeline(
        brain_reply={"target_risk_aversion": 4.0, "max_equity_exposure": 0.6},
        weights=RuntimeError("solver blew up"),
        settings=settings,
    )

    with pytest.raises(PipelineError, match="optimizer failed"):
        pipeline.process_event(
            client_profile={"client_id": "C-1"},
            event_payload={"event_type": "diag"},
            market_data=market_data,
        )


def test_pipeline_end_to_end_with_real_optimizer(
    settings: Settings, market_data: dict[str, Any]
) -> None:
    """Wire the real optimizer to a fake brain and check the equity cap is respected."""
    brain = _FakeBrain({"target_risk_aversion": 8.0, "max_equity_exposure": 0.2})
    optimizer = FiduciaryOptimizer()
    event_bus = WealthEventBus(brain=brain, client=_FakeRedis())  # type: ignore[arg-type]
    pipeline = FiduciaryPipeline(brain=brain, optimizer=optimizer, event_bus=event_bus)  # type: ignore[arg-type]

    result = pipeline.process_event(
        client_profile={"client_id": "C-2002", "profile": "conservative"},
        event_payload={"event_type": "near_retirement"},
        market_data=market_data,
    )

    weights = result["target_weights"]
    equity_weight = weights["VTI"] + weights["VEA"]
    assert equity_weight <= 0.2 + 1e-6
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(w >= -1e-9 for w in weights.values())


def test_pipeline_uses_defaults_when_client_profile_missing_id(
    settings: Settings, market_data: dict[str, Any]
) -> None:
    pipeline, _, _ = _make_pipeline(
        brain_reply={"target_risk_aversion": 3.0, "max_equity_exposure": 0.7},
        weights=[0.25, 0.25, 0.25, 0.25],
        settings=settings,
    )

    result = pipeline.process_event(
        client_profile={},
        event_payload={},
        market_data=market_data,
    )

    assert result["client_id"] == "unknown"
    assert result["event_type"] == "unknown"


def test_pipeline_market_data_length_mismatch_returns_expected_message(
    settings: Settings, market_data: dict[str, Any]
) -> None:
    pipeline, _, _ = _make_pipeline(
        brain_reply={"target_risk_aversion": 3.0, "max_equity_exposure": 0.7},
        weights=[0.25, 0.25, 0.25, 0.25],
        settings=settings,
    )
    market_data["expected_returns"] = [0.08, 0.07, 0.03]

    with pytest.raises(PipelineError, match="expected_returns has 3"):
        pipeline.process_event(
            client_profile={"client_id": "C-1"},
            event_payload={"event_type": "diag"},
            market_data=market_data,
        )


def test_pipeline_market_data_equity_index_out_of_range(
    settings: Settings, market_data: dict[str, Any]
) -> None:
    pipeline, _, _ = _make_pipeline(
        brain_reply={"target_risk_aversion": 3.0, "max_equity_exposure": 0.7},
        weights=[0.25, 0.25, 0.25, 0.25],
        settings=settings,
    )
    market_data["equity_indices"] = [0, 99]

    with pytest.raises(PipelineError, match="out of range"):
        pipeline.process_event(
            client_profile={"client_id": "C-1"},
            event_payload={"event_type": "diag"},
            market_data=market_data,
        )


def test_pipeline_defaults_can_construct_without_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """FiduciaryPipeline() should build with defaults given a valid settings env."""
    monkeypatch.setenv("LYZR_API_KEY", "sk-test")

    calls: list[str] = []

    class _RecordingBrain:
        def evaluate_event(self, *_: Any, **__: Any) -> dict[str, float]:
            calls.append("brain")
            return {"target_risk_aversion": 3.0, "max_equity_exposure": 0.7}

    class _RecordingOptimizer:
        def optimize_allocation(self, **_: Any) -> list[float]:
            calls.append("optimizer")
            return [0.25, 0.25, 0.25, 0.25]

    class _RecordingBus:
        def __init__(self, *_: Any, **__: Any) -> None:
            calls.append("bus")

    monkeypatch.setattr(
        "wealth_advisor.engine.pipeline.FiduciaryBrain", _RecordingBrain
    )
    monkeypatch.setattr(
        "wealth_advisor.engine.pipeline.FiduciaryOptimizer", _RecordingOptimizer
    )
    monkeypatch.setattr(
        "wealth_advisor.engine.pipeline.WealthEventBus", _RecordingBus
    )

    pipeline = FiduciaryPipeline()
    assert "bus" in calls
    result = pipeline.process_event(
        client_profile={"client_id": "C-1"},
        event_payload={"event_type": "test"},
        market_data={
            "asset_names": ["A", "B", "C", "D"],
            "expected_returns": [0.1, 0.1, 0.1, 0.1],
            "covariance_matrix": np.eye(4).tolist(),
            "equity_indices": [0],
        },
    )
    assert result["target_weights"] == {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}


# ----------------------------------------------------------------- market data


def _make_prices(symbols: list[str], n: int = 100) -> pd.DataFrame:
    """Deterministic synthetic daily close prices for unit tests."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {s: 100.0 + rng.standard_normal(n).cumsum() for s in symbols}, index=idx
    )


def test_market_feed_multi_index_multiple_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    """MultiIndex path (common yfinance format, multiple tickers): prices DataFrame returned."""
    syms = ["VOO", "AGG"]
    df = _make_prices(syms)
    raw = pd.concat({"Close": df}, axis=1)  # MultiIndex columns
    monkeypatch.setattr("wealth_advisor.engine.market_data.yf.download", lambda *a, **kw: raw)

    result = MarketDataFeed.get_historical_prices(syms, years=2)
    assert list(result.columns) == syms
    assert len(result) == len(df)
    assert not result.isnull().any().any()


def test_market_feed_multi_index_single_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    """MultiIndex single-ticker path: raw['Close'] returns a Series, converted to DataFrame."""
    syms = ["VOO"]
    df = _make_prices(syms)
    raw = pd.DataFrame(
        df["VOO"].to_numpy(),
        index=df.index,
        columns=pd.MultiIndex.from_tuples([("Close", "VOO")]),
    )
    monkeypatch.setattr("wealth_advisor.engine.market_data.yf.download", lambda *a, **kw: raw)

    result = MarketDataFeed.get_historical_prices(syms, years=1)
    assert "VOO" in result.columns


def test_market_feed_flat_ticker_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-MultiIndex path where raw already has ticker symbols as column names."""
    syms = ["VOO", "AGG"]
    df = _make_prices(syms)
    monkeypatch.setattr("wealth_advisor.engine.market_data.yf.download", lambda *a, **kw: df)

    result = MarketDataFeed.get_historical_prices(syms, years=1)
    assert list(result.columns) == syms


def test_market_feed_flat_close_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-MultiIndex path with a literal 'Close' column (legacy edge case)."""
    syms = ["VOO"]
    df = _make_prices(["VOO"])
    flat = df.rename(columns={"VOO": "Close"})
    monkeypatch.setattr("wealth_advisor.engine.market_data.yf.download", lambda *a, **kw: flat)

    result = MarketDataFeed.get_historical_prices(syms, years=1)
    assert isinstance(result, pd.DataFrame)


def test_market_feed_forward_fills_nans(monkeypatch: pytest.MonkeyPatch) -> None:
    """NaN gaps in the middle of the series are forward-filled."""
    syms = ["VOO", "AGG"]
    df = _make_prices(syms, n=20)
    df.iloc[5, 0] = float("nan")
    df.iloc[12, 1] = float("nan")
    raw = pd.concat({"Close": df}, axis=1)
    monkeypatch.setattr("wealth_advisor.engine.market_data.yf.download", lambda *a, **kw: raw)

    result = MarketDataFeed.get_historical_prices(syms, years=1)
    assert not result.isnull().any().any()


def test_risk_estimator_returns_annualised() -> None:
    """Expected return = mean daily return × 252, keyed by symbol."""
    syms = ["VOO", "AGG"]
    df = _make_prices(syms, n=100)
    er = RiskEstimator.compute_expected_returns(df)

    assert set(er.keys()) == set(syms)
    daily = df.pct_change().dropna()
    for s in syms:
        assert er[s] == pytest.approx(float(daily[s].mean() * 252), rel=1e-9)


def test_risk_estimator_covariance_shape_and_symmetry() -> None:
    """Covariance output has correct shape, labels, and is symmetric."""
    syms = ["VOO", "VEA", "AGG"]
    df = _make_prices(syms, n=150)
    cov = RiskEstimator.compute_covariance_matrix(df)

    assert cov.shape == (len(syms), len(syms))
    assert list(cov.columns) == syms
    assert list(cov.index) == syms
    assert np.allclose(cov.values, cov.values.T, atol=1e-10)


def test_risk_estimator_covariance_positive_semidefinite() -> None:
    """Ledoit-Wolf shrinkage guarantees a PSD covariance matrix."""
    syms = ["VOO", "VEA", "AGG", "TLT"]
    df = _make_prices(syms, n=200)
    cov = RiskEstimator.compute_covariance_matrix(df)

    eigvals = np.linalg.eigvalsh(cov.values)
    assert (eigvals >= -1e-8).all()


def test_risk_estimator_covariance_is_annualised() -> None:
    """Covariance matrix is scaled by 252 (daily → annual)."""
    syms = ["VOO", "AGG"]
    df = _make_prices(syms, n=100)
    returns = df.pct_change().dropna()
    lw = LedoitWolf()
    lw.fit(returns.to_numpy())
    expected_cov = lw.covariance_ * 252

    cov = RiskEstimator.compute_covariance_matrix(df)
    assert np.allclose(cov.values, expected_cov, rtol=1e-9)
