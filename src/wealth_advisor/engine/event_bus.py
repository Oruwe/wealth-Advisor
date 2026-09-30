"""Redis-streams event bus: publishes client life events and dispatches them to the brain."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from typing import Any

import redis

from wealth_advisor.engine.strategic_agent import FiduciaryBrain

logger = logging.getLogger(__name__)

STREAM_NAME = "client_events"
DEFAULT_BLOCK_MS = 5_000
DEFAULT_BATCH = 16


class WealthEventBus:
    """Thin, testable Redis-streams client for the wealth advisor's event flow.

    Producers call `publish_event`; the strategic worker calls `listen_for_triggers` and
    yields each brain evaluation so a downstream consumer (persistence, ack, notification)
    can act on it without owning the loop.
    """

    def __init__(
        self,
        brain: FiduciaryBrain | None = None,
        client: redis.Redis | None = None,
        stream_name: str = STREAM_NAME,
    ) -> None:
        self._brain = brain or FiduciaryBrain()
        self._client = client or redis.Redis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
        self._stream_name = stream_name

    def publish_event(
        self,
        event_type: str,
        client_id: str,
        payload: dict[str, Any],
    ) -> str:
        """Publish an event to the client_events stream; returns the stream entry id."""
        entry = {
            "event_type": event_type,
            "client_id": client_id,
            "payload": json.dumps(payload, sort_keys=True),
        }
        entry_id: str = self._client.xadd(self._stream_name, entry)
        logger.info(
            "event published",
            extra={
                "stream": self._stream_name,
                "entry_id": entry_id,
                "event_type": event_type,
                "client_id": client_id,
            },
        )
        return entry_id

    def listen_for_triggers(
        self,
        block_ms: int = DEFAULT_BLOCK_MS,
        batch: int = DEFAULT_BATCH,
        start_id: str = "$",
    ) -> Iterator[dict[str, Any]]:
        """Continuously read from the stream, log each event, and dispatch it to the brain.

        Yields the brain's bounded parameters for each event so callers can persist or ack
        downstream. `$` means "only events after subscription"; pass an explicit stream id
        to replay from a checkpoint.
        """
        last_id = start_id
        while True:
            responses = self._client.xread(
                {self._stream_name: last_id},
                count=batch,
                block=block_ms,
            )
            if not responses:
                continue
            for _stream, entries in responses:
                for entry_id, fields in entries:
                    last_id = entry_id
                    yield from self._dispatch(entry_id, fields)

    def _dispatch(
        self,
        entry_id: str,
        fields: dict[str, str],
    ) -> Iterator[dict[str, Any]]:
        event_type = fields.get("event_type", "unknown")
        client_id = fields.get("client_id", "unknown")
        try:
            payload = json.loads(fields.get("payload", "{}"))
        except json.JSONDecodeError:
            logger.exception(
                "dropping event with malformed payload",
                extra={"entry_id": entry_id, "event_type": event_type},
            )
            return
        logger.info(
            "event received",
            extra={
                "entry_id": entry_id,
                "event_type": event_type,
                "client_id": client_id,
            },
        )
        client_profile = payload.get("client_profile", {})
        event_payload = payload.get("event", payload)
        try:
            parameters = self._brain.evaluate_event(client_profile, event_payload)
        except Exception:
            logger.exception(
                "brain evaluation failed",
                extra={
                    "entry_id": entry_id,
                    "event_type": event_type,
                    "client_id": client_id,
                },
            )
            return
        yield {
            "entry_id": entry_id,
            "client_id": client_id,
            "event_type": event_type,
            "parameters": parameters,
        }
