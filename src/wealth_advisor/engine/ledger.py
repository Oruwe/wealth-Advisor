"""Cryptographic fiduciary ledger: tamper-evident, append-only audit trail.

Every advice cycle is sealed into an AuditBlock whose SHA-256 signature covers
all payload fields. Each block embeds the previous block's signature hash, so the
chain cannot be silently truncated or reordered.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any
from uuid import uuid4

from filelock import FileLock
from pydantic import BaseModel, Field

_logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data/fiduciary_ledger.jsonl")
GENESIS = "GENESIS"

# Cached verification results keyed by "<file-mtime-ns>:<published_head>".
_VERIFICATION_CACHE: dict[str, tuple[bool, str]] = {}


# ── Serialisation helpers ──────────────────────────────────────────────────────


def _safe_default(obj: Any) -> Any:
    """Fallback for json.dumps: converts Python domain types to JSON-safe primitives."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, Fraction):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__!r} is not JSON serialisable")


def _canonical(data: dict[str, Any]) -> str:
    """Compact, sorted-key, ASCII JSON – the canonical form used for hashing."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        default=_safe_default,
        ensure_ascii=True,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Data model ─────────────────────────────────────────────────────────────────


class AuditBlock(BaseModel):
    """One immutable record in the fiduciary chain."""

    block_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: str
    previous_hash: str
    client_id: str
    client_snapshot: dict[str, Any]
    market_snapshot: dict[str, Any]
    ai_mandate: dict[str, Any]
    optimization_result: dict[str, Any]
    trades: list[dict[str, Any]]
    broker_receipt: dict[str, Any] | None
    # policy_hash records which suitability policy was active when this block was created.
    # It is intentionally excluded from the signature so that policy edits do not invalidate
    # the chain; instead, a mismatch triggers a warning during verification.
    policy_hash: str | None = None
    signature_hash: str

    @classmethod
    def create(
        cls,
        *,
        previous_hash: str,
        client_id: str,
        client_snapshot: dict[str, Any],
        market_snapshot: dict[str, Any],
        ai_mandate: dict[str, Any],
        optimization_result: dict[str, Any],
        trades: list[dict[str, Any]],
        broker_receipt: dict[str, Any] | None = None,
        policy_hash: str | None = None,
    ) -> AuditBlock:
        """Build a new block and compute its signature hash over all payload fields."""
        block_id = uuid4().hex
        timestamp = datetime.now(UTC).isoformat()
        # policy_hash is metadata; excluded from the signed payload for backward compatibility.
        payload: dict[str, Any] = {
            "block_id": block_id,
            "timestamp": timestamp,
            "previous_hash": previous_hash,
            "client_id": client_id,
            "client_snapshot": client_snapshot,
            "market_snapshot": market_snapshot,
            "ai_mandate": ai_mandate,
            "optimization_result": optimization_result,
            "trades": trades,
            "broker_receipt": broker_receipt,
        }
        sig = _sha256(_canonical(payload))
        return cls(**payload, policy_hash=policy_hash, signature_hash=sig)


# ── Ledger ─────────────────────────────────────────────────────────────────────


class FiduciaryLedger:
    """Append-only JSONL ledger of AuditBlocks with a verifiable hash chain."""

    GENESIS = GENESIS

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else _DEFAULT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    # ── Write ──────────────────────────────────────────────────────────────────

    def record_decision(
        self,
        previous_hash: str,
        client_id: str,
        client_snapshot: dict[str, Any],
        market_snapshot: dict[str, Any],
        ai_mandate: dict[str, Any],
        optimization_result: dict[str, Any],
        trades: list[dict[str, Any]],
        broker_receipt: dict[str, Any] | None = None,
        policy_hash: str | None = None,
    ) -> AuditBlock:
        """Seal one fiduciary decision into the ledger and return the block."""
        block = AuditBlock.create(
            previous_hash=previous_hash,
            client_id=client_id,
            client_snapshot=client_snapshot,
            market_snapshot=market_snapshot,
            ai_mandate=ai_mandate,
            optimization_result=optimization_result,
            trades=trades,
            broker_receipt=broker_receipt,
            policy_hash=policy_hash,
        )
        line = _canonical(block.model_dump()) + "\n"
        lock_path = str(self._path) + ".lock"
        with FileLock(lock_path, timeout=10):
            with self._path.open("a", encoding="ascii") as fh:
                fh.write(line)
        return block

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_latest_block(self) -> AuditBlock | None:
        """Return the most recently appended block, or None if the ledger is empty."""
        lines = [ln for ln in self._path.read_text(encoding="ascii").splitlines() if ln.strip()]
        return AuditBlock.model_validate_json(lines[-1]) if lines else None

    def total_blocks(self) -> int:
        return sum(1 for ln in self._path.read_text(encoding="ascii").splitlines() if ln.strip())

    # ── Verify ─────────────────────────────────────────────────────────────────

    def verify_chain(self, published_head: str | None = None) -> tuple[bool, str]:
        """Walk every block, recalculate its signature, and confirm the chain links.

        When *published_head* is given, confirms it exists anywhere in the chain
        history rather than demanding it be the absolute latest block.

        Results are cached by file modification time to avoid O(N²) dashboard loads.
        Returns (True, summary) when valid or (False, first_problem) otherwise.
        """
        try:
            mtime = str(self._path.stat().st_mtime_ns)
        except OSError:
            mtime = "missing"
        cache_key = f"{mtime}:{published_head}"
        if cache_key in _VERIFICATION_CACHE:
            return _VERIFICATION_CACHE[cache_key]

        result = self._verify_uncached(published_head)
        _VERIFICATION_CACHE[cache_key] = result
        return result

    def _verify_uncached(self, published_head: str | None) -> tuple[bool, str]:
        lines = [ln for ln in self._path.read_text(encoding="ascii").splitlines() if ln.strip()]
        if not lines:
            return True, "Empty ledger — no blocks to verify."

        blocks: list[AuditBlock] = []
        for i, line in enumerate(lines):
            try:
                blocks.append(AuditBlock.model_validate_json(line))
            except Exception as exc:
                return False, f"Block {i} could not be parsed: {exc}"

        # Lazy import to avoid a module-level circular-import risk.
        from wealth_advisor.policy.suitability import SUITABILITY_POLICY  # noqa: PLC0415

        current_policy_hash = SUITABILITY_POLICY.policy_hash

        all_hashes: set[str] = set()
        prev_sig = GENESIS
        for i, block in enumerate(blocks):
            try:
                # policy_hash is metadata; excluded from signed payload (backward compatible).
                _EXCLUDED = {"signature_hash", "policy_hash"}
                payload = {k: v for k, v in block.model_dump().items() if k not in _EXCLUDED}
                expected = _sha256(_canonical(payload))
                if block.signature_hash != expected:
                    return (
                        False,
                        f"Block {i} ({block.block_id[:12]}…) signature mismatch: "
                        f"stored {block.signature_hash[:16]}…, computed {expected[:16]}…",
                    )
                if block.previous_hash != prev_sig:
                    return (
                        False,
                        f"Block {i} links to {block.previous_hash[:12]}…"
                        f" but expected {prev_sig[:12]}…",
                    )
                if (
                    block.policy_hash is not None
                    and block.policy_hash != current_policy_hash
                ):
                    _logger.warning(
                        "Block %d (%s…) was created under a different policy version "
                        "(block: %s…, current: %s…) — historical trade is still valid.",
                        i,
                        block.block_id[:12],
                        block.policy_hash[:12],
                        current_policy_hash[:12],
                    )
                all_hashes.add(block.signature_hash)
                prev_sig = block.signature_hash
            except Exception as exc:
                return False, f"Tampering detected at block {i}: {exc}"

        if published_head is not None and published_head not in all_hashes:
            return (
                False,
                f"Published head {published_head[:12]}… not found in chain history.",
            )

        return True, f"Chain intact — {len(blocks)} block(s) verified."
