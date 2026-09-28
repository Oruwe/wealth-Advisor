"""A tamper-evident audit ledger of every advice run and every adviser decision.

The ledger is a JSON Lines file that only ever grows. Each record carries the SHA-256 hash of
the record before it, so editing, removing or reordering any record breaks the chain from that
point on. Advice records also keep everything the run started from, so `replay` can re-derive
every number and re-check everything the agents produced.
"""

import hashlib
import json
import os
from datetime import datetime
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, ValidationError

from wealth_advisor.advisor import AdviceDossier, AgentProvenance
from wealth_advisor.agents.briefing import briefing_facts, stray_numbers
from wealth_advisor.agents.ips_reader import check_quotes
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.portfolio import Portfolio
from wealth_advisor.domain.primitives import DomainModel
from wealth_advisor.rebalancer import propose_rebalance
from wealth_advisor.suitability_gate import check_suitability
from wealth_advisor.tax import estimate_tax

GENESIS = "0" * 64
ENGINE_VERSION = version("wealth-advisor")


class AdviceInputs(DomainModel):
    """Everything an advice run started from."""

    ips_text: str
    portfolio: Portfolio
    prices: PriceSnapshot


class AdviceEntry(DomainModel):
    kind: Literal["advice"] = "advice"
    run_id: str = Field(min_length=1)
    engine_version: str = ENGINE_VERSION
    inputs: AdviceInputs
    dossier: AdviceDossier
    # Where Lyzr AIMS logged the agents' side of the run; None for runs without Lyzr agents.
    provenance: AgentProvenance | None = None


class Decision(StrEnum):
    APPROVED = "approved"
    DECLINED = "declined"


class DecisionEntry(DomainModel):
    kind: Literal["decision"] = "decision"
    advice_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Decision
    adviser_id: str = Field(min_length=1)
    note: str = ""


class LedgerRecord(DomainModel):
    sequence: int = Field(ge=0)
    recorded_at: AwareDatetime
    entry: Annotated[AdviceEntry | DecisionEntry, Field(discriminator="kind")]
    previous_hash: str
    hash: str


class LedgerError(ValueError):
    """The ledger fails verification, or an entry would break its rules."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class Ledger:
    """Append-only. Record corrections as new entries; never edit the file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def head(self) -> str:
        """The hash of the latest record. Publish it: a ledger cut short no longer ends there."""
        records = self.records()
        return records[-1].hash if records else GENESIS

    def records(self) -> list[LedgerRecord]:
        return [LedgerRecord.model_validate_json(line) for line in self._lines()]

    def record_advice(
        self,
        run_id: str,
        inputs: AdviceInputs,
        dossier: AdviceDossier,
        recorded_at: datetime,
        provenance: AgentProvenance | None = None,
    ) -> LedgerRecord:
        """Record an advice run, but only if everything in it still re-derives from its inputs."""
        entry = AdviceEntry(run_id=run_id, inputs=inputs, dossier=dossier, provenance=provenance)
        if problems := replay(entry):
            raise LedgerError(problems)
        return self._append(entry, recorded_at)

    def record_decision(
        self,
        advice_hash: str,
        decision: Decision,
        adviser_id: str,
        recorded_at: datetime,
        note: str = "",
    ) -> LedgerRecord:
        """Record an adviser's decision. An approval re-runs the advice's replay first."""
        entry = DecisionEntry(
            advice_hash=advice_hash, decision=decision, adviser_id=adviser_id, note=note
        )
        advice = {record.hash: record.entry for record in self._intact_records()}.get(advice_hash)
        if (
            decision is Decision.APPROVED
            and isinstance(advice, AdviceEntry)
            and (problems := replay(advice))
        ):
            raise LedgerError([f"the advice no longer re-verifies: {p}" for p in problems])
        return self._append(entry, recorded_at)

    def verify(self, published_head: str | None = None) -> list[str]:
        """Every problem with the ledger: records altered, removed, added or moved; entries that
        break the ledger's rules; and advice that no longer replays. Empty means it is intact."""
        lines = self._lines()
        if problems := _chain_problems(lines):
            return problems
        records: list[LedgerRecord] = []
        for number, line in enumerate(lines, start=1):
            try:
                records.append(LedgerRecord.model_validate_json(line))
            except ValidationError:
                return [f"line {number} is not a ledger record"]
        head = records[-1].hash if records else GENESIS
        if published_head is not None and head != published_head:
            problems.append(
                f"the ledger ends at {head[:12]}, not at the published head {published_head[:12]}"
            )
        problems += _rule_problems(records)
        for record in records:
            if isinstance(record.entry, AdviceEntry):
                problems += [f"record {record.sequence}: {p}" for p in replay(record.entry)]
        return problems

    def _intact_records(self) -> list[LedgerRecord]:
        """The records, if the chain holds; nothing is added to a ledger that fails it."""
        lines = self._lines()
        if problems := _chain_problems(lines):
            raise LedgerError(problems)
        return [LedgerRecord.model_validate_json(line) for line in lines]

    def _append(self, entry: AdviceEntry | DecisionEntry, recorded_at: datetime) -> LedgerRecord:
        records = self._intact_records()
        unsealed = LedgerRecord(
            sequence=len(records),
            recorded_at=recorded_at,
            entry=entry,
            previous_hash=records[-1].hash if records else GENESIS,
            hash="",
        )
        fields = unsealed.model_dump(mode="json", exclude={"hash"})
        record = unsealed.model_copy(update={"hash": _digest(fields)})
        if problems := _rule_problems([*records, record]):
            raise LedgerError(problems)
        with self.path.open("a", encoding="ascii", newline="\n") as file:
            file.write(_canonical({**fields, "hash": record.hash}) + "\n")
            file.flush()
            os.fsync(file.fileno())
        return record

    def _lines(self) -> list[str]:
        if not self.path.exists():
            return []
        # A stray non-ASCII byte decodes to U+FFFD, so it shows up as an altered record.
        text = self.path.read_text(encoding="ascii", errors="replace")
        return text.removesuffix("\n").split("\n") if text else []


def replay(entry: AdviceEntry) -> list[str]:
    """Re-derive everything code produced for an advice run from its recorded inputs, and re-check
    what the agents produced against it. Empty means the whole reasoning chain still holds."""
    inputs, dossier = entry.inputs, entry.dossier
    problems = [
        f"IPS: {p}" for p in check_quotes(inputs.ips_text, dossier.profile, dossier.ips_quotes)
    ]
    try:
        proposal = propose_rebalance(dossier.profile, inputs.portfolio, inputs.prices)
        suitability = check_suitability(dossier.profile, inputs.portfolio, inputs.prices, proposal)
        tax = estimate_tax(dossier.profile, inputs.portfolio, proposal.orders)
    except ValueError as error:
        return [*problems, f"the recorded inputs no longer produce advice: {error}"]
    for name, recorded, fresh in (
        ("proposal", dossier.proposal, proposal),
        ("suitability", dossier.suitability, suitability),
        ("tax", dossier.tax, tax),
    ):
        if recorded != fresh:
            problems.append(f"{name}: the recorded {name} differs from a fresh calculation")
    facts = briefing_facts(dossier.profile, proposal, suitability, tax)
    if stray := stray_numbers(dossier.briefing, facts):
        problems.append(f"briefing: uses numbers that are not in the facts: {', '.join(stray)}")
    return problems


def _rule_problems(records: list[LedgerRecord]) -> list[str]:
    problems: list[str] = []
    advice: dict[str, AdviceEntry] = {}
    run_ids: set[str] = set()
    decided: set[str] = set()
    for previous, record in zip([None, *records], records, strict=False):
        where = f"record {record.sequence}"
        if previous is not None and record.recorded_at < previous.recorded_at:
            problems.append(f"{where} is dated before the record it follows")
        entry = record.entry
        if isinstance(entry, AdviceEntry):
            if entry.run_id in run_ids:
                problems.append(f"{where} repeats run {entry.run_id}")
            run_ids.add(entry.run_id)
            advice[record.hash] = entry
            continue
        decided_on = advice.get(entry.advice_hash)
        if decided_on is None:
            problems.append(f"{where} decides on advice that is not earlier in the ledger")
        elif entry.advice_hash in decided:
            problems.append(f"{where} decides again on advice {entry.advice_hash[:12]}")
        elif entry.decision is Decision.APPROVED and not decided_on.dossier.suitability.approved:
            problems.append(f"{where} approves advice the suitability gate blocked")
        decided.add(entry.advice_hash)
    return problems


def _chain_problems(lines: list[str]) -> list[str]:
    previous = GENESIS
    for sequence, line in enumerate(lines):
        number = sequence + 1
        try:
            fields = json.loads(line)
            stored = fields.pop("hash")
        except json.JSONDecodeError, AttributeError, KeyError, TypeError:
            return [f"line {number} is not a ledger record"]
        if fields.get("sequence") != sequence or fields.get("previous_hash") != previous:
            return [
                f"line {number} breaks the chain: a record before it was removed, added or moved"
            ]
        if _digest(fields) != stored:
            return [f"line {number} has been altered"]
        previous = stored
    return []


def _canonical(fields: dict[str, object]) -> str:
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(fields: dict[str, object]) -> str:
    return hashlib.sha256(_canonical(fields).encode("ascii")).hexdigest()
