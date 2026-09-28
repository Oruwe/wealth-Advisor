"""Inspect the audit ledger and record adviser decisions in it.

    uv run python scripts/ledger.py list
    uv run python scripts/ledger.py verify [--head HASH]
    uv run python scripts/ledger.py approve HASH --adviser NAME [--note TEXT]
    uv run python scripts/ledger.py decline HASH --adviser NAME [--note TEXT]

HASH can be the first few characters of an advice record's hash, as `list` shows it.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from wealth_advisor.formatting import dollars
from wealth_advisor.ledger import (
    GENESIS,
    AdviceEntry,
    Decision,
    Ledger,
    LedgerError,
    LedgerRecord,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--path", type=Path, default=Path("ledger.jsonl"), help="ledger file")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show one line per record")
    verify = commands.add_parser("verify", help="check the chain, the rules and every replay")
    verify.add_argument("--head", help="a head hash published earlier, to detect truncation")
    for name in ("approve", "decline"):
        decide = commands.add_parser(name, help=f"{name} an advice run")
        decide.add_argument("advice_hash")
        decide.add_argument("--adviser", required=True, help="who is deciding")
        decide.add_argument("--note", default="", help="why")
    args = parser.parse_args(argv)
    ledger = Ledger(args.path)

    if args.command == "verify":
        problems = ledger.verify(published_head=args.head)
        for problem in problems:
            print(f"FAIL {problem}")
        count = len(ledger.records()) if not problems else 0
        print("FAILED" if problems else f"OK: {count} records intact, every advice run replays")
        return 1 if problems else 0
    try:
        records = ledger.records()
    except ValueError:
        print("The ledger is damaged; run `verify` to see where.")
        return 1
    if args.command == "list":
        for record in records:
            print(_describe(record))
        print(f"head {records[-1].hash if records else GENESIS}")
        return 0

    matches = [
        record.hash
        for record in records
        if isinstance(record.entry, AdviceEntry) and record.hash.startswith(args.advice_hash)
    ]
    if len(matches) != 1:
        print(f"REFUSED {args.advice_hash!r} matches {len(matches)} advice records, not 1")
        return 1
    decision = Decision.APPROVED if args.command == "approve" else Decision.DECLINED
    try:
        record = ledger.record_decision(
            matches[0], decision, args.adviser, datetime.now(UTC), args.note
        )
    except LedgerError as error:
        for problem in error.problems:
            print(f"REFUSED {problem}")
        return 1
    print(f"Recorded as record {record.sequence}: {_describe(record)}")
    return 0


def _describe(record: LedgerRecord) -> str:
    when = f"#{record.sequence} {record.recorded_at:%Y-%m-%d %H:%M}"
    entry = record.entry
    if isinstance(entry, AdviceEntry):
        dossier = entry.dossier
        verdict = "passed the gate" if dossier.suitability.approved else "BLOCKED by the gate"
        source = (
            f"Lyzr session {entry.provenance.session_id} via {entry.provenance.model}"
            if entry.provenance
            else f"run {entry.run_id}"
        )
        return (
            f"{when} advice {record.hash[:12]} for {dossier.profile.client_id}: "
            f"{len(dossier.proposal.orders)} orders, {verdict}, "
            f"est. tax {dollars(dossier.tax.estimated_tax)} ({source})"
        )
    note = f": {entry.note}" if entry.note else ""
    return f"{when} {entry.decision} by {entry.adviser_id} on {entry.advice_hash[:12]}{note}"


if __name__ == "__main__":
    sys.exit(main())
