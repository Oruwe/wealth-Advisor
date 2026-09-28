import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path

import pytest
from fakes import GROUNDED_BRIEFING, FakeReader, FakeWriter, faithful_reading

from wealth_advisor import ledger as ledger_module
from wealth_advisor.advisor import AdviceDossier, Agents, advise
from wealth_advisor.demo import DEMO_IPS, demo_portfolio, demo_prices
from wealth_advisor.domain.orders import RebalanceProposal
from wealth_advisor.domain.suitability import Rule, Violation
from wealth_advisor.ledger import (
    GENESIS,
    AdviceEntry,
    AdviceInputs,
    Decision,
    DecisionEntry,
    Ledger,
    LedgerError,
    replay,
)

AT = datetime(2026, 9, 28, 10, 0, tzinfo=UTC)
LATER = AT + timedelta(minutes=5)


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl")


@pytest.fixture
def inputs() -> AdviceInputs:
    return AdviceInputs(ips_text=DEMO_IPS, portfolio=demo_portfolio(), prices=demo_prices())


@pytest.fixture
def dossier() -> AdviceDossier:
    agents = Agents(reader=FakeReader(faithful_reading()), writer=FakeWriter(GROUNDED_BRIEFING))
    return advise(DEMO_IPS, demo_portfolio(), demo_prices(), agents)


@pytest.fixture
def approved(ledger: Ledger, inputs: AdviceInputs, dossier: AdviceDossier) -> Ledger:
    """A ledger holding one advice run and the adviser's approval of it."""
    advice = ledger.record_advice("run-1", inputs, dossier, AT)
    ledger.record_decision(advice.hash, Decision.APPROVED, "adviser-7", LATER)
    return ledger


def seal(ledger: Ledger, fields: dict[str, object]) -> str:
    """Append a record with a valid hash but no rule checks, as someone forging a ledger would."""
    fields = {"sequence": len(ledger.records()), "previous_hash": ledger.head, **fields}
    digest = ledger_module._digest(fields)
    with ledger.path.open("a", encoding="ascii", newline="\n") as file:
        file.write(ledger_module._canonical({**fields, "hash": digest}) + "\n")
    return digest


def forge(ledger: Ledger, entry: AdviceEntry | DecisionEntry, recorded_at: datetime = AT) -> str:
    return seal(
        ledger, {"recorded_at": recorded_at.isoformat(), "entry": entry.model_dump(mode="json")}
    )


def no_trades(proposal: RebalanceProposal) -> Callable[..., RebalanceProposal]:
    """A stand-in rebalancer, as if its logic had changed since the advice was recorded."""
    return lambda *_: proposal.model_copy(update={"orders": ()})


def rewrite(ledger: Ledger, change: Callable[[list[str]], list[str]]) -> None:
    lines = ledger.path.read_text(encoding="ascii").splitlines()
    ledger.path.write_text("".join(f"{line}\n" for line in change(lines)), encoding="ascii")


def test_a_new_ledger_is_empty_and_intact(ledger: Ledger) -> None:
    assert (ledger.records(), ledger.head, ledger.verify()) == ([], GENESIS, [])


def test_an_empty_file_is_an_empty_ledger(ledger: Ledger) -> None:
    ledger.path.touch()

    assert (ledger.records(), ledger.head, ledger.verify()) == ([], GENESIS, [])


def test_reports_bytes_that_are_not_ascii_as_an_alteration(approved: Ledger) -> None:
    damaged = approved.path.read_bytes().replace(b'"adviser-7"', b'"adviser-\xe9"')
    approved.path.write_bytes(damaged)

    assert approved.verify() == ["line 2 has been altered"]


def test_chains_each_record_to_the_one_before(
    ledger: Ledger, inputs: AdviceInputs, dossier: AdviceDossier
) -> None:
    first = ledger.record_advice("run-1", inputs, dossier, AT)
    second = ledger.record_advice("run-2", inputs, dossier, LATER)

    assert [first.sequence, second.sequence] == [0, 1]
    assert [first.previous_hash, second.previous_hash] == [GENESIS, first.hash]
    assert ledger.records() == [first, second]
    assert ledger.head == second.hash
    assert ledger.verify() == []


def test_records_the_engine_that_produced_the_advice(
    ledger: Ledger, inputs: AdviceInputs, dossier: AdviceDossier
) -> None:
    entry = ledger.record_advice("run-1", inputs, dossier, AT).entry

    assert isinstance(entry, AdviceEntry)
    assert entry.engine_version == version("wealth-advisor")


def test_writes_one_canonical_ascii_line_per_record(approved: Ledger) -> None:
    lines = approved.path.read_text(encoding="ascii").splitlines()

    assert len(lines) == 2
    for line in lines:
        assert line == json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"))


def test_records_an_adviser_decision(approved: Ledger) -> None:
    advice, decision = approved.records()

    assert decision.entry == DecisionEntry(
        advice_hash=advice.hash, decision=Decision.APPROVED, adviser_id="adviser-7"
    )
    assert approved.verify() == []


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        pytest.param(
            lambda lines: [lines[0].replace('"quantity":"429"', '"quantity":"430"'), lines[1]],
            "line 1 has been altered",
            id="edited",
        ),
        pytest.param(
            lambda lines: [lines[0], lines[1].replace("10:05:00Z", "10:04:00Z")],
            "line 2 has been altered",
            id="redated",
        ),
        pytest.param(
            lambda lines: lines[1:],
            "line 1 breaks the chain: a record before it was removed, added or moved",
            id="removed",
        ),
        pytest.param(
            lambda lines: lines[::-1],
            "line 1 breaks the chain: a record before it was removed, added or moved",
            id="reordered",
        ),
        pytest.param(
            lambda lines: [lines[0], "not json"], "line 2 is not a ledger record", id="junk"
        ),
        pytest.param(
            lambda lines: [lines[0], "[1, 2]"], "line 2 is not a ledger record", id="list"
        ),
        pytest.param(lambda lines: [lines[0], "5"], "line 2 is not a ledger record", id="number"),
        pytest.param(
            lambda lines: [lines[0], '{"sequence": 1}'],
            "line 2 is not a ledger record",
            id="no-hash",
        ),
    ],
)
def test_detects_any_change_to_the_records(
    approved: Ledger, change: Callable[[list[str]], list[str]], problem: str
) -> None:
    rewrite(approved, change)

    assert approved.verify() == [problem]


def test_detects_a_record_spliced_in_from_another_ledger(
    approved: Ledger, tmp_path: Path, inputs: AdviceInputs, dossier: AdviceDossier
) -> None:
    other = Ledger(tmp_path / "other.jsonl")
    advice = other.record_advice("run-9", inputs, dossier, AT)
    other.record_decision(advice.hash, Decision.APPROVED, "adviser-7", LATER)
    spliced = other.path.read_text(encoding="ascii").splitlines()[1]
    rewrite(approved, lambda lines: [lines[0], spliced])

    assert approved.verify() == [
        "line 2 breaks the chain: a record before it was removed, added or moved"
    ]


def test_detects_a_forged_record_with_the_wrong_sequence(ledger: Ledger) -> None:
    seal(ledger, {"sequence": 5, "recorded_at": AT.isoformat(), "entry": {}})

    assert ledger.verify() == [
        "line 1 breaks the chain: a record before it was removed, added or moved"
    ]


def test_keeps_text_in_any_language_as_plain_ascii(
    ledger: Ledger, inputs: AdviceInputs, dossier: AdviceDossier
) -> None:
    notes = f"{DEMO_IPS}Notes: the client\u2019s adviser is Jos\u00e9; fees are quoted in \u20b9.\n"
    ledger.record_advice("run-1", inputs.model_copy(update={"ips_text": notes}), dossier, AT)

    entry = ledger.records()[0].entry
    assert ledger.path.read_bytes().isascii()
    assert ledger.verify() == []
    assert isinstance(entry, AdviceEntry)
    assert entry.inputs.ips_text == notes


def test_a_ledger_cut_short_no_longer_ends_at_its_published_head(approved: Ledger) -> None:
    published = approved.head
    rewrite(approved, lambda lines: lines[:1])
    remaining = approved.head

    assert approved.verify() == []  # the chain alone cannot tell; the published head can
    assert approved.verify(published_head=published) == [
        f"the ledger ends at {remaining[:12]}, not at the published head {published[:12]}"
    ]


def test_refuses_to_add_to_a_ledger_that_fails_verification(
    approved: Ledger, inputs: AdviceInputs, dossier: AdviceDossier
) -> None:
    rewrite(approved, lambda lines: lines[1:])

    with pytest.raises(LedgerError, match="line 1 breaks the chain"):
        approved.record_advice("run-2", inputs, dossier, LATER)


def test_refuses_to_record_advice_that_does_not_replay(
    ledger: Ledger, inputs: AdviceInputs, dossier: AdviceDossier
) -> None:
    altered_tax = dossier.tax.model_copy(update={"estimated_tax": Decimal("1.00")})

    with pytest.raises(LedgerError) as error:
        ledger.record_advice("run-1", inputs, dossier.model_copy(update={"tax": altered_tax}), AT)
    assert error.value.problems == ["tax: the recorded tax differs from a fresh calculation"]
    assert ledger.records() == []


def test_refuses_to_record_the_same_run_twice(
    ledger: Ledger, inputs: AdviceInputs, dossier: AdviceDossier
) -> None:
    ledger.record_advice("run-1", inputs, dossier, AT)

    with pytest.raises(LedgerError, match="record 1 repeats run run-1"):
        ledger.record_advice("run-1", inputs, dossier, LATER)


def test_decides_on_each_advice_run_once(approved: Ledger) -> None:
    advice = approved.records()[0]

    with pytest.raises(LedgerError, match=f"record 2 decides again on advice {advice.hash[:12]}"):
        approved.record_decision(advice.hash, Decision.DECLINED, "adviser-8", LATER)


def test_decides_only_on_advice_already_in_the_ledger(ledger: Ledger) -> None:
    with pytest.raises(LedgerError, match="record 0 decides on advice that is not earlier"):
        ledger.record_decision("a" * 64, Decision.DECLINED, "adviser-7", AT)


def test_refuses_a_record_dated_before_the_one_it_follows(
    ledger: Ledger, inputs: AdviceInputs, dossier: AdviceDossier
) -> None:
    advice = ledger.record_advice("run-1", inputs, dossier, LATER)

    with pytest.raises(LedgerError, match="record 1 is dated before the record it follows"):
        ledger.record_decision(advice.hash, Decision.DECLINED, "adviser-7", AT)


def test_approving_re_runs_the_replay_but_declining_does_not(
    ledger: Ledger, inputs: AdviceInputs, dossier: AdviceDossier, monkeypatch: pytest.MonkeyPatch
) -> None:
    advice = ledger.record_advice("run-1", inputs, dossier, AT)
    # As if the rebalancer changed between recording the advice and the adviser's review.
    monkeypatch.setattr(ledger_module, "propose_rebalance", no_trades(dossier.proposal))

    with pytest.raises(LedgerError, match="the advice no longer re-verifies: proposal: "):
        ledger.record_decision(advice.hash, Decision.APPROVED, "adviser-7", LATER)
    declined = ledger.record_decision(advice.hash, Decision.DECLINED, "adviser-7", LATER, "Stale.")
    assert declined.entry == DecisionEntry(
        advice_hash=advice.hash, decision=Decision.DECLINED, adviser_id="adviser-7", note="Stale."
    )


def test_verify_replays_every_advice_run(
    ledger: Ledger, inputs: AdviceInputs, dossier: AdviceDossier, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger.record_advice("run-1", inputs, dossier, AT)
    monkeypatch.setattr(ledger_module, "propose_rebalance", no_trades(dossier.proposal))

    assert "record 0: proposal: the recorded proposal differs from a fresh calculation" in (
        ledger.verify()
    )


def test_verify_catches_rule_breaks_in_a_forged_but_consistent_chain(
    ledger: Ledger, inputs: AdviceInputs, dossier: AdviceDossier
) -> None:
    violation = Violation(rule=Rule.CASH_RESERVE, detail="cash would be $0.00 after the trades")
    blocked = dossier.model_copy(
        update={"suitability": dossier.suitability.model_copy(update={"violations": (violation,)})}
    )
    advice = forge(ledger, AdviceEntry(run_id="run-1", inputs=inputs, dossier=blocked), LATER)
    forge(ledger, AdviceEntry(run_id="run-1", inputs=inputs, dossier=dossier), AT)
    forge(ledger, DecisionEntry(advice_hash=advice, decision=Decision.APPROVED, adviser_id="a"))
    forge(ledger, DecisionEntry(advice_hash=advice, decision=Decision.DECLINED, adviser_id="a"))
    forge(ledger, DecisionEntry(advice_hash="b" * 64, decision=Decision.DECLINED, adviser_id="a"))

    assert ledger.verify() == [
        "record 1 is dated before the record it follows",
        "record 1 repeats run run-1",
        "record 2 approves advice the suitability gate blocked",
        f"record 3 decides again on advice {advice[:12]}",
        "record 4 decides on advice that is not earlier in the ledger",
        "record 0: suitability: the recorded suitability differs from a fresh calculation",
    ]


def test_a_consistent_line_that_is_not_a_record_fails_verification(ledger: Ledger) -> None:
    seal(ledger, {"recorded_at": AT.isoformat(), "entry": {"kind": "rumour"}})

    assert ledger.verify() == ["line 1 is not a ledger record"]


def test_replay_rechecks_the_ips_quotes(inputs: AdviceInputs, dossier: AdviceDossier) -> None:
    quotes = dossier.ips_quotes | {"risk_tolerance": "Risk tolerance: 7"}
    entry = AdviceEntry(
        run_id="run-1", inputs=inputs, dossier=dossier.model_copy(update={"ips_quotes": quotes})
    )

    assert replay(entry) == [
        "IPS: risk_tolerance: the quote is not in the IPS: 'Risk tolerance: 7'"
    ]


def test_replay_rechecks_the_briefing(inputs: AdviceInputs, dossier: AdviceDossier) -> None:
    briefing = f"{dossier.briefing}Fees come to $99.\n"
    entry = AdviceEntry(
        run_id="run-1", inputs=inputs, dossier=dossier.model_copy(update={"briefing": briefing})
    )

    assert replay(entry) == ["briefing: uses numbers that are not in the facts: 99"]


def test_replay_reports_inputs_that_no_longer_produce_advice(
    inputs: AdviceInputs, dossier: AdviceDossier
) -> None:
    other_client = inputs.model_copy(
        update={"portfolio": inputs.portfolio.model_copy(update={"client_id": "C-2002"})}
    )

    assert replay(AdviceEntry(run_id="run-1", inputs=other_client, dossier=dossier)) == [
        "the recorded inputs no longer produce advice: "
        "profile is for C-1001, portfolio is for C-2002"
    ]
