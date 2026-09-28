from datetime import UTC, datetime
from pathlib import Path

import pytest
from fakes import GROUNDED_BRIEFING, FakeReader, FakeStudio, FakeWriter, faithful_reading
from fastapi.testclient import TestClient
from lyzr import AgentResponse, Studio
from lyzr.exceptions import APIError
from pydantic import SecretStr

from wealth_advisor import ledger as ledger_module
from wealth_advisor.advisor import Agents, advise
from wealth_advisor.demo import DEMO_IPS, demo_portfolio, demo_prices
from wealth_advisor.domain.suitability import Rule, Violation
from wealth_advisor.ledger import (
    AdviceEntry,
    AdviceInputs,
    Decision,
    DecisionEntry,
    Ledger,
    LedgerRecord,
)
from wealth_advisor.settings import Settings
from wealth_advisor.web import views
from wealth_advisor.web.app import AdviceRun, AdviceRunError, create_app
from wealth_advisor.web.runner import lyzr_demo_runner

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
INPUTS = AdviceInputs(ips_text=DEMO_IPS, portfolio=demo_portfolio(), prices=demo_prices())


def fake_run(briefing: str = GROUNDED_BRIEFING) -> AdviceRun:
    agents = Agents(reader=FakeReader(faithful_reading()), writer=FakeWriter(briefing))
    return AdviceRun(INPUTS, advise(DEMO_IPS, demo_portfolio(), demo_prices(), agents), None)


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl")


@pytest.fixture
def client(ledger: Ledger) -> TestClient:
    return TestClient(create_app(ledger, lambda _: fake_run(), clock=lambda: NOW))


def record_run(ledger: Ledger, run_id: str = "run-1") -> LedgerRecord:
    run = fake_run()
    return ledger.record_advice(run_id, run.inputs, run.dossier, NOW, run.provenance)


def test_an_empty_ledger_has_no_runs_yet(client: TestClient) -> None:
    page = client.get("/")

    assert page.status_code == 200
    assert "Ledger intact · 0 records" in page.text
    assert "No advice runs recorded yet." in page.text


def test_a_new_run_is_recorded_and_opens_its_dossier(client: TestClient, ledger: Ledger) -> None:
    response = client.post("/runs", follow_redirects=False)

    (record,) = ledger.records()
    assert isinstance(record.entry, AdviceEntry)
    assert len(record.entry.run_id) == 32
    assert (response.status_code, response.headers["location"]) == (303, f"/advice/{record.hash}")
    assert "Rebalance proposal for C-1001" in client.get(response.headers["location"]).text


def test_without_a_runner_the_console_explains_how_to_run_advice(ledger: Ledger) -> None:
    client = TestClient(create_app(ledger))

    response = client.post("/runs")

    assert response.status_code == 503
    assert "Set <code>LYZR_API_KEY</code>" in client.get("/").text
    assert "Set LYZR_API_KEY in .env and restart the console" in response.text
    assert ledger.records() == []


def test_a_failed_run_records_nothing(ledger: Ledger) -> None:
    def failing(run_id: str) -> AdviceRun:
        raise AdviceRunError("risk_tolerance: the quote is not in the IPS")

    response = TestClient(create_app(ledger, failing)).post("/runs")

    assert response.status_code == 502
    assert "nothing was recorded: risk_tolerance: the quote is not in the IPS" in response.text
    assert ledger.records() == []


def test_the_dashboard_lists_each_run_with_its_gate_verdict_and_decision(
    client: TestClient, ledger: Ledger
) -> None:
    decided, pending = record_run(ledger, "run-1"), record_run(ledger, "run-2")
    ledger.record_decision(decided.hash, Decision.APPROVED, "Triven", NOW)

    page = client.get("/").text

    assert "Ledger intact · 3 records" in page
    assert page.index(f"/advice/{pending.hash}") < page.index(f"/advice/{decided.hash}")
    assert page.count("Passed</span>") == 2
    assert "Approved</span>" in page
    assert "Awaiting decision</span>" in page


def test_the_dossier_shows_what_the_adviser_reviews(client: TestClient, ledger: Ledger) -> None:
    record = record_run(ledger)

    page = client.get(f"/advice/{record.hash}").text

    for shown in (
        "Passed the suitability gate",
        "Replay re-verified now",
        "Awaiting decision",
        "$100,000.00",
        "$345.00",
        "$10,170.00",
        "Not in the model portfolio",
        "$30,030.00",
        "19.80%",
        "65.03%",
        "Loss of $2,300.00",
        "$2,300.00 of net long-term gain at 15.00%",
        "Risk tolerance: 3 on a scale of 1 to 10.",
        "the client is in the 24% federal income-tax bracket",
        "the $2,300 short-term loss offsets half of the $4,600 long-term gain",
        record.hash,
        'value="approved"',
    ):
        assert shown in page


def test_approving_records_the_decision_and_closes_the_form(
    client: TestClient, ledger: Ledger
) -> None:
    record = record_run(ledger)
    form = {"decision": "approved", "adviser": " Triven ", "note": " Reviewed with client "}

    response = client.post(f"/advice/{record.hash}/decision", data=form, follow_redirects=False)

    assert (response.status_code, response.headers["location"]) == (303, f"/advice/{record.hash}")
    assert ledger.records()[-1].entry == DecisionEntry(
        advice_hash=record.hash,
        decision=Decision.APPROVED,
        adviser_id="Triven",
        note="Reviewed with client",
    )
    page = client.get(f"/advice/{record.hash}").text
    assert "by Triven on 2026-09-28 12:00 UTC: Reviewed with client" in page
    assert 'name="decision"' not in page


def test_declining_is_recorded_too(client: TestClient, ledger: Ledger) -> None:
    record = record_run(ledger)

    client.post(f"/advice/{record.hash}/decision", data={"decision": "declined", "adviser": "A"})

    assert "Declined</span>" in client.get(f"/advice/{record.hash}").text


def test_a_decision_needs_the_advisers_name(client: TestClient, ledger: Ledger) -> None:
    record = record_run(ledger)

    response = client.post(
        f"/advice/{record.hash}/decision", data={"decision": "approved", "adviser": "  "}
    )

    assert response.status_code == 422
    assert "Enter your name so the ledger records who made the decision." in response.text
    assert len(ledger.records()) == 1


def test_the_ledger_refuses_a_second_decision(client: TestClient, ledger: Ledger) -> None:
    record = record_run(ledger)
    decide = {"decision": "approved", "adviser": "A"}
    client.post(f"/advice/{record.hash}/decision", data=decide)

    response = client.post(f"/advice/{record.hash}/decision", data=decide)

    assert response.status_code == 409
    assert "The ledger refused the decision: record 2 decides again" in response.text


@pytest.mark.parametrize("which", ["unknown", "decision"])
def test_only_advice_records_have_a_dossier(client: TestClient, ledger: Ledger, which: str) -> None:
    advice = record_run(ledger)
    decision = ledger.record_decision(advice.hash, Decision.DECLINED, "A", NOW)
    missing = "f" * 64 if which == "unknown" else decision.hash

    response = client.get(f"/advice/{missing}")

    assert response.status_code == 404
    assert "No advice run with that hash" in response.text


def test_a_tampered_ledger_is_flagged_and_refuses_decisions(
    client: TestClient, ledger: Ledger
) -> None:
    record = record_run(ledger)
    text = ledger.path.read_text(encoding="ascii")
    ledger.path.write_text(text.replace('"quantity":"429"', '"quantity":"430"'), encoding="ascii")

    dashboard = client.get("/").text
    dossier = client.get(f"/advice/{record.hash}").text
    refused = client.post(
        f"/advice/{record.hash}/decision", data={"decision": "approved", "adviser": "A"}
    )

    assert "Ledger fails verification" in dashboard
    assert "Replay failed" in dossier
    assert refused.status_code == 409
    assert "line 1 has been altered" in refused.text


def test_an_unreadable_ledger_shows_no_dossier(client: TestClient, ledger: Ledger) -> None:
    record = record_run(ledger)
    with ledger.path.open("a", encoding="ascii") as file:
        file.write("not a record\n")

    assert "The ledger can't be read" in client.get("/").text
    response = client.get(f"/advice/{record.hash}")
    assert response.status_code == 409
    assert "no dossier can be shown or decided" in response.text


def test_the_audit_page_verifies_the_ledger_and_a_published_head(
    client: TestClient, ledger: Ledger
) -> None:
    record = record_run(ledger)

    intact = client.get("/audit").text
    wrong_head = client.get("/audit", params={"head": "0" * 64}).text

    assert "Everything checks out" in intact
    assert record.hash in intact
    assert f"the ledger ends at {record.hash[:12]}, not at the published head 000000000000" in (
        wrong_head
    )


def test_blocked_advice_can_only_be_declined(client: TestClient, ledger: Ledger) -> None:
    run = fake_run()
    violation = Violation(rule=Rule.MAX_WEIGHT, detail="equity would be 55.00% of the portfolio")
    blocked = run.dossier.model_copy(
        update={
            "suitability": run.dossier.suitability.model_copy(update={"violations": (violation,)})
        }
    )
    entry = AdviceEntry(run_id="run-1", inputs=run.inputs, dossier=blocked)
    fields = {
        "sequence": 0,
        "previous_hash": ledger_module.GENESIS,
        "recorded_at": NOW.isoformat(),
        "entry": entry.model_dump(mode="json"),
    }
    digest = ledger_module._digest(fields)
    ledger.path.write_text(ledger_module._canonical({**fields, "hash": digest}) + "\n")

    page = client.get(f"/advice/{digest}").text

    assert "Blocked by the suitability gate" in page
    assert "max_weight: equity would be 55.00% of the portfolio" in page
    assert 'value="approved"' not in page
    assert 'value="declined"' in page


def test_text_from_the_ips_and_the_agents_is_escaped(ledger: Ledger) -> None:
    run = fake_run(briefing="Verdict: approved. <script>alert('boo')</script>")
    record = ledger.record_advice("run-1", run.inputs, run.dossier, NOW)

    page = TestClient(create_app(ledger)).get(f"/advice/{record.hash}").text

    assert "<script>" not in page
    assert "&lt;script&gt;alert(&#39;boo&#39;)&lt;/script&gt;" in page


def test_the_chart_places_every_weight_on_one_axis(ledger: Ledger) -> None:
    record = record_run(ledger)

    view = views.dossier(record, ledger.records())

    assert [(row.label, row.before, row.after, row.target, row.cap) for row in view.allocation] == [
        ("Equity", "55.00%", "19.80%", "20.00%", "20.00%"),
        ("Fixed income", "35.00%", "65.03%", "65.00%", "none"),
        ("Commodities", "5.00%", "5.00%", "5.00%", "10.00%"),
        ("Cash", "5.00%", "10.17%", "10.00%", "none"),
    ]
    assert [tick.label for tick in view.ticks] == [f"{step}%" for step in range(0, 80, 10)]
    assert (view.ticks[0].x, view.ticks[-1].x) == (views.PLOT_LEFT, views.PLOT_RIGHT)
    assert [row.label_anchor for row in view.allocation] == ["start", "start", "end", "start"]


def test_an_empty_account_charts_its_weights_at_zero(ledger: Ledger) -> None:
    record = record_run(ledger)
    empty = INPUTS.portfolio.model_copy(update={"holdings": (), "cash": 0})
    entry = record.entry.model_copy(
        update={"inputs": INPUTS.model_copy(update={"portfolio": empty})}
    )

    view = views.dossier(record.model_copy(update={"entry": entry}), [])

    assert [row.before for row in view.allocation] == ["0.00%"] * 4
    assert view.portfolio_value_before == "$0.00"


def test_the_dossier_view_is_only_for_advice(ledger: Ledger) -> None:
    advice = record_run(ledger)
    decision = ledger.record_decision(advice.hash, Decision.APPROVED, "A", NOW)

    with pytest.raises(ValueError, match="record 1 is not an advice run"):
        views.dossier(decision, ledger.records())


def test_the_lyzr_runner_advises_the_demo_client_in_the_runs_session() -> None:
    studio = FakeStudio(
        replies={
            "wealth-advisor-ips-reader": faithful_reading(),
            "wealth-advisor-briefing-writer": AgentResponse(
                response=GROUNDED_BRIEFING, session_id="s"
            ),
        }
    )
    runner = lyzr_demo_runner(_settings(), studio.open)

    run = runner("run-9")

    assert run.inputs == INPUTS
    assert run.dossier.suitability.approved
    assert run.provenance is not None
    assert run.provenance.session_id == "run-9"


def test_the_lyzr_runner_explains_a_connection_failure() -> None:
    def unreachable(**_: object) -> Studio:
        raise APIError("HTTP error occurred: [Errno 11001] getaddrinfo failed")

    runner = lyzr_demo_runner(_settings(), unreachable)

    with pytest.raises(AdviceRunError, match=r"could not complete the Lyzr calls .*getaddrinfo"):
        runner("run-9")


def test_the_lyzr_runner_passes_on_a_grounding_failure() -> None:
    studio = FakeStudio(replies={"wealth-advisor-ips-reader": faithful_reading(risk_tolerance=10)})

    with pytest.raises(AdviceRunError, match="the first number in the quote is 3, not 10"):
        lyzr_demo_runner(_settings(), studio.open)("run-9")


def _settings() -> Settings:
    return Settings(lyzr_api_key=SecretStr("sk-test-not-a-real-key"), lyzr_model="openai/gpt-4.1")
