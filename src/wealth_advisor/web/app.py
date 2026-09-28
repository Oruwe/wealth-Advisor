"""The adviser console: review each advice run, approve or decline it, and audit the ledger.

Server-rendered pages, no JavaScript. The ledger, the clock and the advice runner are passed
in, so the console never talks to Lyzr itself and tests can run it with fakes.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NamedTuple
from uuid import uuid4

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from wealth_advisor.advisor import AdviceDossier, AgentProvenance
from wealth_advisor.ledger import (
    AdviceEntry,
    AdviceInputs,
    Decision,
    Ledger,
    LedgerError,
    LedgerRecord,
)
from wealth_advisor.web import views

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")


class AdviceRun(NamedTuple):
    inputs: AdviceInputs
    dossier: AdviceDossier
    provenance: AgentProvenance | None


class AdviceRunError(RuntimeError):
    """An advice run stopped before it produced advice worth recording."""


# Produces an advice run for a run ID, which is also its Lyzr session.
Runner = Callable[[str], AdviceRun]


def create_app(
    ledger: Ledger,
    runner: Runner | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> FastAPI:
    app = FastAPI(title="Wealth Advisor console", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> Response:
        return _dashboard(request, ledger, can_run=runner is not None)

    @app.post("/runs")
    def new_run(request: Request) -> Response:
        if runner is None:
            error = "Set LYZR_API_KEY in .env and restart the console to run new advice."
            return _dashboard(request, ledger, can_run=False, error=error, status_code=503)
        run_id = uuid4().hex
        try:
            run = runner(run_id)
            record = ledger.record_advice(run_id, run.inputs, run.dossier, clock(), run.provenance)
        except (AdviceRunError, LedgerError) as failure:
            error = f"The run was stopped and nothing was recorded: {failure}"
            return _dashboard(request, ledger, can_run=True, error=error, status_code=502)
        return RedirectResponse(f"/advice/{record.hash}", status_code=303)

    @app.get("/advice/{advice_hash}", response_class=HTMLResponse)
    def advice(request: Request, advice_hash: str) -> Response:
        return _dossier(request, ledger, advice_hash)

    @app.post("/advice/{advice_hash}/decision")
    def decide(
        request: Request,
        advice_hash: str,
        decision: Annotated[Decision, Form()],
        adviser: Annotated[str, Form()] = "",
        note: Annotated[str, Form()] = "",
    ) -> Response:
        if not adviser.strip():
            error = "Enter your name so the ledger records who made the decision."
            return _dossier(request, ledger, advice_hash, error=error, status_code=422)
        try:
            ledger.record_decision(advice_hash, decision, adviser.strip(), clock(), note.strip())
        except LedgerError as refusal:
            error = f"The ledger refused the decision: {refusal}"
            return _dossier(request, ledger, advice_hash, error=error, status_code=409)
        return RedirectResponse(f"/advice/{advice_hash}", status_code=303)

    @app.get("/audit", response_class=HTMLResponse)
    def audit(request: Request, head: str = "") -> Response:
        problems = ledger.verify(published_head=head.strip() or None)
        records = _records(ledger)
        return templates.TemplateResponse(
            request,
            "audit.html",
            {
                "problems": problems,
                "count": len(records) if records is not None else None,
                "head": records[-1].hash if records else None,
                "published_head": head.strip(),
            },
        )

    return app


def _records(ledger: Ledger) -> list[LedgerRecord] | None:
    """The ledger's records, or None when the ledger is too damaged to read."""
    try:
        return ledger.records()
    except ValueError:
        return None


def _dashboard(
    request: Request, ledger: Ledger, can_run: bool, error: str = "", status_code: int = 200
) -> Response:
    records = _records(ledger)
    intact = records is not None and not ledger.verify()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "runs": views.summaries(records) if records is not None else [],
            "damaged": records is None,
            "intact": intact,
            "count": len(records) if records is not None else 0,
            "can_run": can_run,
            "error": error,
        },
        status_code=status_code,
    )


def _dossier(
    request: Request, ledger: Ledger, advice_hash: str, error: str = "", status_code: int = 200
) -> Response:
    records = _records(ledger)
    if records is None:
        return templates.TemplateResponse(request, "damaged.html", {}, status_code=409)
    record = next(
        (r for r in records if r.hash == advice_hash and isinstance(r.entry, AdviceEntry)), None
    )
    if record is None:
        return templates.TemplateResponse(
            request, "missing.html", {"advice_hash": advice_hash}, status_code=404
        )
    return templates.TemplateResponse(
        request,
        "dossier.html",
        {"view": views.dossier(record, records), "error": error},
        status_code=status_code,
    )
