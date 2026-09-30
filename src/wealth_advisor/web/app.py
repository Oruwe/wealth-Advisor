"""The adviser console: review each advice run, approve or decline it, and audit the ledger.

Server-rendered pages, no JavaScript. The ledger, the clock and the advice runner are passed
in, so the console never talks to Lyzr itself and tests can run it with fakes.
"""

import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, NamedTuple
from uuid import uuid4

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from wealth_advisor.advisor import AdviceDossier, AgentProvenance
from wealth_advisor.engine.broker import AlpacaBrokerAdapter
from wealth_advisor.engine.database import (
    AlertModel,
    ClientModel,
    SessionLocal,
    TaxLotModel,
    UserModel,
    init_db,
)
from wealth_advisor.engine.drift import check_firm_wide_drift
from wealth_advisor.web.auth import RequireRole, create_access_token, get_current_user, verify_password

try:
    from apscheduler.schedulers.background import BackgroundScheduler as _BackgroundScheduler
    _APSCHEDULER_AVAILABLE = True
except ImportError:
    _APSCHEDULER_AVAILABLE = False
from wealth_advisor.engine.ledger import AuditBlock, FiduciaryLedger
from wealth_advisor.policy.suitability import SUITABILITY_POLICY as _SUITABILITY_POLICY
from wealth_advisor.engine.market_data import MarketDataFeed, RiskEstimator
from wealth_advisor.engine.pipeline import FiduciaryPipeline
from wealth_advisor.engine.ingestion import parse_brokerage_csv
from wealth_advisor.engine.tax import TaxJurisdiction, TaxLot, Trade
from wealth_advisor.formatting import dollars
from wealth_advisor.ledger import (
    AdviceEntry,
    AdviceInputs,
    Decision,
    Ledger,
    LedgerError,
    LedgerRecord,
)
from wealth_advisor.reporting.pdf_generator import generate_fiduciary_pdf
from wealth_advisor.web import api, views

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")
_logger = logging.getLogger(__name__)

_ETF_UNIVERSE: list[str] = ["VOO", "VEA", "VWO", "AGG", "TLT", "GLD"]
_EQUITY_TICKERS: set[str] = {"VOO", "VEA", "VWO"}

_MOCK_PORTFOLIO: list[TaxLot] = [
    TaxLot("VOO", Decimal("50"), Decimal("22000.00"), date(2022, 3, 15)),
    TaxLot("VEA", Decimal("300"), Decimal("13500.00"), date(2023, 1, 10)),
    TaxLot("AGG", Decimal("400"), Decimal("46000.00"), date(2021, 6, 20)),
    TaxLot("GLD", Decimal("200"), Decimal("37000.00"), date(2024, 1, 1)),
]

# Populated on first pipeline run; used for portfolio display page.
_cached_prices: dict[str, Decimal] = {}
# Keyed by advice hash; stores the live market snapshot used for each run.
_market_snapshots: dict[str, dict[str, Any]] = {}

_fiduciary_pipeline: FiduciaryPipeline | None = None
_fiduciary_ledger: FiduciaryLedger | None = None
_ai_results: dict[str, dict[str, Any]] = {}
_broker_results: dict[str, dict[str, Any]] = {}
_audit_blocks: dict[str, AuditBlock] = {}
_alpaca_api_key = os.environ.get("ALPACA_API_KEY", "")
_alpaca_secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
_alpaca_live = os.environ.get("ALPACA_LIVE_MODE", "false").lower() == "true"
_broker = AlpacaBrokerAdapter(
    dry_run=not (_alpaca_live and bool(_alpaca_api_key) and bool(_alpaca_secret_key)),
    api_key=_alpaca_api_key or "mock",
    secret_key=_alpaca_secret_key or "mock",
)


_AS_OF = date(2026, 9, 29)  # mock reference date for display-only calculations


def _get_fiduciary_pipeline() -> FiduciaryPipeline:
    global _fiduciary_pipeline
    if _fiduciary_pipeline is None:
        _fiduciary_pipeline = FiduciaryPipeline()
    return _fiduciary_pipeline


def _get_fiduciary_ledger() -> FiduciaryLedger:
    global _fiduciary_ledger
    if _fiduciary_ledger is None:
        _fiduciary_ledger = FiduciaryLedger()
    return _fiduciary_ledger


def _record_fiduciary_block(
    advice_hash: str,
    client_id: str,
    client_snapshot: dict[str, Any],
    ai_result: dict[str, Any],
    broker_receipt: dict[str, Any] | None = None,
    market_snapshot: dict[str, Any] | None = None,
) -> None:
    """Seal one advice cycle into the fiduciary ledger; log and swallow on failure."""
    try:
        fl = _get_fiduciary_ledger()
        prev = fl.get_latest_block()
        prev_hash = prev.signature_hash if prev else FiduciaryLedger.GENESIS
        trades: list[dict[str, Any]] = [
            {
                "symbol": t.symbol,
                "action": t.action.value,
                "shares": str(t.shares),
                "estimated_tax_impact": str(t.estimated_tax_impact),
            }
            for t in ai_result.get("trades", [])
            if isinstance(t, Trade)
        ]
        block = fl.record_decision(
            previous_hash=prev_hash,
            client_id=client_id,
            client_snapshot=client_snapshot,
            market_snapshot=market_snapshot or {},
            ai_mandate={
                "risk_parameters": ai_result["risk_parameters"],
                "event_type": ai_result.get("event_type", "unknown"),
            },
            optimization_result={
                "target_weights": ai_result["target_weights"],
                "solver_status": "optimal",
                "timestamp": ai_result.get("timestamp", ""),
            },
            trades=trades,
            broker_receipt=broker_receipt,
            policy_hash=_SUITABILITY_POLICY.policy_hash,
        )
        _audit_blocks[advice_hash] = block
    except Exception as exc:
        _logger.warning("fiduciary ledger recording failed for %s: %s", advice_hash, exc)


def _mock_client(client_id: str) -> dict[str, Any]:
    if client_id == "C-1049":
        return {
            "client_id": client_id,
            "name": "Arjun Mehta",
            "tax_jurisdiction": "INDIA",
            "lrs_quota_used_usd": 180_000,
            "lrs_quota_cap_usd": 250_000,
            "lrs_pct": round(180_000 / 250_000 * 100, 1),
            "risk_tolerance": 5,
            "risk_band": "Moderate",
            "time_horizon_years": 15,
            "marginal_tax_rate_pct": 30,
            "annual_income_usd": dollars(Decimal("200000")),
            "net_worth_usd": dollars(Decimal("850000")),
            "onboarded": "2021-07-15",
        }
    return {
        "client_id": client_id,
        "name": "Triven Oruwe" if client_id == "C-1001" else client_id,
        "tax_jurisdiction": "US",
        "lrs_quota_used_usd": 48_000,
        "lrs_quota_cap_usd": 250_000,
        "lrs_pct": round(48_000 / 250_000 * 100, 1),
        "risk_tolerance": 3,
        "risk_band": "Conservative",
        "time_horizon_years": 10,
        "marginal_tax_rate_pct": 24,
        "annual_income_usd": dollars(Decimal("150000")),
        "net_worth_usd": dollars(Decimal("500000")),
        "onboarded": "2022-03-01",
    }


def _portfolio_prices() -> dict[str, Decimal]:
    """Return the most recently cached live prices, fetching if the cache is empty."""
    if _cached_prices:
        return _cached_prices
    syms = list({lot.symbol for lot in _MOCK_PORTFOLIO})
    try:
        df = MarketDataFeed.get_historical_prices(syms, years=1)
        return {str(col): Decimal(str(round(float(df[col].iloc[-1]), 4))) for col in df.columns}
    except Exception as exc:
        _logger.warning("price fetch for portfolio view failed: %s", exc)
        return {}


def _mock_portfolio_view(client_id: str, lots: list[TaxLot] | None = None) -> dict[str, Any]:
    portfolio = lots if lots is not None else _MOCK_PORTFOLIO
    prices = _portfolio_prices()
    holdings: dict[str, dict[str, Any]] = {}
    for lot in portfolio:
        sym = lot.symbol
        price = prices.get(sym, Decimal("0"))
        cur_val = lot.shares * price
        if sym not in holdings:
            holdings[sym] = {"value": Decimal("0"), "cost_basis": Decimal("0"), "lots": []}
        holdings[sym]["value"] += cur_val
        holdings[sym]["cost_basis"] += lot.cost_basis

    total_value = sum(h["value"] for h in holdings.values())
    chart_labels = list(holdings.keys())
    chart_values = [float(h["value"]) for h in holdings.values()]

    lots_rows = []
    for lot in portfolio:
        price = prices.get(lot.symbol, Decimal("0"))
        cur_val = lot.shares * price
        gain = cur_val - lot.cost_basis
        days_held = (_AS_OF - lot.purchase_date).days
        term = "Long-term" if days_held >= 365 else "Short-term"
        gain_pct = f"{float(gain / lot.cost_basis * 100):+.1f}%" if lot.cost_basis else "—"
        lots_rows.append({
            "symbol": lot.symbol,
            "shares": format(lot.shares, "f"),
            "purchase_date": lot.purchase_date.isoformat(),
            "cost_basis": dollars(lot.cost_basis),
            "current_value": dollars(cur_val),
            "gain": dollars(gain),
            "gain_pct": gain_pct,
            "term": term,
            "is_gain": gain >= 0,
        })

    return {
        "client_id": client_id,
        "total_value": dollars(total_value),
        "chart_labels": chart_labels,
        "chart_values": chart_values,
        "holdings": [
            {
                "symbol": sym,
                "value": dollars(data["value"]),
                "cost_basis": dollars(data["cost_basis"]),
                "pct": (
                    f"{float(data['value'] / total_value * 100):.1f}%"
                    if total_value
                    else "0.0%"
                ),
            }
            for sym, data in holdings.items()
        ],
        "lots": lots_rows,
    }


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        init_db()
    except Exception as exc:
        _logger.warning("DB init failed (is PostgreSQL running?): %s", exc)

    scheduler = None
    if _APSCHEDULER_AVAILABLE:
        try:
            scheduler = _BackgroundScheduler()
            scheduler.add_job(check_firm_wide_drift, "interval", minutes=1)
            scheduler.start()
            _logger.info("Drift monitor scheduler started (interval: 1 min)")
        except Exception as exc:
            _logger.warning("Scheduler failed to start: %s", exc)
            scheduler = None

    yield

    if scheduler is not None:
        try:
            scheduler.shutdown(wait=False)
        except Exception as exc:
            _logger.warning("Scheduler shutdown error: %s", exc)


def _taxlot_from_model(row: TaxLotModel) -> TaxLot:
    """Reconstruct a domain TaxLot from a SQLAlchemy ORM row."""
    try:
        jur = TaxJurisdiction(str(row.asset_jurisdiction))
    except ValueError:
        jur = TaxJurisdiction.US
    return TaxLot(
        symbol=str(row.symbol),
        shares=Decimal(str(row.shares)),
        cost_basis=Decimal(str(row.cost_basis)),
        purchase_date=row.purchase_date,
        asset_jurisdiction=jur,
        investor_jurisdiction=jur,
    )


def _lots_from_db(client_id: str) -> list[TaxLot]:
    """Return a client's tax lots from the DB; empty list on any DB error."""
    try:
        with SessionLocal() as session:
            rows = (
                session.query(TaxLotModel)
                .filter(TaxLotModel.client_id == client_id)
                .all()
            )
        return [_taxlot_from_model(r) for r in rows]
    except Exception as exc:
        _logger.warning("DB query for client %s failed: %s", client_id, exc)
        return []


def _get_alerts(advisor_id: int) -> list[dict[str, Any]]:
    """Return unresolved alerts for an adviser as plain dicts (safe to use after session close)."""
    try:
        with SessionLocal() as session:
            rows = (
                session.query(AlertModel)
                .filter(
                    AlertModel.advisor_id == advisor_id,
                    AlertModel.is_resolved.is_(False),
                )
                .order_by(AlertModel.created_at.desc())
                .all()
            )
            return [
                {
                    "id": r.id,
                    "client_id": str(r.client_id),
                    "alert_type": str(r.alert_type),
                    "message": str(r.message),
                    "created_at": r.created_at,
                }
                for r in rows
            ]
    except Exception as exc:
        _logger.warning("alert query failed for advisor %s: %s", advisor_id, exc)
        return []


def _get_advisor_client_ids(advisor_id: int) -> set[str]:
    """Return the set of client IDs assigned to an adviser; empty set on DB error."""
    try:
        with SessionLocal() as session:
            rows = (
                session.query(ClientModel.id)
                .filter(ClientModel.advisor_id == advisor_id)
                .all()
            )
        return {str(r.id) for r in rows}
    except Exception as exc:
        _logger.warning("advisor client-id query failed: %s", exc)
        return set()


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
    app = FastAPI(
        title="Wealth Advisor console",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )

    _ALLOWED_ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # DNS-rebinding guard: reject requests with an unexpected Host header.
    _ALLOWED_HOSTS = {"localhost:8000", "127.0.0.1:8000", "testserver"}

    @app.middleware("http")
    async def _check_host(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        host = request.headers.get("host", "")
        if host not in _ALLOWED_HOSTS:
            return Response(content=b"Invalid Host header", status_code=400)
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.include_router(api.router)

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request) -> Response:
        return templates.TemplateResponse(request, "login.html", {"error": ""})

    @app.post("/token")
    async def token(
        request: Request,
        form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    ) -> Response:
        with SessionLocal() as db:
            user = db.query(UserModel).filter(
                UserModel.username == form_data.username
            ).first()
        if user is None or not verify_password(form_data.password, str(user.hashed_password)):
            return templates.TemplateResponse(
                request, "login.html",
                {"error": "Invalid username or password."},
                status_code=401,
            )
        jwt_token = create_access_token({"sub": user.username})
        redirect_to = "/portal" if user.role == "CLIENT" else "/"
        resp = RedirectResponse(redirect_to, status_code=303)
        resp.set_cookie(
            key="access_token",
            value=jwt_token,
            httponly=True,
            max_age=28800,
            samesite="lax",
        )
        return resp

    @app.get("/logout")
    def logout() -> Response:
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie("access_token")
        return resp

    @app.get("/portal", response_class=HTMLResponse)
    def portal(
        request: Request,
        current_user: UserModel = Depends(get_current_user),
    ) -> Response:
        if current_user.role != "CLIENT":
            return RedirectResponse("/", status_code=303)
        client_id = str(current_user.linked_client_id or "C-1049")
        lots = _lots_from_db(client_id) or _MOCK_PORTFOLIO
        return templates.TemplateResponse(
            request,
            "client_portal.html",
            {
                "portfolio": _mock_portfolio_view(client_id, lots),
                "client": _mock_client(client_id),
                "current_user": current_user,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    def overview(
        request: Request,
        current_user: UserModel = Depends(get_current_user),
    ) -> Response:
        if current_user.role == "CLIENT":
            return RedirectResponse("/portal", status_code=303)
        return _overview(request, ledger, current_user=current_user)

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request) -> Response:
        return _dashboard(request, ledger, can_run=runner is not None)

    @app.get("/simulate", response_class=HTMLResponse)
    def simulate(
        request: Request,
        current_user: UserModel = Depends(RequireRole(["ADVISOR", "CCO"])),
    ) -> Response:
        return _simulate_page(request, can_run=runner is not None, current_user=current_user)

    @app.post("/runs")
    def new_run(
        request: Request,
        client_id: Annotated[str, Form()] = "C-1001",
        risk_tolerance: Annotated[int, Form()] = 3,
        event_type: Annotated[str, Form()] = "advisor_requested_rebalance",
    ) -> Response:
        if runner is None:
            error = "Set LYZR_API_KEY in .env and restart the console to run new advice."
            return _simulate_page(request, can_run=False, error=error, status_code=503)
        run_id = uuid4().hex
        try:
            run = runner(run_id)
            record = ledger.record_advice(run_id, run.inputs, run.dossier, clock(), run.provenance)
        except (AdviceRunError, LedgerError) as failure:
            error = f"The run was stopped and nothing was recorded: {failure}"
            return _simulate_page(request, can_run=True, error=error, status_code=502)

        client_profile: dict[str, Any] = {
            "client_id": client_id,
            "risk_tolerance": risk_tolerance,
            "time_horizon_years": 10,
            "profile_type": "conservative" if risk_tolerance <= 3 else "moderate",
            "cash_reserve_usd": 2000,
            "tax_jurisdiction": "US",
            "lrs_quota_used_usd": 48000,
        }
        event_payload: dict[str, Any] = {"event_type": event_type, "client_id": client_id}
        try:
            universe = _ETF_UNIVERSE + [
                lot.symbol for lot in _MOCK_PORTFOLIO if lot.symbol not in _ETF_UNIVERSE
            ]
            prices_df = MarketDataFeed.get_historical_prices(universe)
            expected_returns = RiskEstimator.compute_expected_returns(prices_df)
            cov_df = RiskEstimator.compute_covariance_matrix(prices_df)
            last_row = prices_df.iloc[-1]
            live_prices: dict[str, Decimal] = {
                sym: Decimal(str(round(float(last_row[sym]), 4)))
                for sym in universe
                if sym in last_row.index
            }
            _cached_prices.clear()
            _cached_prices.update(live_prices)
            equity_indices = [i for i, s in enumerate(universe) if s in _EQUITY_TICKERS]
            live_market_data: dict[str, Any] = {
                "asset_names": universe,
                "expected_returns": [expected_returns[s] for s in universe],
                "covariance_matrix": cov_df.values.tolist(),
                "equity_indices": equity_indices,
            }
            pipeline = _get_fiduciary_pipeline()
            portfolio_for_run = _lots_from_db(client_id) or _MOCK_PORTFOLIO
            _ai_results[record.hash] = pipeline.process_event(
                client_profile=client_profile,
                event_payload=event_payload,
                market_data=live_market_data,
                current_portfolio=portfolio_for_run,
                current_prices=live_prices,
            )
            _market_snapshots[record.hash] = live_market_data
            _record_fiduciary_block(
                record.hash, client_id, client_profile,
                _ai_results[record.hash],
                market_snapshot=live_market_data,
            )
        except Exception as exc:
            _logger.warning("fiduciary pipeline skipped for run %s: %s", run_id, exc)

        return RedirectResponse(f"/advice/{record.hash}", status_code=303)

    @app.get("/advice/{advice_hash}", response_class=HTMLResponse)
    def advice(request: Request, advice_hash: str) -> Response:
        return _dossier(
            request, ledger, advice_hash,
            ai_result=_ai_results.get(advice_hash),
            broker_result=_broker_results.get(advice_hash),
        )

    @app.get("/advice/{advice_hash}/pdf")
    def download_pdf(advice_hash: str) -> Response:
        ai_result = _ai_results.get(advice_hash) or {}
        audit_block_obj = _audit_blocks.get(advice_hash)
        audit_block: dict[str, Any] = audit_block_obj.model_dump() if audit_block_obj else {}
        client_id = str(ai_result.get("client_id", "C-1001"))
        client_profile = _mock_client(client_id)
        pdf_bytes = generate_fiduciary_pdf(audit_block, ai_result, client_profile)
        filename = f"Fiduciary_Audit_{advice_hash[:8]}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

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
            return _dossier(
                request, ledger, advice_hash, error=error, status_code=422,
                ai_result=_ai_results.get(advice_hash),
                broker_result=_broker_results.get(advice_hash),
            )
        try:
            ledger.record_decision(advice_hash, decision, adviser.strip(), clock(), note.strip())
        except LedgerError as refusal:
            error = f"The ledger refused the decision: {refusal}"
            return _dossier(
                request, ledger, advice_hash, error=error, status_code=409,
                ai_result=_ai_results.get(advice_hash),
                broker_result=_broker_results.get(advice_hash),
            )
        return RedirectResponse(f"/advice/{advice_hash}", status_code=303)

    @app.post("/advice/{advice_hash}/execute")
    async def execute(request: Request, advice_hash: str) -> Response:
        if advice_hash in _broker_results:
            return RedirectResponse(f"/advice/{advice_hash}", status_code=303)

        ai_result = _ai_results.get(advice_hash)
        if ai_result is None:
            error = "No pipeline result found for this advice run — cannot execute trades."
            return _dossier(request, ledger, advice_hash, error=error, status_code=409)

        trades: list[Trade] = ai_result.get("trades", [])
        if not trades:
            error = "There are no trades to execute for this advice run."
            return _dossier(
                request, ledger, advice_hash, error=error, status_code=409,
                ai_result=ai_result,
            )

        client_id: str = ai_result.get("client_id", "unknown")
        try:
            result = await _broker.execute_trades(client_id, trades)
            _broker_results[advice_hash] = dict(result)
        except Exception as exc:
            _logger.exception("broker execution failed for advice %s", advice_hash)
            error = f"Trade execution failed: {exc}"
            return _dossier(
                request, ledger, advice_hash, error=error, status_code=502,
                ai_result=ai_result,
            )

        _record_fiduciary_block(
            advice_hash,
            client_id,
            ai_result.get("client_profile", {"client_id": client_id}),
            ai_result,
            broker_receipt=_broker_results[advice_hash],
            market_snapshot=_market_snapshots.get(advice_hash),
        )
        return RedirectResponse(f"/advice/{advice_hash}", status_code=303)

    @app.get("/clients/{client_id}", response_class=HTMLResponse)
    def client_profile(
        request: Request,
        client_id: str,
        current_user: UserModel = Depends(RequireRole(["ADVISOR", "CCO"])),
    ) -> Response:
        return templates.TemplateResponse(
            request,
            "client_profile.html",
            {"client": _mock_client(client_id), "current_user": current_user},
        )

    @app.post("/clients/{client_id}/portfolio/upload")
    async def upload_portfolio(
        client_id: str,
        file: Annotated[UploadFile, File()],
    ) -> Response:
        raw = await file.read()
        lots = parse_brokerage_csv(raw.decode("utf-8", errors="replace"), TaxJurisdiction.INDIA)
        try:
            with SessionLocal() as session:
                session.query(TaxLotModel).filter(
                    TaxLotModel.client_id == client_id
                ).delete(synchronize_session=False)
                for lot in lots:
                    session.add(TaxLotModel(
                        client_id=client_id,
                        symbol=lot.symbol,
                        shares=lot.shares,
                        cost_basis=lot.cost_basis,
                        purchase_date=lot.purchase_date,
                        asset_jurisdiction=lot.asset_jurisdiction.value,
                    ))
                session.commit()
        except Exception as exc:
            _logger.warning("DB upload for %s failed: %s", client_id, exc)
        return RedirectResponse(f"/clients/{client_id}/portfolio", status_code=303)

    @app.get("/clients/{client_id}/portfolio", response_class=HTMLResponse)
    def client_portfolio(request: Request, client_id: str) -> Response:
        lots = _lots_from_db(client_id) or _MOCK_PORTFOLIO
        return templates.TemplateResponse(
            request,
            "portfolio.html",
            {"portfolio": _mock_portfolio_view(client_id, lots)},
        )

    @app.get("/api/v1/ledger/verify")
    def fiduciary_ledger_verify() -> dict[str, Any]:
        try:
            fl = _get_fiduciary_ledger()
            valid, _ = fl.verify_chain()
            total = fl.total_blocks()
            latest = fl.get_latest_block()
            return {
                "valid": valid,
                "total_blocks": total,
                "latest_hash": latest.signature_hash if latest else "",
            }
        except Exception as exc:
            _logger.warning("fiduciary ledger verify failed: %s", exc)
            return {"valid": False, "total_blocks": 0, "latest_hash": ""}

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


def _overview(
    request: Request,
    ledger: Ledger,
    current_user: UserModel | None = None,
) -> Response:
    records = _records(ledger)
    intact = records is not None and not ledger.verify()

    # Multi-tenancy: restrict non-CCO advisers to their own clients.
    if records is not None and current_user is not None and current_user.role != "CCO":
        allowed = _get_advisor_client_ids(int(str(current_user.id)))
        if allowed:
            records = [
                r for r in records
                if isinstance(r.entry, AdviceEntry)
                and r.entry.dossier.profile.client_id in allowed
            ]

    summaries = views.summaries(records) if records is not None else []
    alerts = _get_alerts(int(str(current_user.id))) if current_user is not None else []
    return templates.TemplateResponse(
        request,
        "overview.html",
        {
            "firm_aum": "$1.2B",
            "pending_actions": 14,
            "tax_alpha_ytd": "$412k",
            "intact": intact,
            "record_count": len(records) if records is not None else 0,
            "recent_runs": summaries[-5:][::-1] if summaries else [],
            "current_user": current_user,
            "alerts": alerts,
        },
    )


def _simulate_page(
    request: Request,
    can_run: bool,
    error: str = "",
    status_code: int = 200,
    current_user: UserModel | None = None,
) -> Response:
    return templates.TemplateResponse(
        request,
        "simulate.html",
        {"can_run": can_run, "error": error, "current_user": current_user},
        status_code=status_code,
    )


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
    request: Request,
    ledger: Ledger,
    advice_hash: str,
    error: str = "",
    status_code: int = 200,
    ai_result: dict[str, Any] | None = None,
    broker_result: dict[str, Any] | None = None,
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
        {
            "view": views.dossier(record, records),
            "error": error,
            "ai_result": ai_result,
            "broker_result": broker_result,
            "audit_block": _audit_blocks.get(advice_hash),
        },
        status_code=status_code,
    )
