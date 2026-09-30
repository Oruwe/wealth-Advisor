"""Firm-wide portfolio drift monitor.

check_firm_wide_drift() is designed to be called by a background scheduler.
For every client that has stored tax lots it:
  1. Fetches the latest market prices.
  2. Computes current allocation weights.
  3. Runs the cvxpy optimizer to find the mandate target weights.
  4. Flags any asset whose weight has drifted more than DRIFT_THRESHOLD from
     its target by writing a DRIFT_WARNING AlertModel row, skipping clients
     that already have an unresolved warning.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from decimal import Decimal

from wealth_advisor.engine.database import AlertModel, ClientModel, SessionLocal, TaxLotModel
from wealth_advisor.engine.market_data import MarketDataFeed, RiskEstimator
from wealth_advisor.engine.optimizer import FiduciaryOptimizer

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.05  # 5 % absolute weight deviation
_EQUITY_TICKERS: frozenset[str] = frozenset({"VOO", "VEA", "VWO"})


# ── Public entry point ─────────────────────────────────────────────────────────


def check_firm_wide_drift() -> None:
    """Scan every client in the DB and create drift alerts where warranted."""
    try:
        with SessionLocal() as session:
            clients = session.query(ClientModel).all()
            client_snapshots = [
                {
                    "client_id": str(c.id),
                    "advisor_id": int(c.advisor_id) if c.advisor_id is not None else 0,
                    "target_risk_aversion": float(c.target_risk_aversion or 1.0),
                    "max_equity_cap": float(c.max_equity_cap or 1.0),
                }
                for c in clients
            ]
    except Exception as exc:
        logger.warning("drift monitor: failed to load clients: %s", exc)
        return

    for snap in client_snapshots:
        try:
            _check_client(
                client_id=snap["client_id"],
                advisor_id=snap["advisor_id"],
                target_risk_aversion=snap["target_risk_aversion"],
                max_equity_cap=snap["max_equity_cap"],
            )
        except Exception as exc:
            logger.warning("drift monitor: unhandled error for %s: %s", snap["client_id"], exc)


# ── Per-client logic ───────────────────────────────────────────────────────────


def _check_client(
    client_id: str,
    advisor_id: int,
    target_risk_aversion: float,
    max_equity_cap: float,
) -> None:
    # ── 1. Load lots from DB ───────────────────────────────────────────────────
    try:
        with SessionLocal() as session:
            lot_rows = (
                session.query(TaxLotModel)
                .filter(TaxLotModel.client_id == client_id)
                .all()
            )
            if not lot_rows:
                return
            symbols: list[str] = list({str(r.symbol) for r in lot_rows})
            shares_by_symbol: dict[str, Decimal] = {}
            for r in lot_rows:
                sym = str(r.symbol)
                shares_by_symbol[sym] = (
                    shares_by_symbol.get(sym, Decimal("0")) + Decimal(str(r.shares))
                )
    except Exception as exc:
        logger.warning("drift monitor: lot load failed for %s: %s", client_id, exc)
        return

    # ── 2. Fetch prices ────────────────────────────────────────────────────────
    try:
        prices_df = MarketDataFeed.get_historical_prices(symbols)
        last_row = prices_df.iloc[-1]
        prices: dict[str, float] = {
            sym: float(last_row[sym])
            for sym in symbols
            if sym in prices_df.columns
        }
        if not prices:
            return
    except Exception as exc:
        logger.warning("drift monitor: price fetch failed for %s: %s", client_id, exc)
        return

    # ── 3. Current weights ─────────────────────────────────────────────────────
    values: dict[str, float] = {
        sym: float(shares_by_symbol.get(sym, Decimal("0"))) * prices[sym]
        for sym in prices
    }
    total_value = sum(values.values())
    if total_value <= 0:
        return
    current_weights: dict[str, float] = {sym: v / total_value for sym, v in values.items()}

    # ── 4. Optimizer target weights ────────────────────────────────────────────
    try:
        expected_returns = RiskEstimator.compute_expected_returns(prices_df)
        cov_df = RiskEstimator.compute_covariance_matrix(prices_df)
        asset_names = [s for s in symbols if s in prices_df.columns]
        equity_indices = [i for i, s in enumerate(asset_names) if s in _EQUITY_TICKERS]
        mu = [float(expected_returns.get(s, 0.0)) for s in asset_names]
        sigma = cov_df.values.tolist()
        target_w_list = FiduciaryOptimizer().optimize_allocation(
            expected_returns=mu,
            covariance_matrix=sigma,
            equity_indices=equity_indices,
            target_risk_aversion=target_risk_aversion,
            max_equity_exposure=max_equity_cap,
        )
        target_weights: dict[str, float] = dict(zip(asset_names, target_w_list, strict=True))
    except Exception as exc:
        logger.warning("drift monitor: optimizer failed for %s: %s", client_id, exc)
        return

    # ── 5. Measure drift ───────────────────────────────────────────────────────
    max_drift = 0.0
    worst_asset = ""
    for sym in asset_names:
        drift = abs(current_weights.get(sym, 0.0) - target_weights.get(sym, 0.0))
        if drift > max_drift:
            max_drift = drift
            worst_asset = sym

    if max_drift < DRIFT_THRESHOLD:
        return

    # ── 6. Create alert if none already exists for today ──────────────────────
    try:
        today = date.today()
        with SessionLocal() as session:
            existing = (
                session.query(AlertModel)
                .filter(
                    AlertModel.client_id == client_id,
                    AlertModel.alert_type == "DRIFT_WARNING",
                    AlertModel.is_resolved.is_(False),
                )
                .first()
            )
            if existing is not None:
                existing_date = (
                    existing.created_at.date()
                    if hasattr(existing.created_at, "date")
                    else today
                )
                if existing_date == today:
                    return

            message = (
                f"Portfolio drifted {max_drift * 100:.1f}% from mandate "
                f"({worst_asset} is the worst offender). Rebalance recommended."
            )
            session.add(AlertModel(
                client_id=client_id,
                advisor_id=advisor_id,
                alert_type="DRIFT_WARNING",
                message=message,
                is_resolved=False,
                created_at=datetime.now(UTC),
            ))
            session.commit()
            logger.info(
                "DRIFT_WARNING created for client %s (%.1f%% drift in %s)",
                client_id, max_drift * 100, worst_asset,
            )
    except Exception as exc:
        logger.warning("drift monitor: alert write failed for %s: %s", client_id, exc)
