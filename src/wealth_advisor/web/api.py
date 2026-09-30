"""JSON REST endpoints for the fiduciary simulation pipeline."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from wealth_advisor.engine.pipeline import FiduciaryPipeline, PipelineError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Four-asset mock market data used by the simulate endpoint.
# Covariance matrix is symmetric positive-definite (diagonal dominant).
_MOCK_MARKET_DATA: dict[str, Any] = {
    "asset_names": ["US_EQUITY", "INTL_EQUITY", "BONDS", "CASH"],
    "expected_returns": [0.10, 0.09, 0.04, 0.02],
    "covariance_matrix": [
        [0.0400, 0.0200, 0.0040, 0.0000],
        [0.0200, 0.0400, 0.0030, 0.0000],
        [0.0040, 0.0030, 0.0025, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0001],
    ],
    "equity_indices": [0, 1],
}

# Baseline conservative client profile (mirrors demo client C-1001).
_BASE_CLIENT_PROFILE: dict[str, Any] = {
    "risk_tolerance": 3,
    "time_horizon_years": 10,
    "profile_type": "conservative",
    "cash_reserve_usd": 2000,
}


class SimulateRequest(BaseModel):
    event_type: str
    client_id: str


@lru_cache(maxsize=1)
def _get_pipeline() -> FiduciaryPipeline:
    """Singleton pipeline; created lazily so missing LYZR_API_KEY fails at request time."""
    return FiduciaryPipeline()


@router.post("/fiduciary/simulate")
def simulate(body: SimulateRequest) -> dict[str, Any]:
    """Run one event through the fiduciary pipeline and return weights and risk parameters."""
    try:
        pipeline = _get_pipeline()
    except Exception as exc:
        logger.error("fiduciary pipeline unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="fiduciary pipeline unavailable — LYZR_API_KEY not configured or Lyzr unreachable",
        ) from exc

    client_profile = {**_BASE_CLIENT_PROFILE, "client_id": body.client_id}
    event_payload = {"event_type": body.event_type, "client_id": body.client_id}

    try:
        result = pipeline.process_event(
            client_profile=client_profile,
            event_payload=event_payload,
            market_data=_MOCK_MARKET_DATA,
        )
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "client_id": result["client_id"],
        "event_type": result["event_type"],
        "risk_parameters": result["risk_parameters"],
        "target_weights": result["target_weights"],
        "timestamp": result["timestamp"],
    }
