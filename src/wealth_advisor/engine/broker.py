"""Broker execution layer: translate Trade objects into broker-specific REST payloads.

AlpacaBrokerAdapter targets the Alpaca paper-trading API.  In dry_run mode (the default)
it logs the exact JSON it would POST and returns a mocked response — no network calls are
made.  Set dry_run=False only when a live ALPACA_API_KEY and ALPACA_SECRET_KEY are
available in the environment.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any
from uuid import uuid4

from wealth_advisor.engine.tax import Trade, TradeAction

logger = logging.getLogger(__name__)

_ALPACA_ORDERS_URL = "https://paper-api.alpaca.markets/v2/orders"


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class OrderPayload(dict[str, Any]):
    """A single Alpaca order payload (typed alias for documentation clarity)."""


class BrokerResult(dict[str, Any]):
    """Return value of execute_trades: contains 'orders' list and 'dry_run' flag."""


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BrokerAdapter(ABC):
    """Async interface every concrete broker adapter must implement."""

    @abstractmethod
    async def execute_trades(
        self,
        client_id: str,
        trades: list[Trade],
    ) -> BrokerResult:
        """Submit *trades* on behalf of *client_id* and return a result record.

        The result must include at minimum:
          - "dry_run":  bool  — whether this was a live or simulated submission
          - "orders":   list  — one entry per trade with at least "order_id" and "status"
        """


# ---------------------------------------------------------------------------
# Alpaca implementation
# ---------------------------------------------------------------------------

class AlpacaBrokerAdapter(BrokerAdapter):
    """Translates Trade objects into Alpaca v2 market orders.

    Args:
        dry_run: When True (default) log payloads and return mocked responses.
                 When False, POST to the Alpaca paper-trading endpoint using
                 api_key and secret_key.
        api_key: Alpaca API key ID (required when dry_run=False).
        secret_key: Alpaca secret key (required when dry_run=False).
    """

    def __init__(
        self,
        dry_run: bool = True,
        api_key: str = "",
        secret_key: str = "",
    ) -> None:
        if not dry_run and (not api_key or not secret_key):
            raise ValueError("api_key and secret_key are required when dry_run is False")
        self._dry_run = dry_run
        self._api_key = api_key
        self._secret_key = secret_key

    async def execute_trades(
        self,
        client_id: str,
        trades: list[Trade],
    ) -> BrokerResult:
        """Build and (dry-)submit one Alpaca market order per trade."""
        payloads = [_to_alpaca_payload(trade) for trade in trades]
        if self._dry_run:
            return self._dry_run_submit(client_id, payloads)
        return await self._live_submit(client_id, payloads)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dry_run_submit(
        self,
        client_id: str,
        payloads: list[OrderPayload],
    ) -> BrokerResult:
        orders: list[dict[str, Any]] = []
        for payload in payloads:
            order_id = uuid4().hex
            logger.info(
                "DRY RUN — would POST to %s: %s",
                _ALPACA_ORDERS_URL,
                json.dumps(payload, sort_keys=True),
                extra={"client_id": client_id, "order_id": order_id},
            )
            orders.append(
                {
                    "order_id": order_id,
                    "client_order_id": f"{client_id}-{order_id}",
                    "symbol": payload["symbol"],
                    "side": payload["side"],
                    "qty": payload["qty"],
                    "type": payload["type"],
                    "time_in_force": payload["time_in_force"],
                    "status": "simulated_accepted",
                    "url": _ALPACA_ORDERS_URL,
                }
            )
        logger.info(
            "dry-run complete: %d order(s) simulated for client %s",
            len(orders),
            client_id,
        )
        return BrokerResult(
            dry_run=True,
            orders=orders,
            client_id=client_id,
        )

    async def _live_submit(
        self,
        client_id: str,
        payloads: list[OrderPayload],
    ) -> BrokerResult:
        # Import httpx lazily so the module is usable without it in dry-run mode.
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for live broker submission: "
                "add it to your dependencies and run 'uv sync'"
            ) from exc

        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Content-Type": "application/json",
        }
        orders: list[dict[str, Any]] = []
        async with httpx.AsyncClient() as client:
            for payload in payloads:
                response = await client.post(
                    _ALPACA_ORDERS_URL,
                    headers=headers,
                    json=payload,
                    timeout=10.0,
                )
                response.raise_for_status()
                body: dict[str, Any] = response.json()
                orders.append(
                    {
                        "order_id": body.get("id", ""),
                        "client_order_id": body.get("client_order_id", ""),
                        "symbol": body.get("symbol", payload["symbol"]),
                        "side": body.get("side", payload["side"]),
                        "qty": body.get("qty", payload["qty"]),
                        "type": body.get("type", payload["type"]),
                        "time_in_force": body.get("time_in_force", payload["time_in_force"]),
                        "status": body.get("status", ""),
                    }
                )
                logger.info(
                    "order submitted",
                    extra={
                        "client_id": client_id,
                        "order_id": body.get("id"),
                        "symbol": payload["symbol"],
                        "side": payload["side"],
                    },
                )
        return BrokerResult(
            dry_run=False,
            orders=orders,
            client_id=client_id,
        )


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _to_alpaca_payload(trade: Trade) -> OrderPayload:
    """Translate one Trade into an Alpaca v2 order payload dict.

    Alpaca rules applied:
      - type:           "market"  (no limit price emitted by this layer)
      - time_in_force:  "day"
      - side:           "buy" or "sell" (from TradeAction)
      - qty:            string representation of Decimal shares (preserves precision,
                        avoids float rounding; Alpaca accepts numeric strings)
    """
    if trade.shares <= Decimal(0):
        raise ValueError(
            f"trade for {trade.symbol} has non-positive shares: {trade.shares}"
        )
    side = "buy" if trade.action == TradeAction.BUY else "sell"
    return OrderPayload(
        symbol=trade.symbol,
        qty=str(trade.shares),
        side=side,
        type="market",
        time_in_force="day",
    )
