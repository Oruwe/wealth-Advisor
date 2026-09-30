"""End-to-end wealth-advisor pipeline: event -> strategic parameters -> deterministic weights.

The pipeline composes three stages:

  1. `FiduciaryBrain`      -- Lyzr agent that turns an event into bounded risk parameters.
  2. `FiduciaryOptimizer`  -- cvxpy solver that turns those bounds into asset weights.
  3. `WealthEventBus`      -- Redis-streams transport for producing/consuming events.

Numbers only ever come from the deterministic optimizer; the agent is bounded to parameters.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from wealth_advisor.engine.cross_border import CrossBorderEngine, UCITSRegistry
from wealth_advisor.engine.event_bus import WealthEventBus
from wealth_advisor.engine.optimizer import FiduciaryOptimizer
from wealth_advisor.engine.strategic_agent import FiduciaryBrain
from wealth_advisor.engine.tax import TaxLot
from wealth_advisor.engine.trade_generator import TradeGenerator

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when pipeline inputs are malformed or a stage fails irrecoverably."""


class FiduciaryPipeline:
    """Compose the strategic brain, the deterministic optimizer, and the event bus."""

    def __init__(
        self,
        brain: FiduciaryBrain | None = None,
        optimizer: FiduciaryOptimizer | None = None,
        event_bus: WealthEventBus | None = None,
    ) -> None:
        self._brain = brain or FiduciaryBrain()
        self._optimizer = optimizer or FiduciaryOptimizer()
        self._event_bus = event_bus or WealthEventBus(brain=self._brain)

    @property
    def event_bus(self) -> WealthEventBus:
        return self._event_bus

    def process_event(
        self,
        client_profile: dict[str, Any],
        event_payload: dict[str, Any],
        market_data: dict[str, Any],
        current_portfolio: list[TaxLot] | None = None,
        current_prices: dict[str, Decimal] | None = None,
    ) -> dict[str, Any]:
        """Run one full event -> parameters -> weights -> trades cycle and return a result record.

        `market_data` must contain:
          - "asset_names":       list[str], N assets in canonical order
          - "expected_returns":  sequence of N floats
          - "covariance_matrix": nested sequence, shape N x N
          - "equity_indices":    list[int] indices into `asset_names` that are equities
        """
        client_id = str(client_profile.get("client_id", "unknown"))
        event_type = str(event_payload.get("event_type", "unknown"))

        asset_names, expected_returns, covariance_matrix, equity_indices = (
            self._extract_market_data(market_data, client_id, event_type)
        )
        parameters = self._evaluate(client_profile, event_payload, client_id, event_type)
        weights = self._optimize(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            equity_indices=equity_indices,
            parameters=parameters,
            client_id=client_id,
            event_type=event_type,
        )

        if len(weights) != len(asset_names):
            raise PipelineError(
                f"optimizer returned {len(weights)} weights but market_data lists "
                f"{len(asset_names)} assets"
            )

        target_weights: dict[str, float] = {
            name: float(weight)
            for name, weight in zip(asset_names, weights, strict=True)
        }

        # Portfolio value needed for both cross-border assessment and trade generation.
        portfolio_value = Decimal(0)
        if current_portfolio is not None and current_prices is not None:
            portfolio_value = sum(
                (lot.shares * current_prices[lot.symbol] for lot in current_portfolio),
                start=Decimal(0),
            )

        # Cross-border compliance: LRS / FEMA / UCITS estate-tax shielding.
        _cross_border = CrossBorderEngine()
        jurisdiction = str(client_profile.get("tax_jurisdiction", "US")).upper()
        us_tickers = set(UCITSRegistry.keys())
        proposed_us_assets_value: Decimal = sum(
            (Decimal(str(target_weights.get(ticker, 0.0))) * portfolio_value
             for ticker in us_tickers),
            start=Decimal(0),
        )
        estate_tax_risk, estate_tax_message = _cross_border.evaluate_estate_tax_risk(
            jurisdiction, proposed_us_assets_value
        )
        ucits_substitutions: dict[str, str] = {}
        if estate_tax_risk:
            new_weights: dict[str, float] = {}
            for name, weight in target_weights.items():
                if name in UCITSRegistry:
                    ucits_equiv = UCITSRegistry[name]
                    new_weights[ucits_equiv] = new_weights.get(ucits_equiv, 0.0) + weight
                    ucits_substitutions[name] = ucits_equiv
                else:
                    new_weights[name] = weight
            target_weights = new_weights

        lrs_ytd = Decimal(str(client_profile.get("lrs_quota_used_usd", 0)))
        lrs_metrics = _cross_border.calculate_lrs_tcs(lrs_ytd, proposed_us_assets_value)
        cross_border_metrics: dict[str, Any] = {
            "jurisdiction": jurisdiction,
            "estate_tax_risk": estate_tax_risk,
            "estate_tax_message": estate_tax_message,
            "ucits_substitutions": ucits_substitutions,
            "lrs_breach": lrs_metrics["lrs_breach"],
            "lrs_utilized_usd": lrs_metrics["lrs_utilized_usd"],
            "lrs_remaining_usd": lrs_metrics["lrs_remaining_usd"],
            "tcs_tax_usd": lrs_metrics["tcs_tax_usd"],
            "tcs_rate_pct": lrs_metrics["tcs_rate_pct"],
            "new_trade_usd": lrs_metrics["new_trade_usd"],
            "remitted_ytd_usd": lrs_metrics["remitted_ytd_usd"],
        }

        trades = []
        if current_portfolio is not None and current_prices is not None:
            # Only pass tickers that have live prices; UCITS substitutes may not be in the feed.
            decimal_weights: dict[str, Decimal] = {
                name: Decimal(str(w))
                for name, w in target_weights.items()
                if name in current_prices
            }
            trades = TradeGenerator().generate_trades(
                current_portfolio=current_portfolio,
                target_weights=decimal_weights,
                portfolio_value=portfolio_value,
                current_prices=current_prices,
            )

        result: dict[str, Any] = {
            "client_id": client_id,
            "event_type": event_type,
            "risk_parameters": parameters,
            "target_weights": target_weights,
            "trades": trades,
            "cross_border_metrics": cross_border_metrics,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        logger.info(
            "event processed",
            extra={"client_id": client_id, "event_type": event_type},
        )
        return result

    def _evaluate(
        self,
        client_profile: dict[str, Any],
        event_payload: dict[str, Any],
        client_id: str,
        event_type: str,
    ) -> dict[str, float]:
        try:
            return self._brain.evaluate_event(client_profile, event_payload)
        except Exception as exc:
            logger.exception(
                "strategic evaluation failed",
                extra={"client_id": client_id, "event_type": event_type},
            )
            raise PipelineError(f"strategic evaluation failed: {exc}") from exc

    def _optimize(
        self,
        *,
        expected_returns: Any,
        covariance_matrix: Any,
        equity_indices: list[int],
        parameters: dict[str, float],
        client_id: str,
        event_type: str,
    ) -> Any:
        try:
            return self._optimizer.optimize_allocation(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                equity_indices=equity_indices,
                target_risk_aversion=parameters["target_risk_aversion"],
                max_equity_exposure=parameters["max_equity_exposure"],
            )
        except Exception as exc:
            logger.exception(
                "optimizer failed",
                extra={"client_id": client_id, "event_type": event_type},
            )
            raise PipelineError(f"optimizer failed: {exc}") from exc

    @staticmethod
    def _extract_market_data(
        market_data: dict[str, Any],
        client_id: str,
        event_type: str,
    ) -> tuple[list[str], Any, Any, list[int]]:
        required = ("asset_names", "expected_returns", "covariance_matrix", "equity_indices")
        missing = [key for key in required if key not in market_data]
        if missing:
            logger.error(
                "malformed market_data",
                extra={
                    "client_id": client_id,
                    "event_type": event_type,
                    "missing": ",".join(missing),
                },
            )
            raise PipelineError(f"market_data is missing keys: {', '.join(missing)}")

        try:
            asset_names = [str(name) for name in market_data["asset_names"]]
            expected_returns = market_data["expected_returns"]
            covariance_matrix = market_data["covariance_matrix"]
            equity_indices = [int(i) for i in market_data["equity_indices"]]
        except (TypeError, ValueError) as exc:
            raise PipelineError(f"malformed market_data: {exc}") from exc

        if not asset_names:
            raise PipelineError("market_data.asset_names is empty")
        if len(expected_returns) != len(asset_names):
            raise PipelineError(
                f"expected_returns has {len(expected_returns)} entries but asset_names has "
                f"{len(asset_names)}"
            )
        for index in equity_indices:
            if not 0 <= index < len(asset_names):
                raise PipelineError(
                    f"equity_indices contains {index}, out of range for {len(asset_names)} assets"
                )
        return asset_names, expected_returns, covariance_matrix, equity_indices
