import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from types import MappingProxyType

from wealth_advisor.domain.client import ClientProfile
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.orders import Order, RebalanceProposal, Side
from wealth_advisor.domain.portfolio import (
    AssetClass,
    Holding,
    Portfolio,
    Security,
    TargetAllocation,
)
from wealth_advisor.domain.primitives import EXACT_CONTEXT
from wealth_advisor.policy.model_portfolio import MODEL_SECURITIES, TARGET_BY_RISK
from wealth_advisor.policy.suitability import SUITABILITY_POLICY

_INVESTED = (AssetClass.EQUITY, AssetClass.FIXED_INCOME, AssetClass.COMMODITY)
_NO_CAPS: Mapping[AssetClass, Decimal] = MappingProxyType({})


class RebalanceError(ValueError):
    """The inputs cannot produce a valid proposal."""


@dataclass(frozen=True)
class _Leg:
    security: Security
    price: Decimal
    quantity: Decimal
    target: Decimal

    def candidate_trades(self) -> list[Decimal]:
        """The ideal trade rounded down and up to whole shares, never selling more than is held."""
        whole, remainder = divmod(self.target - self.quantity * self.price, self.price)
        if remainder == 0:
            down = up = whole
        elif remainder < 0:
            down, up = whole - 1, whole
        else:
            down, up = whole, whole + 1
        return sorted({max(down, -self.quantity), up})

    def value_after(self, trade: Decimal) -> Decimal:
        return (self.quantity + trade) * self.price


def propose_rebalance(
    profile: ClientProfile, portfolio: Portfolio, prices: PriceSnapshot
) -> RebalanceProposal:
    if profile.client_id != portfolio.client_id:
        raise RebalanceError(
            f"profile is for {profile.client_id}, portfolio is for {portfolio.client_id}"
        )
    return rebalance(
        portfolio,
        prices,
        TARGET_BY_RISK[profile.risk_tolerance],
        profile.cash_reserve,
        max_weight=SUITABILITY_POLICY.band_for(profile.risk_tolerance).max_weight,
    )


def rebalance(
    portfolio: Portfolio,
    prices: PriceSnapshot,
    target: TargetAllocation,
    cash_reserve: Decimal,
    max_weight: Mapping[AssetClass, Decimal] = _NO_CAPS,
    model: Mapping[AssetClass, Security] = MODEL_SECURITIES,
) -> RebalanceProposal:
    """Sell holdings outside the model, then trade each model security in whole shares, choosing
    the rounding (down or up, per security) with the least squared dollar drift from target that
    keeps cash at or above the reserve and every capped asset class within its cap. Ties go to
    the lower turnover."""
    with localcontext(EXACT_CONTEXT):
        total = portfolio.market_value(prices)
        if cash_reserve > total:
            raise RebalanceError(f"cash reserve {cash_reserve} exceeds portfolio value {total}")
        for asset_class, cap in max_weight.items():
            weight = target.weights[asset_class]
            if weight > cap:
                raise RebalanceError(f"target {asset_class} weight {weight} exceeds its cap {cap}")

        exits, held = _split_by_model(portfolio, model)
        exit_orders = [
            Order(
                side=Side.SELL,
                security=holding.security,
                quantity=holding.quantity,
                price=prices.price_of(holding.security.symbol),
            )
            for holding in exits
        ]
        cash = portfolio.cash + sum(
            (order.quantity * order.price for order in exit_orders), start=Decimal(0)
        )

        targets = _dollar_targets(total, target, cash_reserve)
        legs = {
            asset_class: _Leg(
                security=model[asset_class],
                price=prices.price_of(model[asset_class].symbol),
                quantity=held.get(model[asset_class].symbol, Decimal(0)),
                target=targets[asset_class],
            )
            for asset_class in _INVESTED
        }
        limits = {asset_class: cap * total for asset_class, cap in max_weight.items()}
        trades = _best_trades(legs, cash, cash_reserve, limits)
        cash_after = cash - sum(
            (trade * legs[asset_class].price for asset_class, trade in trades.items()),
            start=Decimal(0),
        )

        trade_orders = [
            Order(
                side=Side.BUY if trade > 0 else Side.SELL,
                security=legs[asset_class].security,
                quantity=abs(trade),
                price=legs[asset_class].price,
            )
            for asset_class, trade in trades.items()
            if trade != 0
        ]
        return RebalanceProposal(
            client_id=portfolio.client_id,
            as_of=portfolio.as_of,
            target=target,
            target_values={
                **targets,
                AssetClass.CASH: total - sum(targets.values(), start=Decimal(0)),
            },
            orders=tuple(
                sorted(
                    [*exit_orders, *trade_orders],
                    key=lambda order: (order.side is Side.BUY, order.security.symbol),
                )
            ),
            value_after={
                **{cls: legs[cls].value_after(trade) for cls, trade in trades.items()},
                AssetClass.CASH: cash_after,
            },
            cash_after=cash_after,
        )


def _split_by_model(
    portfolio: Portfolio, model: Mapping[AssetClass, Security]
) -> tuple[list[Holding], dict[str, Decimal]]:
    model_by_symbol = {security.symbol: security for security in model.values()}
    exits: list[Holding] = []
    held: dict[str, Decimal] = {}
    for holding in portfolio.holdings:
        expected = model_by_symbol.get(holding.security.symbol)
        if expected is None:
            exits.append(holding)
        elif holding.security != expected:
            raise RebalanceError(
                f"{holding.security.symbol} is held as {holding.security.asset_class}, "
                f"but the model uses it for {expected.asset_class}"
            )
        else:
            held[holding.security.symbol] = holding.quantity
    return exits, held


def _dollar_targets(
    total: Decimal, target: TargetAllocation, cash_reserve: Decimal
) -> dict[AssetClass, Decimal]:
    """Weight times portfolio value, scaled down pro rata when the reserve needs more than the cash
    weight, and rounded down to the cent so the targets never exceed the money available."""
    cash_target = target.cash * total
    scale = Fraction(1)
    if cash_reserve > cash_target:
        scale = Fraction(total - cash_reserve) / Fraction(total - cash_target)
    return {
        asset_class: _cents_down(Fraction(target.weights[asset_class] * total) * scale)
        for asset_class in _INVESTED
    }


def _cents_down(amount: Fraction) -> Decimal:
    return Decimal(math.floor(amount * 100)) / 100


def _best_trades(
    legs: Mapping[AssetClass, _Leg],
    cash: Decimal,
    cash_reserve: Decimal,
    limits: Mapping[AssetClass, Decimal],
) -> dict[AssetClass, Decimal]:
    def allowed(trades: tuple[Decimal, ...]) -> bool:
        after = {
            asset_class: leg.value_after(trade)
            for (asset_class, leg), trade in zip(legs.items(), trades, strict=True)
        }
        spent = sum(
            (trade * leg.price for leg, trade in zip(legs.values(), trades, strict=True)),
            start=Decimal(0),
        )
        return cash - spent >= cash_reserve and all(
            after[asset_class] <= limit for asset_class, limit in limits.items()
        )

    def rank(trades: tuple[Decimal, ...]) -> tuple[Decimal, Decimal, tuple[Decimal, ...]]:
        pairs = list(zip(legs.values(), trades, strict=True))
        drift = sum(((leg.target - leg.value_after(t)) ** 2 for leg, t in pairs), start=Decimal(0))
        turnover = sum((abs(t) * leg.price for leg, t in pairs), start=Decimal(0))
        return drift, turnover, trades

    allowed_trades = [
        trades
        for trades in product(*(leg.candidate_trades() for leg in legs.values()))
        if allowed(trades)
    ]
    best = min(allowed_trades, key=rank)
    return dict(zip(legs, best, strict=True))
