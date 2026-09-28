from collections.abc import Iterator
from decimal import Decimal, localcontext
from fractions import Fraction

from wealth_advisor.domain.client import ClientProfile
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.orders import RebalanceProposal, Side
from wealth_advisor.domain.portfolio import AssetClass, Portfolio, Security
from wealth_advisor.domain.primitives import EXACT_CONTEXT
from wealth_advisor.domain.suitability import Rule, SuitabilityReport, Violation
from wealth_advisor.policy.suitability import SUITABILITY_POLICY, SuitabilityPolicy


def check_suitability(
    profile: ClientProfile,
    portfolio: Portfolio,
    prices: PriceSnapshot,
    proposal: RebalanceProposal,
    policy: SuitabilityPolicy = SUITABILITY_POLICY,
) -> SuitabilityReport:
    """Replay the proposal's orders on the portfolio and check the result against the client's IPS
    and the policy. The figures the proposal reports about itself are verified, never trusted."""
    with localcontext(EXACT_CONTEXT):
        violations = tuple(_violations(profile, portfolio, prices, proposal, policy))
    return SuitabilityReport(
        client_id=proposal.client_id, as_of=proposal.as_of, violations=violations
    )


def _violations(
    profile: ClientProfile,
    portfolio: Portfolio,
    prices: PriceSnapshot,
    proposal: RebalanceProposal,
    policy: SuitabilityPolicy,
) -> Iterator[Violation]:
    clients = (profile.client_id, portfolio.client_id, proposal.client_id)
    if len(set(clients)) > 1:
        yield Violation(
            rule=Rule.SAME_CLIENT,
            detail=f"profile, portfolio and proposal belong to {', '.join(clients)}",
        )
    dates = (portfolio.as_of, prices.as_of, proposal.as_of)
    if len(set(dates)) > 1:
        yield Violation(
            rule=Rule.SAME_DATE,
            detail=f"portfolio, prices and proposal are dated {', '.join(map(str, dates))}",
        )

    quantities = {holding.security: holding.quantity for holding in portfolio.holdings}
    unit_prices = {security: prices.price_of(security.symbol) for security in quantities}
    cash = portfolio.cash
    bought_unapproved: set[str] = set()
    for order in proposal.orders:
        symbol = order.security.symbol
        snapshot_price = prices.prices.get(symbol)
        if snapshot_price is None:
            yield Violation(
                rule=Rule.SNAPSHOT_PRICES,
                detail=f"{order.side} {symbol} at {order.price}, but the snapshot has no price",
            )
        elif snapshot_price != order.price:
            yield Violation(
                rule=Rule.SNAPSHOT_PRICES,
                detail=f"{order.side} {symbol} at {order.price}, "
                f"but the snapshot price is {snapshot_price}",
            )
        held = quantities.get(order.security, Decimal(0))
        if order.side is Side.SELL:
            if order.quantity > held:
                yield Violation(
                    rule=Rule.NO_OVERSELLING,
                    detail=f"sells {order.quantity} {symbol} but only {held} are held",
                )
            quantities[order.security] = held - order.quantity
            cash += order.quantity * order.price
        else:
            if symbol not in policy.approved_securities:
                bought_unapproved.add(symbol)
                yield Violation(
                    rule=Rule.APPROVED_SECURITIES,
                    detail=f"buys {symbol}, which is not on the approved list",
                )
            quantities[order.security] = held + order.quantity
            cash -= order.quantity * order.price
        unit_prices.setdefault(
            order.security, order.price if snapshot_price is None else snapshot_price
        )

    values = _values(quantities, unit_prices, cash)
    if values != proposal.value_after or cash != proposal.cash_after:
        yield Violation(
            rule=Rule.MATCHING_FIGURES,
            detail="the values the proposal reports differ from what its orders produce",
        )

    for security, quantity in sorted(quantities.items(), key=lambda item: item[0].symbol):
        symbol = security.symbol
        if quantity > 0 and symbol not in policy.approved_securities | bought_unapproved:
            yield Violation(
                rule=Rule.APPROVED_SECURITIES,
                detail=f"keeps {symbol}, which is not on the approved list",
            )

    total = sum(values.values(), start=Decimal(0))
    band = policy.band_for(profile.risk_tolerance)
    for asset_class, cap in band.max_weight.items():
        if total > 0 and values[asset_class] > cap * total:
            share = _percent(Fraction(values[asset_class]) / Fraction(total))
            yield Violation(
                rule=Rule.MAX_WEIGHT,
                detail=f"{asset_class} would be {share} of the portfolio; "
                f"{band.name} clients may hold at most {_percent(Fraction(cap))}",
            )

    if cash < profile.cash_reserve:
        yield Violation(
            rule=Rule.CASH_RESERVE,
            detail=f"cash would be {_dollars(cash)} after the trades; "
            f"the IPS requires at least {_dollars(profile.cash_reserve)}",
        )


def _values(
    quantities: dict[Security, Decimal], unit_prices: dict[Security, Decimal], cash: Decimal
) -> dict[AssetClass, Decimal]:
    values = dict.fromkeys(AssetClass, Decimal(0))
    values[AssetClass.CASH] += cash
    for security, quantity in quantities.items():
        values[security.asset_class] += quantity * unit_prices[security]
    return values


def _percent(fraction: Fraction) -> str:
    basis_points = round(fraction * 10_000)
    return f"{basis_points // 100}.{basis_points % 100:02d}%"


def _dollars(amount: Decimal) -> str:
    return f"-${-amount:,.2f}" if amount < 0 else f"${amount:,.2f}"
