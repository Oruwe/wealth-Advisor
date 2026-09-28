from collections.abc import Iterator, Sequence
from datetime import date, timedelta
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import NamedTuple

from wealth_advisor.domain.client import ClientProfile
from wealth_advisor.domain.orders import Order, Side
from wealth_advisor.domain.portfolio import Portfolio, Security, TaxLot
from wealth_advisor.domain.primitives import EXACT_CONTEXT
from wealth_advisor.domain.tax import LotSale, TaxImpact, Term, WashSaleWarning
from wealth_advisor.policy.tax_rates import long_term_rate

WASH_SALE_WINDOW = timedelta(days=30)


class TaxError(ValueError):
    """The orders cannot be matched to the portfolio's tax lots."""


class NettedGains(NamedTuple):
    short_term: Decimal
    long_term: Decimal
    loss: Decimal


def net_capital_gains(short_term: Decimal, long_term: Decimal) -> NettedGains:
    """Net short- and long-term results the way Schedule D does: a loss in one offsets a gain in
    the other, and whatever loss remains is reported rather than taxed."""
    if short_term >= 0 and long_term >= 0:
        return NettedGains(short_term, long_term, Decimal(0))
    total = short_term + long_term
    if total <= 0:
        return NettedGains(Decimal(0), Decimal(0), -total)
    if short_term > 0:
        return NettedGains(total, Decimal(0), Decimal(0))
    return NettedGains(Decimal(0), total, Decimal(0))


def estimate_tax(
    profile: ClientProfile, portfolio: Portfolio, orders: Sequence[Order]
) -> TaxImpact:
    """Match each sale to the highest-cost lots first (HIFO) and estimate the federal income tax
    on the result. Excludes state tax and the 3.8% net investment income tax."""
    if profile.client_id != portfolio.client_id:
        raise TaxError(
            f"profile is for {profile.client_id}, portfolio is for {portfolio.client_id}"
        )
    with localcontext(EXACT_CONTEXT):
        lots = {holding.security: holding.lots for holding in portfolio.holdings}
        left = {security: [lot.quantity for lot in held] for security, held in lots.items()}
        lot_sales = [
            sale
            for order in orders
            if order.side is Side.SELL
            for sale in _relieve(
                order, lots.get(order.security, ()), left.get(order.security, []), portfolio.as_of
            )
        ]
        short = sum((s.gain for s in lot_sales if s.term is Term.SHORT), start=Decimal(0))
        long = sum((s.gain for s in lot_sales if s.term is Term.LONG), start=Decimal(0))
        netted = net_capital_gains(short, long)
        short_rate = profile.marginal_tax_rate
        long_rate = long_term_rate(short_rate)
        return TaxImpact(
            client_id=portfolio.client_id,
            as_of=portfolio.as_of,
            lot_sales=tuple(lot_sales),
            short_term_gain=short,
            long_term_gain=long,
            short_term_rate=short_rate,
            long_term_rate=long_rate,
            estimated_tax=_to_cents(
                Fraction(netted.short_term * short_rate + netted.long_term * long_rate)
            ),
            net_capital_loss=netted.loss,
            wash_sales=tuple(_wash_sales(lots, left, orders, lot_sales, portfolio.as_of)),
        )


def _relieve(
    order: Order, lots: tuple[TaxLot, ...], left: list[Decimal], sold_on: date
) -> Iterator[LotSale]:
    to_sell = order.quantity
    # A stable sort keeps the portfolio's oldest-first lot order among lots of equal cost.
    for index in sorted(range(len(lots)), key=lambda i: -_cost_per_share(lots[i])):
        quantity = min(to_sell, left[index])
        if quantity == 0:
            continue
        lot = lots[index]
        left[index] -= quantity
        to_sell -= quantity
        yield LotSale(
            security=order.security,
            acquired_on=lot.acquired_on,
            quantity=quantity,
            cost_basis=_to_cents(_cost_per_share(lot) * Fraction(quantity)),
            proceeds=quantity * order.price,
            term=Term.LONG if sold_on > _anniversary(lot.acquired_on) else Term.SHORT,
        )
    if to_sell > 0:
        raise TaxError(
            f"sells {order.quantity} {order.security.symbol}, "
            f"but its lots hold only {order.quantity - to_sell}"
        )


def _cost_per_share(lot: TaxLot) -> Fraction:
    return Fraction(lot.cost_basis) / Fraction(lot.quantity)


def _anniversary(day: date) -> date:
    try:
        return day.replace(year=day.year + 1)
    except ValueError:
        return day.replace(year=day.year + 1, day=28)


def _to_cents(amount: Fraction) -> Decimal:
    return Decimal(round(amount * 100)) / 100


def _wash_sales(
    lots: dict[Security, tuple[TaxLot, ...]],
    left: dict[Security, list[Decimal]],
    orders: Sequence[Order],
    lot_sales: list[LotSale],
    sold_on: date,
) -> Iterator[WashSaleWarning]:
    """A loss may be disallowed if shares of the same security were bought within 30 days of the
    sale and are still held, or are bought by the same orders."""
    window_opens = sold_on - WASH_SALE_WINDOW
    losing = dict.fromkeys(sale.security for sale in lot_sales if sale.gain < 0)
    for security in losing:
        loss = -sum(
            (s.gain for s in lot_sales if s.security == security and s.gain < 0), start=Decimal(0)
        )
        recent = sorted(
            lot.acquired_on
            for lot, remaining in zip(lots.get(security, ()), left.get(security, []), strict=True)
            if remaining > 0 and lot.acquired_on >= window_opens
        )
        if recent:
            reason = f"{security.symbol} bought on {recent[-1]} is still held"
        elif any(order.side is Side.BUY and order.security == security for order in orders):
            reason = f"the same orders also buy {security.symbol}"
        else:
            continue
        yield WashSaleWarning(symbol=security.symbol, loss_at_risk=loss, reason=reason)
