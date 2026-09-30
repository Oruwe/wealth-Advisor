import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, localcontext
from fractions import Fraction

from wealth_advisor.domain.primitives import EXACT_CONTEXT
from wealth_advisor.engine.substitutes import SubstituteRegistry, WashSaleTracker
from wealth_advisor.engine.tax import TaxJurisdiction, TaxLot, Trade, TradeAction

# Indian tax rates for foreign (US) equity assets.
# Short-term: held < 24 calendar months → taxed at slab rate (top bracket).
# Long-term:  held >= 24 calendar months → LTCG rate without indexation.
_INDIA_US_ST_RATE = Fraction(30, 100)
_INDIA_US_LT_RATE = Fraction(20, 100)


class WashSaleViolation(ValueError):
    """A buy would land on a symbol whose loss was just harvested in the same rebalance."""


class TradeGenerator:
    """Turn target weights into whole-share buy/sell trades with HIFO tax-loss harvesting."""

    def generate_trades(
        self,
        current_portfolio: Sequence[TaxLot],
        target_weights: Mapping[str, Decimal],
        portfolio_value: Decimal,
        current_prices: Mapping[str, Decimal],
        investor_jurisdiction: TaxJurisdiction = TaxJurisdiction.US,
        as_of: date | None = None,
        wash_sale_tracker: WashSaleTracker | None = None,
        substitute_registry: SubstituteRegistry | None = None,
    ) -> list[Trade]:
        if investor_jurisdiction == TaxJurisdiction.INDIA and as_of is None:
            raise ValueError("as_of is required when investor_jurisdiction is INDIA")

        with localcontext(EXACT_CONTEXT):
            lots_by_symbol: dict[str, list[TaxLot]] = defaultdict(list)
            for lot in current_portfolio:
                lots_by_symbol[lot.symbol].append(lot)

            current_value: dict[str, Decimal] = {}
            for symbol, lots in lots_by_symbol.items():
                if symbol not in current_prices:
                    raise ValueError(f"missing price for held symbol {symbol}")
                price = current_prices[symbol]
                current_value[symbol] = sum(
                    (lot.shares * price for lot in lots), start=Decimal(0)
                )

            sells: list[Trade] = []
            buys: list[Trade] = []
            losing_symbols: set[str] = set()

            for symbol in sorted(set(target_weights) | set(lots_by_symbol)):
                target_value = target_weights.get(symbol, Decimal(0)) * portfolio_value
                held_value = current_value.get(symbol, Decimal(0))
                diff = target_value - held_value
                if diff == 0:
                    continue
                if symbol not in current_prices:
                    raise ValueError(f"missing price for target symbol {symbol}")
                price = current_prices[symbol]

                if diff < 0:
                    trade, was_loss = self._plan_sell(
                        symbol,
                        -diff,
                        price,
                        lots_by_symbol.get(symbol, []),
                        investor_jurisdiction=investor_jurisdiction,
                        as_of=as_of,
                    )
                    if trade is None:
                        continue
                    sells.append(trade)
                    if was_loss:
                        losing_symbols.add(symbol)
                        if wash_sale_tracker is not None and as_of is not None:
                            wash_sale_tracker.record_harvest(symbol, as_of)
                else:
                    pre_locked = (
                        as_of is not None
                        and wash_sale_tracker is not None
                        and wash_sale_tracker.is_locked(symbol, as_of)
                    )
                    if symbol in losing_symbols or pre_locked:
                        registry = substitute_registry or SubstituteRegistry()
                        active_locks = (
                            wash_sale_tracker.get_active_locks(as_of)
                            if (wash_sale_tracker is not None and as_of is not None)
                            else set()
                        )
                        substitute = registry.get_substitute(
                            symbol, losing_symbols | active_locks
                        )
                        if substitute == "CASH":
                            continue
                        sub_price = current_prices.get(substitute)
                        if sub_price is None:
                            continue
                        shares = Decimal(math.floor(Fraction(diff) / Fraction(sub_price)))
                        if shares == 0:
                            continue
                        expiry = (
                            wash_sale_tracker.lock_expires_on(symbol)
                            if wash_sale_tracker is not None and pre_locked
                            else None
                        )
                        lock_suffix = f" (locked until {expiry})" if expiry else ""
                        routing_reason = (
                            f"Replaced {symbol}{lock_suffix} with {substitute} "
                            f"to preserve beta (0% cash drag)"
                        )
                        buys.append(
                            Trade(
                                symbol=substitute,
                                action=TradeAction.BUY,
                                shares=shares,
                                estimated_tax_impact=Decimal(0),
                                routing_reason=routing_reason,
                            )
                        )
                    else:
                        shares = Decimal(math.floor(Fraction(diff) / Fraction(price)))
                        if shares == 0:
                            continue
                        buys.append(
                            Trade(
                                symbol=symbol,
                                action=TradeAction.BUY,
                                shares=shares,
                                estimated_tax_impact=Decimal(0),
                            )
                        )

            return sells + buys

    @staticmethod
    def _plan_sell(
        symbol: str,
        dollar_delta: Decimal,
        price: Decimal,
        lots: Sequence[TaxLot],
        investor_jurisdiction: TaxJurisdiction = TaxJurisdiction.US,
        as_of: date | None = None,
    ) -> tuple[Trade | None, bool]:
        total_shares = sum((lot.shares for lot in lots), start=Decimal(0))
        if total_shares == 0:
            return None, False

        ideal = Fraction(dollar_delta) / Fraction(price)
        if ideal >= Fraction(total_shares):
            # Closing the position: a fractional final share is allowed.
            shares_to_sell = total_shares
        else:
            shares_to_sell = Decimal(math.floor(ideal))
            if shares_to_sell == 0:
                return None, False

        india_us_mode = investor_jurisdiction == TaxJurisdiction.INDIA and as_of is not None

        if india_us_mode:
            assert as_of is not None  # narrowing: india_us_mode guarantees this
            # Sort by tax-weighted gain per share ascending: most negative = best harvest first.
            sell_date = as_of
            ordered = sorted(
                lots,
                key=lambda lot: _india_us_tax_sort_key(lot, price, sell_date),
            )
        else:
            # HIFO: highest cost per share first; ties break to the older lot.
            ordered = sorted(
                lots,
                key=lambda lot: (
                    -Fraction(lot.cost_basis) / Fraction(lot.shares),
                    lot.purchase_date,
                ),
            )

        remaining = Fraction(shares_to_sell)
        realised = Fraction(0)
        for lot in ordered:
            if remaining <= 0:
                break
            take = min(Fraction(lot.shares), remaining)
            cost_per_share = Fraction(lot.cost_basis) / Fraction(lot.shares)
            gain_per_share = Fraction(price) - cost_per_share
            if india_us_mode and lot.asset_jurisdiction == TaxJurisdiction.US:
                assert as_of is not None
                rate = _india_us_rate(lot.purchase_date, as_of)
                realised += gain_per_share * take * rate
            else:
                realised += gain_per_share * take
            remaining -= take

        trade = Trade(
            symbol=symbol,
            action=TradeAction.SELL,
            shares=shares_to_sell,
            estimated_tax_impact=_to_cents(realised),
        )
        return trade, realised < 0


def _india_us_rate(purchase_date: date, as_of: date) -> Fraction:
    """Indian tax rate for a US-asset lot based on 24-month calendar holding period."""
    try:
        cutoff = purchase_date.replace(year=purchase_date.year + 2)
    except ValueError:
        # Feb 29 in a leap year: use Feb 28 two years later.
        cutoff = date(purchase_date.year + 2, 2, 28)
    return _INDIA_US_ST_RATE if as_of < cutoff else _INDIA_US_LT_RATE


def _india_us_tax_sort_key(lot: TaxLot, price: Decimal, as_of: date) -> Fraction:
    """Tax-weighted gain per share for an India-investor lot. Ascending → best loss harvest first."""
    cost_per_share = Fraction(lot.cost_basis) / Fraction(lot.shares)
    gain_per_share = Fraction(price) - cost_per_share
    if lot.asset_jurisdiction == TaxJurisdiction.US:
        return gain_per_share * _india_us_rate(lot.purchase_date, as_of)
    return gain_per_share


def _to_cents(amount: Fraction) -> Decimal:
    return Decimal(round(amount * 100)) / 100
