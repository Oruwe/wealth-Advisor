from datetime import date
from decimal import ROUND_DOWN, Decimal
from typing import NamedTuple

from hypothesis import strategies as st

from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.portfolio import (
    AssetClass,
    Holding,
    Portfolio,
    Security,
    TargetAllocation,
    TaxLot,
)
from wealth_advisor.policy.model_portfolio import MODEL_SECURITIES

symbols = st.from_regex(r"[A-Z]{1,5}", fullmatch=True)
money = st.decimals(min_value=0, max_value=Decimal("1e9"), places=2)
quantities = st.decimals(min_value=Decimal("0.000001"), max_value=Decimal("1e6"), places=6)
dates = st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31))

MODEL_SYMBOLS = frozenset(security.symbol for security in MODEL_SECURITIES.values())


@st.composite
def tax_lots(draw: st.DrawFn, acquired_by: date) -> TaxLot:
    return TaxLot(
        quantity=draw(quantities),
        cost_basis=draw(money),
        acquired_on=draw(st.dates(min_value=date(2000, 1, 1), max_value=acquired_by)),
    )


@st.composite
def portfolios(draw: st.DrawFn) -> Portfolio:
    as_of = draw(dates)
    holdings = tuple(
        Holding(
            security=Security(symbol=symbol, asset_class=draw(st.sampled_from(AssetClass))),
            lots=tuple(draw(st.lists(tax_lots(acquired_by=as_of), min_size=1, max_size=3))),
        )
        for symbol in draw(st.lists(symbols, max_size=6, unique=True))
    )
    return Portfolio(client_id="C-1001", as_of=as_of, cash=draw(money), holdings=holdings)


@st.composite
def target_allocations(draw: st.DrawFn) -> TargetAllocation:
    low, mid, high = sorted(draw(st.lists(st.integers(0, 10_000), min_size=3, max_size=3)))
    basis_points = (low, mid - low, high - mid, 10_000 - high)
    equity, fixed_income, commodity, cash = (Decimal(bp).scaleb(-4) for bp in basis_points)
    return TargetAllocation(
        equity=equity, fixed_income=fixed_income, commodity=commodity, cash=cash
    )


class RebalanceCase(NamedTuple):
    portfolio: Portfolio
    prices: PriceSnapshot
    target: TargetAllocation
    cash_reserve: Decimal


@st.composite
def rebalance_cases(draw: st.DrawFn, *, whole_shares: bool = False) -> RebalanceCase:
    """Model and outside holdings in 1-3 tax lots each, a price for every symbol, any target, and
    a reserve that fits."""
    as_of = draw(dates)
    shares = (
        st.integers(1, 10_000).map(Decimal)
        if whole_shares
        else st.decimals(min_value=Decimal("0.000001"), max_value=Decimal(10_000), places=6)
    )
    price = st.decimals(
        min_value=Decimal("0.01"), max_value=Decimal(2_000), places=2 if whole_shares else 4
    )
    lots = st.lists(
        st.builds(
            TaxLot,
            quantity=shares,
            cost_basis=st.decimals(min_value=0, max_value=Decimal(10_000_000), places=2),
            acquired_on=st.dates(min_value=date(2000, 1, 1), max_value=as_of),
        ),
        min_size=1,
        max_size=3,
    )
    outside = draw(
        st.lists(symbols.filter(lambda s: s not in MODEL_SYMBOLS), max_size=4, unique=True)
    )
    securities = [
        *draw(st.lists(st.sampled_from(list(MODEL_SECURITIES.values())), unique=True)),
        *(Security(symbol=s, asset_class=draw(st.sampled_from(AssetClass))) for s in outside),
    ]
    portfolio = Portfolio(
        client_id="C-1001",
        as_of=as_of,
        cash=draw(st.decimals(min_value=0, max_value=Decimal(10_000_000), places=2)),
        holdings=tuple(
            Holding(security=security, lots=tuple(draw(lots))) for security in securities
        ),
    )
    prices = PriceSnapshot(
        as_of=as_of,
        prices={symbol: draw(price) for symbol in sorted(MODEL_SYMBOLS | set(outside))},
    )
    total = portfolio.market_value(prices)
    reserve = (total * draw(st.integers(0, 100)) / 100).quantize(Decimal("0.01"), ROUND_DOWN)
    return RebalanceCase(portfolio, prices, draw(target_allocations()), reserve)
