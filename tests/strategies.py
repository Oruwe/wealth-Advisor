from datetime import date
from decimal import Decimal

from hypothesis import strategies as st

from wealth_advisor.domain.portfolio import AssetClass, Holding, Portfolio, Security, TaxLot

symbols = st.from_regex(r"[A-Z]{1,5}", fullmatch=True)
money = st.decimals(min_value=0, max_value=Decimal("1e9"), places=2)
quantities = st.decimals(min_value=Decimal("0.000001"), max_value=Decimal("1e6"), places=6)
dates = st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31))


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
