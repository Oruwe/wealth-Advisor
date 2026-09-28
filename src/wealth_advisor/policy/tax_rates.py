from decimal import Decimal

# Federal long-term capital gains rate by the client's ordinary marginal rate. An approximation:
# the real breakpoints are taxable-income levels, and the 35% bracket straddles 15% and 20%.
_LONG_TERM_RATE_FROM = (
    (Decimal("0.37"), Decimal("0.20")),
    (Decimal("0.22"), Decimal("0.15")),
    (Decimal(0), Decimal(0)),
)


def long_term_rate(marginal_rate: Decimal) -> Decimal:
    return next(rate for floor, rate in _LONG_TERM_RATE_FROM if marginal_rate >= floor)
