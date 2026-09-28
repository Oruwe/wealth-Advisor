from decimal import Decimal
from fractions import Fraction


def dollars(amount: Decimal) -> str:
    return f"-${-amount:,.2f}" if amount < 0 else f"${amount:,.2f}"


def percent(fraction: Fraction) -> str:
    basis_points = round(fraction * 10_000)
    return f"{basis_points // 100}.{basis_points % 100:02d}%"
