from decimal import Decimal

import pytest

from wealth_advisor.policy.tax_rates import long_term_rate


@pytest.mark.parametrize(
    ("marginal", "long_term"),
    [
        ("0.10", "0"),
        ("0.12", "0"),
        ("0.22", "0.15"),
        ("0.24", "0.15"),
        ("0.25", "0.15"),
        ("0.35", "0.15"),
        ("0.37", "0.20"),
    ],
)
def test_maps_the_marginal_rate_to_a_long_term_rate(marginal: str, long_term: str) -> None:
    assert long_term_rate(Decimal(marginal)) == Decimal(long_term)
