from decimal import Decimal
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from wealth_advisor.domain.primitives import (
    ClientId,
    Money,
    Price,
    Quantity,
    Symbol,
    TaxRate,
    Weight,
)


@pytest.mark.parametrize(
    ("type_", "value"),
    [
        pytest.param(Money, "0", id="money-zero"),
        pytest.param(Money, "1234.56", id="money-cents"),
        pytest.param(Money, "10.500", id="money-trailing-zero"),
        pytest.param(Quantity, "0.000001", id="quantity-fractional-share"),
        pytest.param(Price, "412.3456", id="price-sub-cent"),
        pytest.param(Weight, "0", id="weight-zero"),
        pytest.param(Weight, "1", id="weight-one"),
        pytest.param(TaxRate, "0.37", id="tax-rate-top-bracket"),
    ],
)
def test_accepts_values_within_range_and_precision(type_: Any, value: str) -> None:
    assert TypeAdapter(type_).validate_python(value) == Decimal(value)


@pytest.mark.parametrize(
    ("type_", "value"),
    [
        pytest.param(Money, "-0.01", id="money-negative"),
        pytest.param(Money, "10.005", id="money-fraction-of-a-cent"),
        pytest.param(Money, "NaN", id="money-nan"),
        pytest.param(Quantity, "0", id="quantity-zero"),
        pytest.param(Quantity, "1.0000001", id="quantity-too-precise"),
        pytest.param(Price, "0", id="price-zero"),
        pytest.param(Price, "Infinity", id="price-infinite"),
        pytest.param(Weight, "1.0001", id="weight-over-one"),
        pytest.param(TaxRate, "24", id="tax-rate-percent-not-fraction"),
    ],
)
def test_rejects_values_outside_range_or_precision(type_: Any, value: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(type_).validate_python(value)


def test_float_arithmetic_noise_cannot_become_money() -> None:
    with pytest.raises(ValidationError, match="no more than 2 decimal places"):
        TypeAdapter(Money).validate_python(0.1 + 0.2)


def test_symbols_are_trimmed_and_upper_cased() -> None:
    assert TypeAdapter(Symbol).validate_python(" brk.b ") == "BRK.B"


@pytest.mark.parametrize("value", ["", "VANGUARD TOTAL", "1ABC", "TOOLONGSYMBOL"])
def test_rejects_malformed_symbols(value: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Symbol).validate_python(value)


@pytest.mark.parametrize("value", ["", "John Smith", "-C1001"])
def test_rejects_malformed_client_ids(value: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ClientId).validate_python(value)
