from decimal import Decimal
from unicodedata import lookup

import pytest

from wealth_advisor.agents.grounding import (
    first_number,
    first_signed_number,
    normalise,
    numbers_in,
    unsupported_numbers,
)


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        pytest.param("cash would be $10,170.00", {"10170"}, id="dollars-with-separators"),
        pytest.param("the 24% bracket", {"24"}, id="percent"),
        pytest.param("keep $25k, or 1.5 million", {"25000", "1500000"}, id="k-and-million"),
        pytest.param("as of 2026-09-25", {"2026", "9", "25"}, id="iso-date"),
        pytest.param("client C-1001", {"1001"}, id="identifier"),
        pytest.param("for 3 months", {"3"}, id="m-starting-a-word-is-not-million"),
        pytest.param("his 1st goal", set(), id="ordinal-is-not-a-number"),
        pytest.param("a loss of -$2,300.00", {"2300"}, id="sign-is-ignored"),
        pytest.param("a .5% fee, item No.7", {"0.5", "7"}, id="leading-decimal-point"),
        pytest.param("in Q3, ticker ABC1", set(), id="digits-inside-a-word-are-not-numbers"),
    ],
)
def test_finds_every_number_as_its_value(text: str, numbers: set[str]) -> None:
    assert numbers_in(text) == {Decimal(number) for number in numbers}


def test_19_8_and_19_80_are_the_same_number() -> None:
    assert numbers_in("19.8%") == numbers_in("19.80%")


def test_normalising_ignores_case_spacing_and_curly_punctuation() -> None:
    curly = (
        f"Client{lookup('RIGHT SINGLE QUOTATION MARK')}s  risk \n tolerance {lookup('EN DASH')} 3"
    )

    assert normalise(curly) == "client's risk tolerance - 3"


def test_reports_each_made_up_number_once_in_the_order_written() -> None:
    assert unsupported_numbers("sell 34 for $7, then $5, then $7 again", {Decimal(34)}) == [
        "7",
        "5",
    ]


def test_skips_list_markers_only_at_the_start_of_a_line() -> None:
    draft = "1. Sell 34 VTI.\n2) The fee is $7. That is all."

    assert unsupported_numbers(draft, {Decimal(34)}) == ["7"]


@pytest.mark.parametrize(
    ("text", "first"),
    [
        pytest.param("Risk tolerance: 3 on a scale of 1 to 10.", Decimal(3), id="scale"),
        pytest.param("keep $25k in cash", Decimal(25_000), id="suffix"),
        pytest.param("about ten years", None, id="words"),
    ],
)
def test_finds_the_first_number(text: str, first: Decimal | None) -> None:
    assert first_number(text) == first


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Risk tolerance: 7", Decimal(7), id="positive-no-sign"),
        pytest.param("score is -7", Decimal(-7), id="minus-before-digit"),
        pytest.param("loss of -$2,300.00", Decimal(-2300), id="minus-before-dollar"),
        pytest.param("balance: -$25k", Decimal(-25_000), id="minus-dollar-suffix"),
        pytest.param("about ten years", None, id="spelled-out-word-is-none"),
        pytest.param("value $7", Decimal(7), id="dollar-no-minus-is-positive"),
    ],
)
def test_first_signed_number_respects_minus_sign(text: str, expected: Decimal | None) -> None:
    assert first_signed_number(text) == expected


def test_sign_mismatch_means_negative_quote_does_not_match_positive_value() -> None:
    # A quote that says "-3" cannot prove that the value is +3.
    assert first_signed_number("risk tolerance: -3 on a scale") == Decimal(-3)
    assert first_signed_number("risk tolerance: -3 on a scale") != Decimal(3)
