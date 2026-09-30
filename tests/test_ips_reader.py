from typing import Any

import pytest
from examples import profile
from fakes import FakeReader, faithful_reading

from wealth_advisor.agents.ips_reader import (
    GroundedProfile,
    IpsReadingError,
    _IPS_END,
    _IPS_START,
    read_ips,
)
from wealth_advisor.demo import DEMO_IPS

IPS_MESSAGE = f"{_IPS_START}\n{DEMO_IPS}\n{_IPS_END}"
SCALE_QUOTE = "Risk tolerance: 3 on a scale of 1 to 10."


def problems_reading(ips: str, **changes: Any) -> list[str]:
    with pytest.raises(IpsReadingError) as error:
        read_ips(ips, "C-1001", FakeReader(faithful_reading(**changes)))
    return error.value.problems


def test_turns_a_grounded_reading_into_the_client_profile_and_its_quotes() -> None:
    reader = FakeReader(faithful_reading())

    assert read_ips(DEMO_IPS, "C-1001", reader) == GroundedProfile(
        profile(),
        quotes={
            "risk_tolerance": "Risk tolerance: 3 on a scale of 1 to 10.",
            "time_horizon_years": "Time horizon: 10 years.",
            "cash_reserve": "keep at least $2,000 in cash at all times",
            "marginal_tax_rate": "the client is in the 24% federal income-tax bracket",
        },
    )
    assert reader.messages == [IPS_MESSAGE]


def test_rejects_a_quote_the_ips_does_not_contain() -> None:
    invented = "Risk tolerance: 7"

    assert problems_reading(DEMO_IPS, risk_tolerance=7, risk_tolerance_quote=invented) == [
        f"risk_tolerance: the quote is not in the IPS: {invented!r}"
    ]


def test_a_scale_in_the_quote_cannot_pass_for_the_score() -> None:
    assert problems_reading(DEMO_IPS, risk_tolerance=10) == [
        f"risk_tolerance: the first number in the quote is 3, not 10: {SCALE_QUOTE!r}"
    ]


def test_rejects_a_quote_about_another_fact() -> None:
    other = "A loss of more than 10% in one year"

    assert problems_reading(DEMO_IPS, risk_tolerance=10, risk_tolerance_quote=other) == [
        f"risk_tolerance: the quote does not mention risk: {other!r}"
    ]


def test_rejects_a_value_its_quote_does_not_state() -> None:
    assert problems_reading(DEMO_IPS, time_horizon_years=15) == [
        "time_horizon_years: the first number in the quote is 10, not 15: 'Time horizon: 10 years.'"
    ]


def test_numbers_written_in_words_do_not_count() -> None:
    words = "The client plans to retire in about ten years."

    assert problems_reading(DEMO_IPS, time_horizon_quote=words) == [
        f"time_horizon_years: the quote states no number: {words!r}"
    ]


def test_lists_every_fact_the_ips_leaves_out() -> None:
    missing = problems_reading(
        DEMO_IPS, cash_reserve="0", cash_reserve_quote="", marginal_tax_rate_quote=" "
    )

    assert missing == [
        "cash_reserve: the IPS does not state it",
        "marginal_tax_rate: the IPS does not state it",
    ]


def test_only_the_tax_rate_may_be_quoted_as_a_percent() -> None:
    assert problems_reading(DEMO_IPS, cash_reserve="20") == [
        "cash_reserve: the first number in the quote is 2000, not 20: "
        "'keep at least $2,000 in cash at all times'"
    ]


def test_matches_quotes_whatever_their_line_breaks_and_case() -> None:
    wrapped = "LIQUIDITY: keep at least $2,000 in cash\nat all times"
    reader = FakeReader(faithful_reading(cash_reserve_quote=wrapped))

    assert read_ips(DEMO_IPS, "C-1001", reader).profile == profile()


def test_accepts_amounts_written_with_a_k_suffix() -> None:
    ips = f"{DEMO_IPS}Update: keep $25k in cash instead."
    reader = FakeReader(faithful_reading(cash_reserve="25000", cash_reserve_quote="$25k in cash"))

    assert read_ips(ips, "C-1001", reader).profile.cash_reserve == 25_000


@pytest.mark.parametrize("written", ["two thousand", "2000 or 5000"])
def test_rejects_a_value_that_is_not_one_number(written: str) -> None:
    assert problems_reading(DEMO_IPS, cash_reserve=written) == [
        f"cash_reserve: {written!r} is not a number"
    ]


def test_retries_once_listing_the_problems() -> None:
    reader = FakeReader(faithful_reading(risk_tolerance=10), faithful_reading())

    assert read_ips(DEMO_IPS, "C-1001", reader).profile == profile()
    assert reader.messages[1] == (
        f"{IPS_MESSAGE}\n\nYour last reading had these problems:\n"
        f"- risk_tolerance: the first number in the quote is 3, not 10: {SCALE_QUOTE!r}\n"
        "Read the IPS again and fix them."
    )


def test_gives_up_when_the_retry_is_still_not_grounded() -> None:
    reader = FakeReader(faithful_reading(time_horizon_years=15))

    with pytest.raises(IpsReadingError, match="the first number in the quote is 10, not 15"):
        read_ips(DEMO_IPS, "C-1001", reader)
    assert len(reader.messages) == 2


def test_rejects_a_quote_whose_first_number_has_a_minus_sign() -> None:
    negative_quote = "risk tolerance is marked as -3 here"
    ips = f"{DEMO_IPS} Note: {negative_quote}."

    assert problems_reading(ips, risk_tolerance_quote=negative_quote) == [
        f"risk_tolerance: the first number in the quote is -3, not 3: {negative_quote!r}"
    ]


def test_a_grounded_but_invalid_fact_fails_the_profile_rules_without_a_retry() -> None:
    ips = f"{DEMO_IPS}Correction: risk tolerance 11."
    reader = FakeReader(
        faithful_reading(risk_tolerance=11, risk_tolerance_quote="risk tolerance 11")
    )

    with pytest.raises(IpsReadingError) as error:
        read_ips(ips, "C-1001", reader)
    assert error.value.problems == ["risk_tolerance: Input should be less than or equal to 10"]
    assert len(reader.messages) == 1
