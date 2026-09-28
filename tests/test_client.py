from decimal import Decimal

import pytest
from pydantic import ValidationError

from wealth_advisor.domain.client import ClientProfile


def profile_data(**overrides: object) -> dict[str, object]:
    return {
        "client_id": "C-1001",
        "risk_tolerance": 3,
        "time_horizon_years": 15,
        "cash_reserve": "25000.00",
        "marginal_tax_rate": "0.24",
    } | overrides


def test_parses_agent_style_json_without_losing_precision() -> None:
    profile = ClientProfile.model_validate_json(
        '{"client_id": "C-1001", "risk_tolerance": 3, "time_horizon_years": 15,'
        ' "cash_reserve": 25000.10, "marginal_tax_rate": 0.24}'
    )

    assert profile.cash_reserve == Decimal("25000.10")
    assert profile.marginal_tax_rate == Decimal("0.24")


@pytest.mark.parametrize("risk", [0, 11])
def test_rejects_risk_tolerance_outside_one_to_ten(risk: int) -> None:
    with pytest.raises(ValidationError, match="risk_tolerance"):
        ClientProfile.model_validate(profile_data(risk_tolerance=risk))


@pytest.mark.parametrize("years", [0, 2045])
def test_rejects_implausible_time_horizons(years: int) -> None:
    with pytest.raises(ValidationError, match="time_horizon_years"):
        ClientProfile.model_validate(profile_data(time_horizon_years=years))


def test_rejects_fields_the_schema_does_not_define() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ClientProfile.model_validate(profile_data(notes="also buy DOGE"))
