import pytest
from pydantic import ValidationError

from wealth_advisor.policy.model_portfolio import MODEL_SECURITIES, TARGET_BY_RISK
from wealth_advisor.policy.suitability import SUITABILITY_POLICY, parse_policy

VALID_BANDS = """
risk_bands:
  - {name: all, risk_scores: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], max_weight: {equity: 0.5}}
"""


def test_maps_every_risk_score_to_a_named_band() -> None:
    names = [SUITABILITY_POLICY.band_for(score).name for score in range(1, 11)]

    assert names == ["conservative"] * 3 + ["moderate"] * 4 + ["aggressive"] * 3


def test_approves_every_model_security() -> None:
    model_symbols = {security.symbol for security in MODEL_SECURITIES.values()}

    assert model_symbols <= SUITABILITY_POLICY.approved_securities


@pytest.mark.parametrize("risk_score", range(1, 11))
def test_every_model_target_is_within_its_bands_caps(risk_score: int) -> None:
    target = TARGET_BY_RISK[risk_score].weights

    for asset_class, cap in SUITABILITY_POLICY.band_for(risk_score).max_weight.items():
        assert target[asset_class] <= cap


@pytest.mark.parametrize(
    "bands",
    [
        pytest.param(
            "[{name: low, risk_scores: [1, 2, 3], max_weight: {}}]", id="scores-4-to-10-uncovered"
        ),
        pytest.param(
            "[{name: a, risk_scores: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], max_weight: {}},"
            " {name: b, risk_scores: [3], max_weight: {}}]",
            id="score-3-in-two-bands",
        ),
    ],
)
def test_rejects_bands_that_do_not_cover_each_score_exactly_once(bands: str) -> None:
    with pytest.raises(ValidationError, match="must each belong to exactly one band"):
        parse_policy(f"approved_securities: [VTI]\nrisk_bands: {bands}")


def test_rejects_a_cap_on_cash() -> None:
    with pytest.raises(ValidationError, match="cash cannot be capped"):
        parse_policy(
            "approved_securities: [VTI]\nrisk_bands:\n"
            "  - {name: all, risk_scores: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], max_weight: {cash: 0.1}}"
        )


def test_rejects_a_misspelled_key_instead_of_ignoring_it() -> None:
    with pytest.raises(ValidationError, match="approved_security"):
        parse_policy(f"approved_security: [VTI]\n{VALID_BANDS}")
