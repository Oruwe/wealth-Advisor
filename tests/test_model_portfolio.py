from decimal import Decimal
from itertools import pairwise

from wealth_advisor.domain.portfolio import AssetClass
from wealth_advisor.policy.model_portfolio import MODEL_SECURITIES, TARGET_BY_RISK


def test_every_risk_score_has_a_target() -> None:
    assert sorted(TARGET_BY_RISK) == list(range(1, 11))


def test_equity_never_falls_and_cash_never_rises_as_risk_rises() -> None:
    for lower, higher in pairwise(TARGET_BY_RISK[score] for score in range(1, 11)):
        assert higher.equity >= lower.equity
        assert higher.cash <= lower.cash


def test_conservative_scores_hold_at_most_twenty_percent_equity() -> None:
    assert all(TARGET_BY_RISK[score].equity <= Decimal("0.20") for score in (1, 2, 3))


def test_each_invested_asset_class_has_one_model_security_of_that_class() -> None:
    assert set(MODEL_SECURITIES) == {
        AssetClass.EQUITY,
        AssetClass.FIXED_INCOME,
        AssetClass.COMMODITY,
    }
    assert all(security.asset_class is cls for cls, security in MODEL_SECURITIES.items())
