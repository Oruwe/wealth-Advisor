from collections import Counter
from importlib.resources import files
from typing import Annotated, Self

import yaml
from pydantic import Field, field_validator, model_validator

from wealth_advisor.domain.portfolio import AssetClass
from wealth_advisor.domain.primitives import DomainModel, Symbol, Weight

RiskScore = Annotated[int, Field(ge=1, le=10)]


class RiskBand(DomainModel):
    name: str = Field(min_length=1)
    risk_scores: frozenset[RiskScore] = Field(min_length=1)
    max_weight: dict[AssetClass, Weight]

    @field_validator("max_weight")
    @classmethod
    def _cash_is_not_capped(cls, caps: dict[AssetClass, Weight]) -> dict[AssetClass, Weight]:
        if AssetClass.CASH in caps:
            raise ValueError("cash cannot be capped; the IPS cash reserve sets its floor")
        return caps


class SuitabilityPolicy(DomainModel):
    approved_securities: frozenset[Symbol] = Field(min_length=1)
    risk_bands: tuple[RiskBand, ...]

    @model_validator(mode="after")
    def _each_score_in_exactly_one_band(self) -> Self:
        counts = Counter(score for band in self.risk_bands for score in band.risk_scores)
        misplaced = [score for score in range(1, 11) if counts[score] != 1]
        if misplaced:
            raise ValueError(f"risk scores {misplaced} must each belong to exactly one band")
        return self

    def band_for(self, risk_score: int) -> RiskBand:
        return next(band for band in self.risk_bands if risk_score in band.risk_scores)


def parse_policy(text: str) -> SuitabilityPolicy:
    return SuitabilityPolicy.model_validate(yaml.safe_load(text))


SUITABILITY_POLICY = parse_policy(
    files("wealth_advisor.policy").joinpath("suitability.yaml").read_text(encoding="utf-8")
)
