import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from importlib.resources import files
from types import MappingProxyType
from typing import Annotated, Any, Self

import yaml
from pydantic import Field, field_validator, model_validator

from wealth_advisor.domain.portfolio import AssetClass
from wealth_advisor.domain.primitives import DomainModel, Symbol, Weight

RiskScore = Annotated[int, Field(ge=1, le=10)]


class RiskBand(DomainModel):
    name: str = Field(min_length=1)
    risk_scores: frozenset[RiskScore] = Field(min_length=1)
    # Typed as Mapping (read-only view); after validation the dict is wrapped in MappingProxyType.
    max_weight: Mapping[AssetClass, Weight]
    min_weight: Mapping[AssetClass, Weight] = Field(default_factory=dict)

    @field_validator("max_weight")
    @classmethod
    def _cash_is_not_capped(cls, caps: dict[AssetClass, Weight]) -> dict[AssetClass, Weight]:
        if AssetClass.CASH in caps:
            raise ValueError("cash cannot be capped; the IPS cash reserve sets its floor")
        return caps

    @model_validator(mode="after")
    def _freeze_weights(self) -> Self:
        # Replace the plain dicts Pydantic stored with immutable MappingProxyType so that
        # the frozen model is truly immutable (not just preventing attribute re-assignment).
        if not isinstance(self.max_weight, MappingProxyType):
            object.__setattr__(self, "max_weight", MappingProxyType(dict(self.max_weight)))
        if not isinstance(self.min_weight, MappingProxyType):
            object.__setattr__(self, "min_weight", MappingProxyType(dict(self.min_weight)))
        return self


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

    @property
    def policy_hash(self) -> str:
        """SHA-256 of the canonical serialised policy; stable across Python restarts."""
        data: dict[str, Any] = self.model_dump(mode="json")
        # frozenset serialises to a list whose order is non-deterministic; sort for stability.
        if isinstance(data.get("approved_securities"), list):
            data["approved_securities"] = sorted(data["approved_securities"])
        for band in data.get("risk_bands", []):
            if isinstance(band.get("risk_scores"), list):
                band["risk_scores"] = sorted(band["risk_scores"])
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def parse_policy(text: str) -> SuitabilityPolicy:
    data = yaml.safe_load(text)
    # YAML 1.1 parses bare "ON/OFF/YES/NO" as booleans; coerce approved_securities
    # to strings so tickers like "ON" (ON Semiconductor) are not silently misread.
    if isinstance(data, dict) and isinstance(data.get("approved_securities"), list):
        data["approved_securities"] = [str(s) for s in data["approved_securities"]]
    return SuitabilityPolicy.model_validate(data)


SUITABILITY_POLICY = parse_policy(
    files("wealth_advisor.policy").joinpath("suitability.yaml").read_text(encoding="utf-8")
)
