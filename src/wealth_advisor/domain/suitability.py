from datetime import date
from enum import StrEnum

from wealth_advisor.domain.primitives import ClientId, DomainModel


class Rule(StrEnum):
    SAME_CLIENT = "same_client"
    SAME_DATE = "same_date"
    SNAPSHOT_PRICES = "snapshot_prices"
    NO_OVERSELLING = "no_overselling"
    MATCHING_FIGURES = "matching_figures"
    APPROVED_SECURITIES = "approved_securities"
    MAX_WEIGHT = "max_weight"
    CASH_RESERVE = "cash_reserve"


class Violation(DomainModel):
    rule: Rule
    detail: str


class SuitabilityReport(DomainModel):
    """The gate's verdict: only a proposal with no violations may reach the adviser."""

    client_id: ClientId
    as_of: date
    violations: tuple[Violation, ...]

    @property
    def approved(self) -> bool:
        return not self.violations
