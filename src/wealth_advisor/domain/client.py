from pydantic import Field

from wealth_advisor.domain.primitives import ClientId, DomainModel, Money, TaxRate


class ClientProfile(DomainModel):
    """The facts from a client's Investment Policy Statement (IPS) that proposals must respect."""

    client_id: ClientId
    risk_tolerance: int = Field(
        ge=1, le=10, description="1 = cannot accept losses, 10 = accepts large swings for growth"
    )
    time_horizon_years: int = Field(ge=1, le=100)
    cash_reserve: Money = Field(description="Cash that must stay uninvested for liquidity needs")
    marginal_tax_rate: TaxRate = Field(
        description="Federal marginal income-tax rate as a fraction, e.g. 0.24 for 24%"
    )
