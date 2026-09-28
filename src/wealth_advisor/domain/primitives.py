from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints


def _normalise_symbol(value: object) -> object:
    return value.strip().upper() if isinstance(value, str) else value


# Precision limits double as a float guard: noise such as 0.1 + 0.2 has too many decimal places.
Money = Annotated[Decimal, Field(ge=0, decimal_places=2)]
Quantity = Annotated[Decimal, Field(gt=0, decimal_places=6)]
Price = Annotated[Decimal, Field(gt=0, decimal_places=4)]
Weight = Annotated[Decimal, Field(ge=0, le=1, decimal_places=4)]
TaxRate = Annotated[Decimal, Field(ge=0, lt=1, decimal_places=4)]

Symbol = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][A-Z0-9.\-]{0,9}$"),
    BeforeValidator(_normalise_symbol),
]
ClientId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")]


class DomainModel(BaseModel):
    """Immutable, and an unknown field is an error rather than silently dropped."""

    model_config = ConfigDict(frozen=True, extra="forbid")
