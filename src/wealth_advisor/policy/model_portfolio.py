from collections.abc import Mapping
from decimal import Decimal
from types import MappingProxyType

from wealth_advisor.domain.portfolio import AssetClass, Security, TargetAllocation


def _mix(equity: str, fixed_income: str, commodity: str, cash: str) -> TargetAllocation:
    return TargetAllocation(
        equity=Decimal(equity),
        fixed_income=Decimal(fixed_income),
        commodity=Decimal(commodity),
        cash=Decimal(cash),
    )


# Illustrative glide path: equity rises and cash falls with risk tolerance, and the conservative
# scores (1-3) hold at most 20% equity.
TARGET_BY_RISK: Mapping[int, TargetAllocation] = MappingProxyType(
    {
        #  equity, fixed income, commodity, cash
        1: _mix("0.10", "0.65", "0.05", "0.20"),
        2: _mix("0.15", "0.65", "0.05", "0.15"),
        3: _mix("0.20", "0.65", "0.05", "0.10"),
        4: _mix("0.35", "0.55", "0.05", "0.05"),
        5: _mix("0.45", "0.45", "0.05", "0.05"),
        6: _mix("0.55", "0.35", "0.05", "0.05"),
        7: _mix("0.65", "0.25", "0.05", "0.05"),
        8: _mix("0.75", "0.15", "0.05", "0.05"),
        9: _mix("0.85", "0.07", "0.05", "0.03"),
        10: _mix("0.90", "0.03", "0.05", "0.02"),
    }
)

# The one security each invested asset class is bought through; cash stays uninvested.
MODEL_SECURITIES: Mapping[AssetClass, Security] = MappingProxyType(
    {
        AssetClass.EQUITY: Security(symbol="VTI", asset_class=AssetClass.EQUITY),
        AssetClass.FIXED_INCOME: Security(symbol="BND", asset_class=AssetClass.FIXED_INCOME),
        AssetClass.COMMODITY: Security(symbol="GLD", asset_class=AssetClass.COMMODITY),
    }
)
