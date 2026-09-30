"""Cross-border compliance: LRS limits, TCS tax, and UCITS estate-tax shielding.

Applies to Indian investors under FEMA / LRS and exposed to US estate-tax rules.
All arithmetic uses Decimal; no floats cross this module's boundary.
"""

from decimal import Decimal
from typing import Any

# US estate-tax exemption for non-resident aliens (Section 2101).
_US_ESTATE_TAX_THRESHOLD = Decimal("60000")

# FEMA / LRS annual remittance cap per individual (USD).
_LRS_LIMIT = Decimal("250000")

# TCS-free threshold: ₹10 lakh per FY at ≈ ₹83/USD.
_TCS_THRESHOLD = Decimal("12000")

# Section 206C(1G): TCS rate on LRS remittances above the threshold.
_TCS_RATE = Decimal("0.20")

# Ireland-domiciled UCITS equivalents for US-listed ETFs.
# Holding UCITS instead of the US security sidesteps the $60k NRA estate-tax trap.
UCITSRegistry: dict[str, str] = {
    "VOO": "VUAA",
    "SPY": "CSPX",
    "QQQ": "CNDX",
    "VTI": "VWRA",
    "VEA": "IWDA",
    "AGG": "AGGU",
    "BND": "AGGU",
}


class CrossBorderEngine:
    """Evaluate LRS / TCS obligations and US estate-tax exposure for Indian investors."""

    def evaluate_estate_tax_risk(
        self,
        client_jurisdiction: str,
        proposed_us_assets_value: Decimal,
    ) -> tuple[bool, str]:
        """Return (risk_flag, message) when proposed US holdings breach the $60k NRA threshold."""
        if client_jurisdiction == "INDIA" and proposed_us_assets_value > _US_ESTATE_TAX_THRESHOLD:
            return (
                True,
                "US Estate Tax Trap ($60k threshold) detected. "
                "Substituting with Ireland-domiciled UCITS proxies.",
            )
        return False, ""

    def calculate_lrs_tcs(
        self,
        remitted_ytd_usd: Decimal,
        new_trade_usd: Decimal,
    ) -> dict[str, Any]:
        """Calculate LRS utilisation and incremental TCS liability for a new remittance.

        TCS is charged on the portion of the new trade that pushes total FY remittances
        above ₹10 lakh (≈ $12,000).  It is an advance-tax credit, not a final tax.
        """
        total = remitted_ytd_usd + new_trade_usd
        lrs_breach = total > _LRS_LIMIT
        lrs_remaining = max(_LRS_LIMIT - remitted_ytd_usd, Decimal(0))

        # Incremental taxable = new amount that crosses the TCS threshold this trade.
        already_taxable = max(remitted_ytd_usd - _TCS_THRESHOLD, Decimal(0))
        total_taxable = max(total - _TCS_THRESHOLD, Decimal(0))
        incremental_taxable = total_taxable - already_taxable
        tcs_tax_usd = (incremental_taxable * _TCS_RATE).quantize(Decimal("0.01"))

        return {
            "lrs_breach": lrs_breach,
            "lrs_utilized_usd": total,
            "lrs_remaining_usd": lrs_remaining,
            "tcs_tax_usd": tcs_tax_usd,
            "tcs_rate_pct": int(_TCS_RATE * 100),
            "new_trade_usd": new_trade_usd,
            "remitted_ytd_usd": remitted_ytd_usd,
        }
