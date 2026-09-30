"""Custodian CSV ingestion: parse a brokerage statement into TaxLot objects.

Expected CSV headers: Symbol, Shares, Cost Basis, Purchase Date.
Rows with missing or unparseable critical fields are skipped silently so a
partially-corrupted export does not abort the entire import.
"""

from __future__ import annotations

import csv
import io
from datetime import date
from decimal import Decimal, InvalidOperation

from wealth_advisor.engine.tax import TaxJurisdiction, TaxLot

_DATE_FMTS = ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y")


def _parse_date(raw: str) -> date | None:
    raw = raw.strip()
    for fmt in _DATE_FMTS:
        try:
            return date.fromisoformat(raw) if fmt == "%Y-%m-%d" else _strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _strptime(raw: str, fmt: str) -> date:
    from datetime import datetime
    return datetime.strptime(raw, fmt).date()


def _parse_decimal(raw: str) -> Decimal | None:
    cleaned = raw.strip().replace(",", "").replace("$", "")
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def parse_brokerage_csv(csv_content: str, jurisdiction: TaxJurisdiction) -> list[TaxLot]:
    """Parse a custodian CSV export and return a list of TaxLot objects.

    Accepts headers: Symbol, Shares, Cost Basis, Purchase Date.
    Rows missing any critical field are skipped.
    """
    lots: list[TaxLot] = []
    reader = csv.DictReader(io.StringIO(csv_content))
    for row in reader:
        symbol = row.get("Symbol", "").strip().upper()
        if not symbol:
            continue

        shares = _parse_decimal(row.get("Shares", ""))
        cost_basis = _parse_decimal(row.get("Cost Basis", ""))
        purchase_date = _parse_date(row.get("Purchase Date", ""))

        if shares is None or cost_basis is None or purchase_date is None:
            continue
        if shares <= 0 or cost_basis < 0:
            continue

        lots.append(
            TaxLot(
                symbol=symbol,
                shares=shares,
                cost_basis=cost_basis,
                purchase_date=purchase_date,
                asset_jurisdiction=jurisdiction,
                investor_jurisdiction=jurisdiction,
            )
        )
    return lots
