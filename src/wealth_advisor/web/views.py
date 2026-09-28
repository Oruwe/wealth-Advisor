"""What the adviser console shows, computed from ledger records. No HTTP here."""

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from fractions import Fraction

from wealth_advisor.agents.briefing import briefing_facts
from wealth_advisor.domain.portfolio import AssetClass
from wealth_advisor.domain.tax import Term
from wealth_advisor.formatting import dollars, percent
from wealth_advisor.ledger import AdviceEntry, Decision, DecisionEntry, LedgerRecord, replay
from wealth_advisor.policy.suitability import SUITABILITY_POLICY

ASSET_CLASSES = {
    AssetClass.EQUITY: "Equity",
    AssetClass.FIXED_INCOME: "Fixed income",
    AssetClass.COMMODITY: "Commodities",
    AssetClass.CASH: "Cash",
}

# The allocation chart's geometry, in SVG units: 960 wide, so text renders near its set size.
CHART_WIDTH = 960.0
PLOT_LEFT, PLOT_RIGHT = 140.0, 880.0
ROW_HEIGHT, PLOT_TOP = 52.0, 18.0
# An after-weight label starts LABEL_GAP right of its dot and reaches LABEL_ROOM.
LABEL_GAP, LABEL_ROOM = 12.0, 72.0


@dataclass(frozen=True)
class Status:
    label: str
    tone: str  # good, warning or neutral: how the badge reads
    detail: str = ""


@dataclass(frozen=True)
class AdviceSummary:
    """One advice run, as the dashboard lists it."""

    sequence: int
    recorded_at: str
    hash: str
    client_id: str
    orders: int
    passed_gate: bool
    estimated_tax: str
    status: Status
    lyzr_session: str | None


@dataclass(frozen=True)
class Fact:
    label: str
    value: str
    quote: str


@dataclass(frozen=True)
class AllocationRow:
    label: str
    before: str
    after: str
    target: str
    cap: str
    before_value: str
    after_value: str
    y: float
    before_x: float
    after_x: float
    target_x: float
    cap_x: float | None
    label_x: float
    label_anchor: str  # start or end: the after-weight label sits right of its dot, or left


@dataclass(frozen=True)
class Tick:
    label: str
    x: float


@dataclass(frozen=True)
class LotSaleRow:
    symbol: str
    acquired_on: str
    shares: str
    cost_basis: str
    proceeds: str
    gain: str
    term: str


@dataclass(frozen=True)
class Dossier:
    """Everything the dossier page shows about one advice run."""

    record: LedgerRecord
    entry: AdviceEntry
    recorded_at: str
    facts: dict[str, object]
    ips_facts: list[Fact]
    portfolio_value_before: str
    allocation: list[AllocationRow]
    ticks: list[Tick]
    chart_width: float
    chart_height: float
    lot_sales: list[LotSaleRow]
    status: Status
    replay_problems: list[str]


def when(moment: datetime) -> str:
    return f"{moment:%Y-%m-%d %H:%M} UTC"


def decisions(records: list[LedgerRecord]) -> dict[str, tuple[LedgerRecord, DecisionEntry]]:
    """Each decided advice run's hash, with the record that decided it."""
    return {
        record.entry.advice_hash: (record, record.entry)
        for record in records
        if isinstance(record.entry, DecisionEntry)
    }


def status_of(advice_hash: str, records: list[LedgerRecord]) -> Status:
    decided = decisions(records).get(advice_hash)
    if decided is None:
        return Status("Awaiting decision", "warning")
    record, decision = decided
    note = f": {decision.note}" if decision.note else ""
    detail = f"by {decision.adviser_id} on {when(record.recorded_at)}{note}"
    if decision.decision is Decision.APPROVED:
        return Status("Approved", "good", detail)
    return Status("Declined", "neutral", detail)


def summaries(records: list[LedgerRecord]) -> list[AdviceSummary]:
    """Every advice run in the ledger, newest first."""
    return [
        AdviceSummary(
            sequence=record.sequence,
            recorded_at=when(record.recorded_at),
            hash=record.hash,
            client_id=record.entry.dossier.profile.client_id,
            orders=len(record.entry.dossier.proposal.orders),
            passed_gate=record.entry.dossier.suitability.approved,
            estimated_tax=dollars(record.entry.dossier.tax.estimated_tax),
            status=status_of(record.hash, records),
            lyzr_session=record.entry.provenance.session_id if record.entry.provenance else None,
        )
        for record in reversed(records)
        if isinstance(record.entry, AdviceEntry)
    ]


def dossier(record: LedgerRecord, records: list[LedgerRecord]) -> Dossier:
    """The dossier page for an advice record."""
    entry = record.entry
    if not isinstance(entry, AdviceEntry):
        raise ValueError(f"record {record.sequence} is not an advice run")
    inputs, advice = entry.inputs, entry.dossier
    before = inputs.portfolio.value_by_asset_class(inputs.prices)
    allocation, ticks = _allocation(entry, before)
    return Dossier(
        record=record,
        entry=entry,
        recorded_at=when(record.recorded_at),
        facts=briefing_facts(advice.profile, advice.proposal, advice.suitability, advice.tax),
        ips_facts=_ips_facts(entry),
        portfolio_value_before=dollars(sum(before.values(), start=Decimal(0))),
        allocation=allocation,
        ticks=ticks,
        chart_width=CHART_WIDTH,
        chart_height=PLOT_TOP + ROW_HEIGHT * len(allocation) + 28,
        lot_sales=[
            LotSaleRow(
                symbol=sale.security.symbol,
                acquired_on=sale.acquired_on.isoformat(),
                shares=format(sale.quantity.normalize(), "f"),
                cost_basis=dollars(sale.cost_basis),
                proceeds=dollars(sale.proceeds),
                gain=dollars(sale.gain),
                term="Long-term" if sale.term is Term.LONG else "Short-term",
            )
            for sale in advice.tax.lot_sales
        ],
        status=status_of(record.hash, records),
        replay_problems=replay(entry),
    )


def _ips_facts(entry: AdviceEntry) -> list[Fact]:
    profile, quotes = entry.dossier.profile, entry.dossier.ips_quotes
    values = {
        "risk_tolerance": ("Risk tolerance", f"{profile.risk_tolerance} of 10"),
        "time_horizon_years": ("Time horizon", f"{profile.time_horizon_years} years"),
        "cash_reserve": ("Cash reserve", dollars(profile.cash_reserve)),
        "marginal_tax_rate": ("Marginal tax rate", percent(Fraction(profile.marginal_tax_rate))),
    }
    return [Fact(label, value, quotes.get(name, "")) for name, (label, value) in values.items()]


def _allocation(
    entry: AdviceEntry, before: dict[AssetClass, Decimal]
) -> tuple[list[AllocationRow], list[Tick]]:
    proposal = entry.dossier.proposal
    band = SUITABILITY_POLICY.band_for(entry.dossier.profile.risk_tolerance)
    total_before = sum(before.values(), start=Decimal(0))
    total_after = sum(proposal.value_after.values(), start=Decimal(0))
    shares = {
        asset_class: (
            _share(before[asset_class], total_before),
            _share(proposal.value_after[asset_class], total_after),
            Fraction(proposal.target.weights[asset_class]),
            Fraction(band.max_weight[asset_class]) if asset_class in band.max_weight else None,
        )
        for asset_class in AssetClass
    }
    largest = max(share for row in shares.values() for share in row if share is not None)
    axis_max = Fraction(max(1, math.ceil(largest * 10)), 10)

    def x(share: Fraction) -> float:
        return round(PLOT_LEFT + float(share / axis_max) * (PLOT_RIGHT - PLOT_LEFT), 1)

    rows = []
    for index, (asset_class, (before_share, after_share, target, cap)) in enumerate(shares.items()):
        after_x, cap_x = x(after_share), x(cap) if cap is not None else None
        # A cap line where the label would sit would run through it, so the label goes left.
        crowded = cap_x is not None and after_x + LABEL_GAP - 4 <= cap_x <= after_x + LABEL_ROOM
        rows.append(
            AllocationRow(
                label=ASSET_CLASSES[asset_class],
                before=percent(before_share),
                after=percent(after_share),
                target=percent(target),
                cap=percent(cap) if cap is not None else "none",
                before_value=dollars(before[asset_class]),
                after_value=dollars(proposal.value_after[asset_class]),
                y=PLOT_TOP + ROW_HEIGHT * index + ROW_HEIGHT / 2,
                before_x=x(before_share),
                after_x=after_x,
                target_x=x(target),
                cap_x=cap_x,
                label_x=after_x - LABEL_GAP if crowded else after_x + LABEL_GAP,
                label_anchor="end" if crowded else "start",
            )
        )
    tenths = int(axis_max * 10)
    ticks = [Tick(f"{step * 10}%", x(Fraction(step, 10))) for step in range(tenths + 1)]
    return rows, ticks


def _share(value: Decimal, total: Decimal) -> Fraction:
    return Fraction(value) / Fraction(total) if total else Fraction(0)
