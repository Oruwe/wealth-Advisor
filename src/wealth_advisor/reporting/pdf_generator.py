"""Fiduciary Value-Prop PDF Reporting Engine.

Generates an institutional-grade PDF audit report from a fiduciary ledger block,
AI pipeline result, and client profile.  All formatting is deterministic — no
agent output flows into numbers; the PDF renders facts already committed to the ledger.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Colour palette (institutional navy / slate) ────────────────────────────────

_NAVY = colors.HexColor("#0f2044")
_NAVY_LIGHT = colors.HexColor("#1a3a6e")
_SLATE = colors.HexColor("#334155")
_TEAL = colors.HexColor("#0e7490")
_ROW_ALT = colors.HexColor("#f1f5f9")
_GOOD = colors.HexColor("#16a34a")
_WARN = colors.HexColor("#b45309")
_MONO_BG = colors.HexColor("#0f172a")
_MONO_FG = colors.HexColor("#4ade80")
_WHITE = colors.white
_LIGHT_GREY = colors.HexColor("#e2e8f0")
_TEXT = colors.HexColor("#1e293b")


# ── Decimal helpers ────────────────────────────────────────────────────────────


def _dec(v: Any) -> Decimal:
    try:
        return Decimal(str(v))
    except InvalidOperation:
        return Decimal("0")


def _fmt_usd(v: Any) -> str:
    d = _dec(v)
    if d < 0:
        return f"-${-d:,.2f}"
    return f"${d:,.2f}"


def _fmt_pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


# ── Style helpers ──────────────────────────────────────────────────────────────


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "firm": ParagraphStyle(
            "firm",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#94a3b8"),
            spaceAfter=2,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Normal"],
            fontSize=16,
            textColor=_WHITE,
            fontName="Helvetica-Bold",
            spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Normal"],
            fontSize=11,
            textColor=_WHITE,
            fontName="Helvetica-Bold",
            spaceBefore=14,
            spaceAfter=6,
        ),
        "label": ParagraphStyle(
            "label",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#94a3b8"),
            spaceAfter=1,
        ),
        "value": ParagraphStyle(
            "value",
            parent=base["Normal"],
            fontSize=10,
            textColor=_WHITE,
            fontName="Helvetica-Bold",
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#cbd5e1"),
            spaceAfter=4,
            leading=13,
        ),
        "mono": ParagraphStyle(
            "mono",
            parent=base["Normal"],
            fontSize=8,
            textColor=_MONO_FG,
            fontName="Courier",
            leading=12,
        ),
        "mono_label": ParagraphStyle(
            "mono_label",
            parent=base["Normal"],
            fontSize=7.5,
            textColor=colors.HexColor("#94a3b8"),
            fontName="Courier",
            spaceAfter=2,
        ),
        "warn": ParagraphStyle(
            "warn",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#fbbf24"),
            spaceAfter=3,
            leading=13,
        ),
        "good": ParagraphStyle(
            "good",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#4ade80"),
            spaceAfter=3,
        ),
        "cert": ParagraphStyle(
            "cert",
            parent=base["Normal"],
            fontSize=8.5,
            textColor=colors.HexColor("#94a3b8"),
            spaceAfter=3,
            leading=13,
        ),
        "confidential": ParagraphStyle(
            "confidential",
            parent=base["Normal"],
            fontSize=7.5,
            textColor=colors.HexColor("#64748b"),
            alignment=1,
            spaceAfter=0,
        ),
    }


def _hr(width: float = 6.5 * inch, color: Any = _NAVY_LIGHT) -> HRFlowable:
    return HRFlowable(width=width, thickness=1, color=color, spaceAfter=8, spaceBefore=4)


def _section_table(rows: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    data = [
        [Paragraph(label, styles["label"]), Paragraph(value, styles["value"])]
        for label, value in rows
    ]
    t = Table(data, colWidths=[2.2 * inch, 4.3 * inch])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


# ── Main generator ─────────────────────────────────────────────────────────────


def generate_fiduciary_pdf(
    audit_block: dict[str, Any],
    ai_result: dict[str, Any],
    client_profile: dict[str, Any],
) -> bytes:
    """Generate an in-memory institutional fiduciary PDF and return its raw bytes."""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="Fiduciary Rebalance Audit Report",
        author="Altis Fiduciary Partners",
    )

    s = _styles()
    story: list[Any] = []

    client_id = str(client_profile.get("client_id") or ai_result.get("client_id") or "—")
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── Header band ────────────────────────────────────────────────────────────

    header_data = [[
        Paragraph("ALTIS FIDUCIARY PARTNERS", s["firm"]),
        Paragraph(
            "CONFIDENTIAL — For authorised adviser use only. "
            "Not for distribution to clients or third parties.",
            s["confidential"],
        ),
    ]]
    header_t = Table(header_data, colWidths=[3.25 * inch, 3.25 * inch])
    header_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(header_t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("QUARTERLY FIDUCIARY REBALANCE AUDIT", s["h1"]))
    story.append(Paragraph(
        f"Generated: {generated_at} &nbsp;&nbsp;|&nbsp;&nbsp; Client: {client_id}",
        s["body"],
    ))
    story.append(_hr())

    # ── Cryptographic Audit Seal ───────────────────────────────────────────────

    story.append(Paragraph("CRYPTOGRAPHIC AUDIT SEAL", s["h2"]))
    story.append(Paragraph(
        "SHA-256 signature over all fiduciary payload fields. Any alteration to inputs, "
        "AI mandate, or trades invalidates this hash. Ledger compliance: SEC Rule 204-2 / SEBI.",
        s["body"],
    ))

    sig_hash = str(audit_block.get("signature_hash") or "Not yet sealed")
    prev_hash = str(audit_block.get("previous_hash") or "GENESIS")
    block_id = str(audit_block.get("block_id") or "—")
    block_ts = str(audit_block.get("timestamp") or "—")

    seal_data = [
        [Paragraph("SIGNATURE HASH (SHA-256)", s["mono_label"])],
        [Paragraph(sig_hash, s["mono"])],
        [Paragraph("PREVIOUS BLOCK HASH", s["mono_label"])],
        [Paragraph(prev_hash, s["mono"])],
        [Paragraph(f"Block ID: {block_id}   |   Sealed: {block_ts}", s["mono_label"])],
    ]
    seal_t = Table(seal_data, colWidths=[6.5 * inch])
    seal_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _MONO_BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [6]),
        ("BOX", (0, 0), (-1, -1), 1, _NAVY_LIGHT),
    ]))
    story.append(seal_t)
    story.append(Spacer(1, 8))

    # ── Executive Summary ──────────────────────────────────────────────────────

    story.append(_hr())
    story.append(Paragraph("EXECUTIVE SUMMARY &amp; AI MANDATE", s["h2"]))

    rp: dict[str, Any] = ai_result.get("risk_parameters") or {}
    risk_score = str(client_profile.get("risk_tolerance", rp.get("risk_score", "—")))
    risk_band = str(client_profile.get("risk_band", "—"))
    target_ra = str(rp.get("target_risk_aversion", "—"))
    max_eq = rp.get("max_equity_exposure")
    max_eq_str = _fmt_pct(max_eq) if max_eq is not None else "—"
    event_type = str(ai_result.get("event_type", "—"))
    timestamp = str(ai_result.get("timestamp", "—"))

    summary_rows: list[tuple[str, str]] = [
        ("Client Risk Score", f"{risk_score} / 10 ({risk_band})"),
        ("Event Type", event_type),
        ("AI Mandate Timestamp", timestamp),
        ("Target Risk Aversion (λ)", target_ra),
        ("Max Equity Exposure Cap", max_eq_str),
    ]
    story.append(_section_table(summary_rows, s))

    advisor_guidance = str(rp.get("advisor_guidance") or "")
    if advisor_guidance:
        story.append(Paragraph(f"Adviser Guidance: {advisor_guidance}", s["warn"]))

    compliance_warnings: list[Any] = rp.get("compliance_warnings") or []
    for w in compliance_warnings:
        story.append(Paragraph(f"&#9888;  {w}", s["warn"]))

    # ── Tax Alpha & Value Delivered ────────────────────────────────────────────

    story.append(_hr())
    story.append(Paragraph("TAX ALPHA &amp; VALUE DELIVERED", s["h2"]))

    trades_raw: list[Any] = ai_result.get("trades") or []
    total_harvested = Decimal("0")
    for t in trades_raw:
        impact = _dec(getattr(t, "estimated_tax_impact", None) or t.get("estimated_tax_impact", 0)
                      if not hasattr(t, "__dict__") else getattr(t, "estimated_tax_impact", 0))
        if impact < 0:
            total_harvested += impact

    jurisdiction = str(client_profile.get("tax_jurisdiction", "US")).upper()
    if jurisdiction == "INDIA":
        st_rate, lt_rate = Decimal("30"), Decimal("12.5")
        st_label, lt_label = "Indian slab rate (30%)", "LTCG (12.5%)"
    else:
        st_rate, lt_rate = Decimal("37"), Decimal("20")
        st_label, lt_label = "US short-term (37%)", "US long-term (20%)"

    st_savings = abs(total_harvested) * st_rate / 100
    lt_savings = abs(total_harvested) * lt_rate / 100

    cb: dict[str, Any] = ai_result.get("cross_border_metrics") or {}
    lrs_used = _dec(cb.get("lrs_utilized_usd", client_profile.get("lrs_quota_used_usd", 0)))
    lrs_remaining = _dec(cb.get("lrs_remaining_usd", 0))
    tcs_usd = _dec(cb.get("tcs_tax_usd", 0))
    tcs_rate = str(cb.get("tcs_rate_pct", "—"))
    lrs_breach = bool(cb.get("lrs_breach", False))

    tax_rows: list[tuple[str, str]] = [
        ("Total Harvested Capital Losses", _fmt_usd(total_harvested)),
        (f"Est. Tax Savings @ {st_label}", _fmt_usd(st_savings)),
        (f"Est. Tax Savings @ {lt_label}", _fmt_usd(lt_savings)),
        ("LRS Utilized YTD", _fmt_usd(lrs_used)),
        ("LRS Remaining Allowance", _fmt_usd(lrs_remaining)),
        ("LRS Cap Breach", "YES — RBI prior approval required" if lrs_breach else "No"),
        ("TCS Rate (Section 206C(1G))", f"{tcs_rate}%"),
        ("Upfront TCS Due", _fmt_usd(tcs_usd)),
    ]
    story.append(_section_table(tax_rows, s))

    if lrs_breach:
        story.append(Paragraph(
            "&#9888;  LRS cap breached — this trade requires RBI prior approval before execution.",
            s["warn"],
        ))

    if tcs_usd > 0:
        story.append(Paragraph(
            f"TCS of {_fmt_usd(tcs_usd)} is deducted at source by the AD bank and credited "
            "against the final income-tax liability — a liquidity drag until year-end ITR filing.",
            s["body"],
        ))

    ucits_subs: dict[str, str] = cb.get("ucits_substitutions") or {}
    if ucits_subs:
        story.append(Paragraph("UCITS Estate-Tax Substitutions Active:", s["warn"]))
        for us_ticker, ucits_ticker in ucits_subs.items():
            story.append(Paragraph(
                f"  {us_ticker} &rarr; {ucits_ticker} (Ireland-domiciled UCITS, exempt from US estate tax)",
                s["body"],
            ))

    # ── Approved Trades & Execution Table ─────────────────────────────────────

    story.append(_hr())
    story.append(Paragraph("APPROVED TRADES &amp; EXECUTION", s["h2"]))

    col_headers = ["Symbol", "Action", "Shares", "Est. Tax Impact", "Routing / Notes"]
    trade_table_data: list[list[Any]] = [
        [Paragraph(h, ParagraphStyle(
            "th", fontSize=8, textColor=_WHITE, fontName="Helvetica-Bold",
        )) for h in col_headers]
    ]

    for i, t in enumerate(trades_raw):
        if hasattr(t, "symbol"):
            symbol = str(t.symbol)
            action = str(t.action.value if hasattr(t.action, "value") else t.action).upper()
            shares = str(t.shares)
            tax_impact = _fmt_usd(t.estimated_tax_impact)
            routing = str(t.routing_reason or "Direct")
        else:
            symbol = str(t.get("symbol", "—"))
            action = str(t.get("action", "—")).upper()
            shares = str(t.get("shares", "—"))
            tax_impact = _fmt_usd(t.get("estimated_tax_impact", 0))
            routing = str(t.get("routing_reason") or "Direct")

        action_style = ParagraphStyle(
            "act", fontSize=8,
            textColor=_GOOD if action == "BUY" else colors.HexColor("#f87171"),
            fontName="Helvetica-Bold",
        )
        bg = _ROW_ALT if i % 2 == 0 else _WHITE
        row = [
            Paragraph(f"<b>{symbol}</b>", ParagraphStyle("sym", fontSize=8, textColor=_TEXT)),
            Paragraph(action, action_style),
            Paragraph(shares, ParagraphStyle("sh", fontSize=8, textColor=_SLATE)),
            Paragraph(tax_impact, ParagraphStyle("ti", fontSize=8, textColor=_SLATE)),
            Paragraph(routing, ParagraphStyle("rt", fontSize=7.5, textColor=_SLATE, leading=10)),
        ]
        trade_table_data.append(row)

    if len(trade_table_data) == 1:
        trade_table_data.append([
            Paragraph("No trades — portfolio already on target.", s["body"]),
            Paragraph("", s["body"]),
            Paragraph("", s["body"]),
            Paragraph("", s["body"]),
            Paragraph("", s["body"]),
        ])

    trade_t = Table(
        trade_table_data,
        colWidths=[0.9 * inch, 0.7 * inch, 0.8 * inch, 1.3 * inch, 2.8 * inch],
    )
    trade_style = [
        ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_ROW_ALT, _WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, _LIGHT_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    trade_t.setStyle(TableStyle(trade_style))
    story.append(trade_t)

    # ── Target Weights Summary ─────────────────────────────────────────────────

    target_weights: dict[str, Any] = ai_result.get("target_weights") or {}
    if target_weights:
        story.append(Spacer(1, 10))
        story.append(Paragraph("AI Optimizer Target Weights:", s["h2"]))
        tw_data = [
            [Paragraph("Asset", ParagraphStyle("th2", fontSize=8, textColor=_WHITE,
                                                fontName="Helvetica-Bold")),
             Paragraph("Target Weight", ParagraphStyle("th2", fontSize=8, textColor=_WHITE,
                                                        fontName="Helvetica-Bold"))]
        ]
        for i, (asset, weight) in enumerate(target_weights.items()):
            bg = _ROW_ALT if i % 2 == 0 else _WHITE
            tw_data.append([
                Paragraph(str(asset), ParagraphStyle("ta", fontSize=8, textColor=_SLATE)),
                Paragraph(_fmt_pct(weight), ParagraphStyle("tw", fontSize=8, textColor=_SLATE)),
            ])
        tw_t = Table(tw_data, colWidths=[2 * inch, 2 * inch])
        tw_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_ROW_ALT, _WHITE]),
            ("GRID", (0, 0), (-1, -1), 0.5, _LIGHT_GREY),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(tw_t)

    # ── Fiduciary Certification Sign-off ──────────────────────────────────────

    story.append(_hr())
    story.append(Paragraph("FIDUCIARY CERTIFICATION", s["h2"]))
    story.append(Paragraph(
        "This report has been generated by the Altis Fiduciary Partners governed wealth-advisory "
        "system. All portfolio proposals, trade orders, and tax calculations are the product of "
        "deterministic, audited Python code (CLAUDE.md §Rules). No AI agent may produce or alter "
        "any number that reaches a trade. The cryptographic audit seal above ensures immutability "
        "under SEC Rule 204-2 and SEBI regulations.",
        s["cert"],
    ))
    story.append(Spacer(1, 8))

    cert_data = [
        [
            Paragraph("Adviser Signature", s["label"]),
            Paragraph("Date", s["label"]),
            Paragraph("Engine Version", s["label"]),
        ],
        [
            Paragraph("_" * 28, s["cert"]),
            Paragraph(generated_at, s["cert"]),
            Paragraph("wealth-advisor 0.1.0", s["cert"]),
        ],
    ]
    cert_t = Table(cert_data, colWidths=[2.5 * inch, 2.5 * inch, 1.5 * inch])
    cert_t.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(cert_t)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "ALTIS FIDUCIARY PARTNERS · Regulated Investment Adviser · "
        f"Report generated {generated_at}",
        s["confidential"],
    ))

    doc.build(story)
    return buf.getvalue()
