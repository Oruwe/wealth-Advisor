# wealth-advisor

A governed wealth-advisory agent for registered investment advisers: it profiles
clients, rebalances portfolios with deterministic math, and blocks unsuitable
trades before an adviser ever sees them. Built for problem #05 of the
HiDevs × Lyzr AI Quest.

## Design rule

| Job | Done by | Why |
|---|---|---|
| Read and write language: IPS documents, rationale, adviser briefings | Lyzr agents | LLMs are good at language |
| Produce every number that reaches a trade | Plain, tested Python | Same input, same output, every time |
| Decide what is allowed | Policy rules plus Lyzr Safe AI | Rules can be audited; prompts cannot |

An agent may explain a number. It never produces one.

## Build steps

The system is built one step at a time. Each step is a pull request that ships
with its tests.

| Step | Scope | Status |
|---|---|---|
| 1 | Foundation: toolchain, settings, quality gates, CI | Done |
| 2 | Domain models: client profile (IPS), holdings, asset classes | Done |
| 3 | Deterministic rebalancer | Done |
| 4 | Suitability gate: policy rules that block unsuitable trades | Done |
| 5 | Tax-lot engine | Next |
| 6 | Lyzr agents with Safe AI guardrails | Planned |
| 7 | Audit ledger and AIMS | Planned |
| 8 | Adviser UI and deployment | Planned |

## How rebalancing works

1. The client's risk score (1–10) picks a target allocation from a fixed table
   (`policy/model_portfolio.py`). Conservative scores hold at most 20% equity.
2. Holdings outside the model (VTI, BND, GLD, plus uninvested cash) are sold in full.
3. Each asset class gets a dollar target: its weight times the portfolio value,
   scaled down when the client's cash reserve needs more than the cash weight, and
   rounded down to the cent so targets never exceed the money available.
4. Each model security's ideal trade is rounded down and up to whole shares. Every
   combination (at most 8) is scored by its squared dollar drift from target, and the
   best one that keeps cash at or above the reserve, and every asset class within the
   client's suitability caps, wins. Ties go to the smaller trade.

All of this runs under a `Decimal` context that raises on any rounding or float, so
the same inputs give the same orders on every machine. Hypothesis checks the
guarantees on random portfolios: cash never drops below the reserve, value is
conserved, every asset class lands within one share of its target, and rebalancing
twice changes nothing.

## Suitability gate

No proposal should reach an adviser without passing the gate (`suitability_gate.py`).
It replays the proposal's orders on the client's portfolio and checks the result, so it
never trusts the figures a proposal reports about itself. The rules live in
`policy/suitability.yaml`, which compliance can edit without touching code; a typo or
an incomplete rule set stops the app from starting.

| Rule | Blocks a proposal that |
|---|---|
| `approved_securities` | buys or keeps anything off the approved list: meme coins, options, single stocks |
| `max_weight` | leaves an asset class above its cap for the client's risk band, e.g. conservative ≤ 20% equity |
| `cash_reserve` | spends the cash reserve in the client's IPS |
| `no_overselling` | sells more than the client holds |
| `snapshot_prices` | prices an order away from the day's price snapshot |
| `matching_figures` | reports values its orders don't produce |
| `same_client`, `same_date` | mixes up clients or dates |

A blocked proposal gets a report naming every rule it breaks, in words an adviser can
read. The rebalancer respects the same caps, and a property test checks that every
proposal it makes, for any portfolio and any risk score, passes the gate.

## Quickstart

You only need [uv](https://docs.astral.sh/uv/getting-started/installation/); it
downloads Python 3.14 for you.

```bash
uv sync                    # create .venv with every dependency
uv run pytest              # run the tests
uv run pre-commit install  # run the checks on every commit
```

Copy `.env.example` to `.env` and set `LYZR_API_KEY` before running anything
that calls Lyzr.

## Quality gates

CI runs the same commands on every push and pull request:

```bash
uv run ruff check           # lint, including security rules
uv run ruff format --check  # formatting
uv run mypy                 # strict type checking of src/ and tests/
uv run pytest --cov         # tests; fails if branch coverage drops below 90%
```
