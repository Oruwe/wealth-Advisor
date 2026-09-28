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
| 5 | Tax-lot engine | Done |
| 6 | Lyzr agents with Safe AI guardrails | Done |
| 7 | Audit ledger and AIMS | Next |
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

## Tax impact

`estimate_tax` (`tax.py`) turns a proposal's sales into specific-lot instructions and an
estimated federal tax bill:

- **Lots**: each sale relieves the highest-cost lots first (HIFO), which usually realises
  the least gain; on a tie, the older lot goes first. A partly sold lot's cost basis is
  split pro rata and rounded to the nearest cent.
- **Holding period**: a lot is long-term only if sold after the first anniversary of its
  purchase.
- **Netting**: short- and long-term results are netted the way Schedule D nets them.
  Short-term gains are taxed at the client's marginal rate, and long-term gains at 0%, 15%
  or 20% depending on that rate. A net loss is reported, never turned into a refund.
- **Wash sales**: a loss is flagged, with the amount at risk, when shares of the same
  security bought in the last 30 days are still held, or when the same orders buy it.

For the $100k example, selling TSLA and VTI realises $4,600 of long-term gains and $2,300
of short-term losses; after netting, $2,300 is taxed at 15%, about $345.

Estimates exclude state tax and the 3.8% net investment income tax, and the long-term rate
is approximated from the marginal rate (the real breakpoints are income levels).

## Agents

Two Lyzr agents handle the language at each end of the pipeline (`advisor.py`). Code does
everything in between:

```
IPS text ──▶ IPS reader ──▶ rebalancer ──▶ suitability gate ──▶ tax engine ──▶ briefing writer
             (Lyzr agent)   └───────────── deterministic Python ─────────────┘  (Lyzr agent)
```

- **IPS reader** (`agents/ips_reader.py`) turns an Investment Policy Statement into the
  client's risk score, horizon, cash reserve and tax rate, and must quote, word for word,
  the passage behind each fact. Code accepts a fact only if its quote really is in the IPS
  (ignoring case, line breaks and curly quotes) and states that value and no other number,
  so the 10 in "3 on a scale of 1 to 10" can never pass for the score. A rejected reading
  gets one retry that lists its problems; if that fails too, or the facts break the profile
  rules, the run stops. Nothing is guessed, and the dossier keeps each quote next to its
  fact for the adviser to check.
- **Briefing writer** (`agents/briefing.py`) explains the proposal, the gate's verdict and
  the tax estimate to the adviser. It sees only facts that code has already formatted, and
  every number it writes must be one of them: `19.8%` matches `19.80%`, but a number it
  worked out itself does not. A draft that breaks the rule gets one rewrite, told which
  numbers were wrong; if the rewrite breaks it too, the run stops. A number check can't
  catch right numbers joined by wrong reasoning, so the facts spell out every step the
  writer has to explain. For tax, that is each result, then what netting leaves to tax and
  at which rate, so the writer restates the calculation instead of reconstructing it.
- **Safe AI**: both agents run at temperature 0 under one Lyzr Safe AI policy that detects
  prompt injection, masks secrets, and redacts names, email addresses, phone numbers, SSNs
  and card numbers. The IPS is fenced in `<ips>` tags and treated as data, never as
  instructions.
- **Lifecycle**: `lyzr_agents()` (`agents/lyzr_runtime.py`) creates the policy and both
  agents in Lyzr Studio for a run and deletes them afterwards, even when the run fails.

Tests and CI never call Lyzr. They use fake agents, plus an offline contract test that runs
the agent and policy settings through the Lyzr SDK's own signatures and validation.

### Run it live

```bash
cp .env.example .env                 # then set LYZR_API_KEY in .env
uv run python scripts/lyzr_smoke.py  # advises the $100k demo client through real agents
```

The script prints the dossier as JSON, then the briefing. Agents use `openai/gpt-4.1`
unless `LYZR_MODEL` says otherwise, e.g. `LYZR_MODEL=anthropic/claude-sonnet-4-5` in `.env`.

## Quickstart

You only need [uv](https://docs.astral.sh/uv/getting-started/installation/); it
downloads Python 3.14 for you.

```bash
uv sync                    # create .venv with every dependency
uv run pytest              # run the tests
uv run pre-commit install  # run the checks on every commit
```

Copy `.env.example` to `.env` and set `LYZR_API_KEY` before running anything
that calls Lyzr (see [Run it live](#run-it-live)).

## Quality gates

CI runs the same commands on every push and pull request:

```bash
uv run ruff check           # lint, including security rules
uv run ruff format --check  # formatting
uv run mypy                 # strict type checking of src/, tests/ and scripts/
uv run pytest --cov         # tests; fails if branch coverage drops below 90%
```
