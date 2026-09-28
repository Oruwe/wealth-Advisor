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
| 2 | Domain models: client profile (IPS), holdings, asset classes | Next |
| 3 | Deterministic rebalancer | Planned |
| 4 | Suitability gate: policy rules that block unsuitable trades | Planned |
| 5 | Tax-lot engine | Planned |
| 6 | Lyzr agents with Safe AI guardrails | Planned |
| 7 | Audit ledger and AIMS | Planned |
| 8 | Adviser UI and deployment | Planned |

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
