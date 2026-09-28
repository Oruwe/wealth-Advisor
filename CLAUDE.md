# wealth-advisor

Governed wealth-advisory agent (HiDevs × Lyzr AI Quest, problem #05). Python 3.14, uv, `src/` layout.

## Rules

- Lyzr agents handle language only. Every number that reaches a trade comes from deterministic,
  tested Python; an agent may explain a number but never produce one.
- The same inputs must always produce the same proposal.
- Money, prices, quantities and weights are `Decimal` with fixed precision limits
  (`domain/primitives.py`), so float noise such as `0.1 + 0.2` is rejected at the boundary.
  Construct models with `Decimal`, never `float`; parse external data with `model_validate*`.
- Time is an input: domain and finance code take an explicit `as_of` date and never read the
  clock.
- Domain models are frozen and reject unknown fields, so agent output cannot smuggle in extras.
- Secrets live in `Settings` as `SecretStr` and are never logged or serialised.
- Build one step per PR (roadmap in README.md). Every step ships with tests, and all quality
  gates must pass before pushing.

## Commands

- `uv sync`: install
- `uv run ruff check --fix && uv run ruff format`: lint and format
- `uv run mypy`: strict type check of `src/` and `tests/`
- `uv run pytest --cov`: tests with the 90% branch-coverage gate
