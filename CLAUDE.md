# wealth-advisor

Governed wealth-advisory agent (HiDevs × Lyzr AI Quest, problem #05). Python 3.14, uv, `src/` layout.

## Rules

- Lyzr agents handle language only. Every number that reaches a trade comes from deterministic,
  tested Python; an agent may explain a number but never produce one.
- The same inputs must always produce the same proposal.
- Secrets live in `Settings` as `SecretStr` and are never logged or serialised.
- Build one step per PR (roadmap in README.md). Every step ships with tests, and all quality
  gates must pass before pushing.

## Commands

- `uv sync`: install
- `uv run ruff check --fix && uv run ruff format`: lint and format
- `uv run mypy`: strict type check of `src/` and `tests/`
- `uv run pytest --cov`: tests with the 90% branch-coverage gate
