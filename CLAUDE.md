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
- Finance calculations run under `EXACT_CONTEXT` (`domain/primitives.py`): any rounding or
  float raises. Round only on purpose, with an explicit rounding mode.
- Trades are whole shares; only selling a position in full may trade a fractional share.
- Policy lives in `policy/`: the target table and model securities as read-only mappings, and
  the suitability rules in `suitability.yaml`, validated when the module is imported.
- The suitability gate re-derives the post-trade portfolio from the orders. Never make it rely
  on figures a proposal reports about itself, and keep the rebalancer within the same caps.
- Tax: sales relieve the highest-cost lots first (HIFO; ties go to the older lot), a partly
  sold lot's basis is split pro rata to the nearest cent (half-even), and estimates are
  federal only. Wash sales are flagged, not silently adjusted.
- Agents are checked by code, not trusted: `read_ips` accepts a fact only when its verbatim
  quote is in the IPS, mentions the fact, and has that value as its first number, and
  `write_briefing` rejects any number that is not in the formatted facts. Each gets one retry
  with the problems listed. Keep these guards in code; never loosen them to make a model pass.
- Give the writer every intermediate figure it needs to explain a result (for tax: each result,
  then what netting leaves to tax and at which rate). The number check cannot catch correct
  numbers joined by wrong reasoning.
- Untrusted text (an IPS) goes to an agent fenced in tags, and the agent is told it is data.
- Tests and CI never call Lyzr: use the fakes in `tests/fakes.py`. `scripts/lyzr_smoke.py` is
  the only live path. The demo client (C-1001, $100k) lives in `wealth_advisor/demo.py`.
- Warnings are errors in tests; the one exception is lyzr-adk's own Pydantic deprecations.
- The audit ledger (`ledger.py`) is append-only: never edit or rewrite `ledger.jsonl`, and
  record corrections as new entries. Hash only canonical JSON (sorted keys, compact, ASCII).
  Recording advice requires a clean replay, and approving re-runs it.
- Changing the rebalancer, gate, tax engine or briefing facts changes what old records replay
  to, so bump the package version: each advice record names the engine version that made it.
- Secrets live in `Settings` as `SecretStr` and are never logged or serialised.
- Build one step per PR (roadmap in README.md). Every step ships with tests, and all quality
  gates must pass before pushing.

## Commands

- `uv sync`: install
- `uv run ruff check --fix && uv run ruff format`: lint and format
- `uv run mypy`: strict type check of `src/`, `tests/` and `scripts/`
- `uv run pytest --cov`: tests with the 90% branch-coverage gate
- `uv run python scripts/lyzr_smoke.py`: advise the demo client through live Lyzr agents
  (needs `LYZR_API_KEY` in `.env`)
- `uv run python scripts/ledger.py verify`: check the audit ledger's chain, rules and replays
