---
name: fiduciary-guardrails
description: Strict FINRA suitability rules and tax-loss harvesting bounds. Use whenever generating portfolio allocation, tax, or rebalancing code.
---

## Core Rules

1. **No LLM-generated trade amounts** — AI only generates target parameters (0.0 to 1.0). Raw trade quantities and dollar amounts must always come from deterministic, tested Python code.

2. **Equity exposure caps**
   - Moderate risk profile: max equity exposure is **80%**
   - Conservative risk profile: max equity exposure is **20%**

3. **Wash-sale rule** — AI must flag any attempt to buy a security within **30 days** of a harvested loss on that same security. Do not silently adjust; raise the flag explicitly.

4. **Math execution layer** — All portfolio optimisation and rebalancing math must be executed by **cvxpy** in the execution layer. AI reasoning may inform parameters but never replaces the solver.
