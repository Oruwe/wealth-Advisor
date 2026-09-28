"""Advise the demo client through real Lyzr agents, and record the run in the audit ledger.
Needs LYZR_API_KEY in a local `.env`.

    uv run python scripts/lyzr_smoke.py

It creates the Safe AI policy and both agents in Lyzr Studio, and deletes them when it ends.
The run is appended to `ledger.jsonl`; see `scripts/ledger.py` to verify or approve it.
"""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from wealth_advisor.advisor import advise
from wealth_advisor.agents.lyzr_runtime import lyzr_agents
from wealth_advisor.demo import DEMO_IPS, demo_portfolio, demo_prices
from wealth_advisor.ledger import AdviceInputs, Ledger
from wealth_advisor.settings import Settings


def main() -> None:
    settings = Settings()
    inputs = AdviceInputs(ips_text=DEMO_IPS, portfolio=demo_portfolio(), prices=demo_prices())
    with lyzr_agents(settings) as agents:
        dossier = advise(inputs.ips_text, inputs.portfolio, inputs.prices, agents)
    record = Ledger(Path("ledger.jsonl")).record_advice(
        uuid4().hex, inputs, dossier, datetime.now(UTC)
    )
    print(dossier.model_dump_json(indent=2))
    print(f"\nBriefing from {settings.lyzr_model}:\n\n{dossier.briefing}")
    print(f"\nRecorded as record {record.sequence} in ledger.jsonl, hash {record.hash}")
    print("Verify it:  uv run python scripts/ledger.py verify")
    print(f"Approve it: uv run python scripts/ledger.py approve {record.hash[:12]} --adviser YOU")


if __name__ == "__main__":
    main()
