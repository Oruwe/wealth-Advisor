"""Advise the demo client through real Lyzr agents, and record the run in the audit ledger.
Needs LYZR_API_KEY in a local `.env`.

    uv run python scripts/lyzr_smoke.py

The Safe AI policy and both agents stay in Lyzr Studio between runs, so AIMS keeps their
history; the first run creates them. Every agent call of a run shares one Lyzr session, named
after the run. The run is appended to `ledger.jsonl`; see `scripts/ledger.py` to verify or
approve it.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from lyzr.exceptions import LyzrError

from wealth_advisor.advisor import advise
from wealth_advisor.agents.briefing import BriefingError
from wealth_advisor.agents.ips_reader import IpsReadingError
from wealth_advisor.agents.lyzr_runtime import AgentDriftError, AgentOutputError, lyzr_agents
from wealth_advisor.demo import DEMO_IPS, demo_portfolio, demo_prices
from wealth_advisor.ledger import AdviceInputs, Ledger
from wealth_advisor.settings import Settings


def main() -> int:
    settings = Settings()
    inputs = AdviceInputs(ips_text=DEMO_IPS, portfolio=demo_portfolio(), prices=demo_prices())
    run_id = uuid4().hex
    try:
        with lyzr_agents(settings, run_id, inputs.portfolio.client_id) as run:
            dossier = advise(inputs.ips_text, inputs.portfolio, inputs.prices, run.agents)
    except LyzrError as error:
        print(f"Could not complete the Lyzr calls: {error}")
        print("Check your internet connection and LYZR_API_KEY, then run this again.")
        return 1
    except (IpsReadingError, BriefingError, AgentDriftError, AgentOutputError) as error:
        print(f"The run was stopped and nothing was recorded: {error}")
        return 1
    record = Ledger(Path("ledger.jsonl")).record_advice(
        run_id, inputs, dossier, datetime.now(UTC), run.provenance
    )
    print(dossier.model_dump_json(indent=2))
    print(f"\nBriefing from {settings.lyzr_model}:\n\n{dossier.briefing}")
    print(f"\nRecorded as record {record.sequence} in ledger.jsonl, hash {record.hash}")
    print(f"Lyzr session {run_id}, agents:")
    print(f"  {run.provenance.reader.name} ({run.provenance.reader.id})")
    print(f"  {run.provenance.writer.name} ({run.provenance.writer.id})")
    print("Verify it:  uv run python scripts/ledger.py verify")
    print(f"Approve it: uv run python scripts/ledger.py approve {record.hash[:12]} --adviser YOU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
