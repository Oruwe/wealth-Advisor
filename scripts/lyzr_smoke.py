"""Advise the demo client through real Lyzr agents. Needs LYZR_API_KEY in a local `.env`.

    uv run python scripts/lyzr_smoke.py

It creates the Safe AI policy and both agents in Lyzr Studio, and deletes them when it ends.
"""

from wealth_advisor.advisor import advise
from wealth_advisor.agents.lyzr_runtime import lyzr_agents
from wealth_advisor.demo import DEMO_IPS, demo_portfolio, demo_prices
from wealth_advisor.settings import Settings


def main() -> None:
    settings = Settings()
    with lyzr_agents(settings) as agents:
        dossier = advise(DEMO_IPS, demo_portfolio(), demo_prices(), agents)
    print(dossier.model_dump_json(indent=2))
    print(f"\nBriefing from {settings.lyzr_model}:\n\n{dossier.briefing}")


if __name__ == "__main__":
    main()
