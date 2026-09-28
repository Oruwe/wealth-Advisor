"""The console's live advice runner: the demo client, advised through the Lyzr agents."""

from collections.abc import Callable

from lyzr import Studio
from lyzr.exceptions import LyzrError

from wealth_advisor.advisor import advise
from wealth_advisor.agents.briefing import BriefingError
from wealth_advisor.agents.ips_reader import IpsReadingError
from wealth_advisor.agents.lyzr_runtime import AgentDriftError, AgentOutputError, lyzr_agents
from wealth_advisor.demo import DEMO_IPS, demo_portfolio, demo_prices
from wealth_advisor.ledger import AdviceInputs
from wealth_advisor.settings import Settings
from wealth_advisor.web.app import AdviceRun, AdviceRunError, Runner


def lyzr_demo_runner(settings: Settings, studio_factory: Callable[..., Studio] = Studio) -> Runner:
    """Advise the demo client through the Lyzr agents, in one Lyzr session per run."""

    def run(run_id: str) -> AdviceRun:
        inputs = AdviceInputs(ips_text=DEMO_IPS, portfolio=demo_portfolio(), prices=demo_prices())
        client = inputs.portfolio.client_id
        try:
            with lyzr_agents(settings, run_id, client, studio_factory) as lyzr:
                dossier = advise(inputs.ips_text, inputs.portfolio, inputs.prices, lyzr.agents)
        except LyzrError as error:
            raise AdviceRunError(
                f"could not complete the Lyzr calls ({error}); check the internet connection "
                "and LYZR_API_KEY"
            ) from error
        except (IpsReadingError, BriefingError, AgentDriftError, AgentOutputError) as error:
            raise AdviceRunError(str(error)) from error
        return AdviceRun(inputs, dossier, lyzr.provenance)

    return run
