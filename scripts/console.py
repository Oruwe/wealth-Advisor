"""Run the adviser console at http://127.0.0.1:8000.

    uv run python scripts/console.py

Reviewing, approving and auditing need only the ledger (`ledger.jsonl`). Running new advice
for the demo client also needs LYZR_API_KEY in a local `.env`. The console listens on this
computer only.
"""

import os
from pathlib import Path

import uvicorn
from pydantic import ValidationError

from wealth_advisor.ledger import Ledger
from wealth_advisor.settings import Settings
from wealth_advisor.web.app import Runner, create_app
from wealth_advisor.web.runner import lyzr_demo_runner


def main() -> None:
    runner: Runner | None
    try:
        runner = lyzr_demo_runner(Settings())
    except ValidationError:
        print("LYZR_API_KEY is not set: the console can review, approve and audit, but not run")
        print("new advice. Set it in .env and restart to enable runs.")
        runner = None
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run(create_app(Ledger(Path("ledger.jsonl")), runner), host=host, port=8000)


if __name__ == "__main__":
    main()
