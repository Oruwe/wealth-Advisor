import pytest

from wealth_advisor.demo import demo_portfolio, demo_prices
from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.portfolio import Portfolio


@pytest.fixture
def portfolio() -> Portfolio:
    return demo_portfolio()


@pytest.fixture
def prices() -> PriceSnapshot:
    return demo_prices()
