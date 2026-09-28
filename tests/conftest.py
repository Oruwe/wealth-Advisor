import pytest
from examples import example_portfolio, example_prices

from wealth_advisor.domain.market import PriceSnapshot
from wealth_advisor.domain.portfolio import Portfolio


@pytest.fixture
def portfolio() -> Portfolio:
    return example_portfolio()


@pytest.fixture
def prices() -> PriceSnapshot:
    return example_prices()
