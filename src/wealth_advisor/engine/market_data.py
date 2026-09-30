"""Live market data feed and Ledoit-Wolf shrinkage covariance estimator."""

from __future__ import annotations

import datetime
import logging

import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


class MarketDataFeed:
    """Fetches historical OHLCV data from Yahoo Finance."""

    @staticmethod
    def get_historical_prices(symbols: list[str], years: int = 2) -> pd.DataFrame:
        """Return daily Close prices for *symbols* over the last *years* years.

        Missing observations are forward-filled so there are no NaN gaps in
        the middle of the series (leading NaNs before a ticker's listing date
        are left as-is and will be dropped by the RiskEstimator).
        """
        end = datetime.date.today()
        start = end.replace(year=end.year - years)
        raw = yf.download(
            symbols,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
        )
        prices: pd.DataFrame
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]] if "Close" in raw.columns else raw
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=symbols[0])
        prices = prices.ffill()
        # Ensure columns match the requested symbols in the requested order
        present = [s for s in symbols if s in prices.columns]
        return prices[present]


class RiskEstimator:
    """Computes forward-looking risk and return estimates from price history."""

    @staticmethod
    def compute_expected_returns(prices: pd.DataFrame) -> dict[str, float]:
        """Annualised expected return per symbol (mean daily return × 252)."""
        returns = prices.pct_change().dropna()
        return {str(col): float(val) for col, val in (returns.mean() * 252).items()}

    @staticmethod
    def compute_covariance_matrix(prices: pd.DataFrame) -> pd.DataFrame:
        """Annualised Ledoit-Wolf shrunk covariance matrix (daily cov × 252)."""
        returns = prices.pct_change().dropna()
        lw = LedoitWolf()
        lw.fit(returns.to_numpy())
        return pd.DataFrame(
            lw.covariance_ * 252,
            index=returns.columns,
            columns=returns.columns,
        )
