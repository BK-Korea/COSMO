"""Yahoo Finance data retrieval module"""
import yfinance as yf
from typing import Dict, Any, List
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from config.config import settings


class YahooFinanceService:
    """Service for retrieving financial data from Yahoo Finance"""

    def __init__(self):
        self.cache = {}
        self.cache_dir = settings.yfinance_cache_dir

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))

    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a ticker

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')

        Returns:
            Dictionary containing ticker information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                'symbol': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
                'previous_close': info.get('previousClose', 'N/A'),
                'open': info.get('open', 'N/A'),
                'day_high': info.get('dayHigh', 'N/A'),
                'day_low': info.get('dayLow', 'N/A'),
                'volume': info.get('volume', 'N/A'),
                'average_volume': info.get('averageVolume', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'forward_pe': info.get('forwardPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A'),
            }
        except Exception as e:
            raise ValueError(f"Error fetching data for ticker {ticker}: {str(e)}")

    def get_historical_data(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data

        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with historical data
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            return hist
        except Exception as e:
            raise ValueError(f"Error fetching historical data for {ticker}: {str(e)}")

    def get_financials(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing income statement, balance sheet, and cash flow
        """
        try:
            stock = yf.Ticker(ticker)
            return {
                'income_statement': stock.income_stmt,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow,
                'quarterly_income_statement': stock.quarterly_income_stmt,
                'quarterly_balance_sheet': stock.quarterly_balance_sheet,
                'quarterly_cash_flow': stock.quarterly_cashflow,
            }
        except Exception as e:
            raise ValueError(f"Error fetching financials for {ticker}: {str(e)}")

    def get_recommendations(self, ticker: str) -> pd.DataFrame:
        """Get analyst recommendations"""
        try:
            stock = yf.Ticker(ticker)
            return stock.recommendations
        except Exception as e:
            raise ValueError(f"Error fetching recommendations for {ticker}: {str(e)}")

    def get_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent news about the ticker

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of news items to return (default: 10)

        Returns:
            List of news dictionaries with title, publisher, link, summary
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            # Limit to most recent news items
            return news[:limit] if news else []
        except Exception as e:
            raise ValueError(f"Error fetching news for {ticker}: {str(e)}")

    def get_major_holders(self, ticker: str) -> pd.DataFrame:
        """Get major shareholders information

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with major holders information
        """
        try:
            stock = yf.Ticker(ticker)
            return stock.major_holders
        except Exception as e:
            raise ValueError(f"Error fetching major holders for {ticker}: {str(e)}")

    def get_institutional_holders(self, ticker: str) -> pd.DataFrame:
        """Get institutional shareholders

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with institutional holders
        """
        try:
            stock = yf.Ticker(ticker)
            return stock.institutional_holders
        except Exception as e:
            raise ValueError(f"Error fetching institutional holders for {ticker}: {str(e)}")

    def get_insider_transactions(self, ticker: str) -> pd.DataFrame:
        """Get insider trading transactions

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with insider transactions
        """
        try:
            stock = yf.Ticker(ticker)
            return stock.insider_transactions
        except Exception as e:
            raise ValueError(f"Error fetching insider transactions for {ticker}: {str(e)}")

    def format_ticker_summary(self, ticker: str) -> str:
        """
        Format a comprehensive summary of ticker information

        Args:
            ticker: Stock ticker symbol

        Returns:
            Formatted string summary
        """
        info = self.get_ticker_info(ticker)

        summary = f"""
Stock Information for {info['symbol']} - {info['name']}

Basic Information:
- Sector: {info['sector']}
- Industry: {info['industry']}

Price Information:
- Current Price: ${info['current_price']}
- Previous Close: ${info['previous_close']}
- Day Range: ${info['day_low']} - ${info['day_high']}
- 52 Week Range: ${info['fifty_two_week_low']} - ${info['fifty_two_week_high']}

Volume:
- Current Volume: {info['volume']:,} if isinstance(info['volume'], (int, float)) else info['volume']
- Average Volume: {info['average_volume']:,} if isinstance(info['average_volume'], (int, float)) else info['average_volume']

Valuation:
- Market Cap: {info['market_cap']:,} if isinstance(info['market_cap'], (int, float)) else info['market_cap']
- P/E Ratio: {info['pe_ratio']}
- Forward P/E: {info['forward_pe']}
- Dividend Yield: {info['dividend_yield']}

Description:
{info['description'][:500]}...
"""
        return summary

    def get_multiple_tickers(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get information for multiple tickers"""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_ticker_info(ticker)
            except Exception as e:
                results[ticker] = {'error': str(e)}
        return results
