import yfinance as yf
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from tavily import TavilyClient
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

load_dotenv()
def get_price(ticker: str) -> dict:
    """
    Returns current and historical prices for a ticker.
    Appends .NS for NSE stocks automatically if it's an Indian stock (basic check).
    """
    try:
        # Simple heuristic, assumes user might just type 'RELIANCE'
        if not ticker.endswith('.NS') and not ticker.endswith('.BO') and not '.' in ticker:
            ticker = ticker + '.NS'
            
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty:
            return {"error": f"No price data found for {ticker}."}
        
        current_price = hist['Close'].iloc[-1]
        historical_prices = hist['Close'].to_dict()
        # Convert timestamp keys to string
        historical_prices = {k.strftime('%Y-%m-%d'): v for k, v in historical_prices.items()}
        
        return {
            "ticker": ticker,
            "current_price": current_price,
            "1_month_history": historical_prices
        }
    except Exception as e:
        return {"error": str(e)}

def generate_price_chart_base64(ticker: str, hist_data: dict) -> str:
    """
    Generates a base64 encoded PNG chart from historical price data.
    """
    try:
        if not hist_data:
            return ""
        
        dates = list(hist_data.keys())
        prices = list(hist_data.values())
        
        plt.figure(figsize=(8, 4))
        plt.plot(dates, prices, marker='.', linestyle='-', color='b')
        plt.title(f"{ticker} - 1 Month Price History")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"![{ticker} Price Chart](data:image/png;base64,{base64_img})"
    except Exception as e:
        return f"*(Chart generation failed: {str(e)})*"

def get_financials(ticker: str) -> dict:
    """
    Returns key ratios and statements (revenue, profit, P/E, debt-to-equity).
    """
    try:
        if not ticker.endswith('.NS') and not ticker.endswith('.BO') and not '.' in ticker:
            ticker = ticker + '.NS'
            
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extracting relevant financial data
        financials = {
            "ticker": ticker,
            "market_cap": info.get("marketCap", "N/A"),
            "trailing_pe": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "price_to_book": info.get("priceToBook", "N/A"),
            "debt_to_equity": info.get("debtToEquity", "N/A"),
            "total_revenue": info.get("totalRevenue", "N/A"),
            "profit_margins": info.get("profitMargins", "N/A"),
            "return_on_equity": info.get("returnOnEquity", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A")
        }
        return financials
    except Exception as e:
        return {"error": str(e)}

def calc_portfolio_risk(holdings: dict) -> dict:
    """
    Computes volatility and concentration for a set of holdings and weights.
    holdings: dict of ticker -> weight (e.g., {'TCS.NS': 0.4, 'HDFCBANK.NS': 0.3})
    """
    try:
        if not holdings:
            return {"error": "Holdings cannot be empty."}
            
        # Normalize weights just in case
        total_weight = sum(holdings.values())
        weights = {k: v / total_weight for k, v in holdings.items()}
        
        tickers = list(weights.keys())
        
        # Format tickers for yfinance
        formatted_tickers = []
        for t in tickers:
            if not t.endswith('.NS') and not t.endswith('.BO') and not '.' in t:
                formatted_tickers.append(t + '.NS')
            else:
                formatted_tickers.append(t)
                
        # Fetch 1 year of daily historical data for all tickers
        data = yf.download(formatted_tickers, period="1y")['Close']
        
        if data.empty:
             return {"error": "Could not fetch data for given tickers."}
             
        if isinstance(data, pd.Series): # Only one stock
             data = data.to_frame()
             
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate covariance matrix (annualized)
        cov_matrix = returns.cov() * 252
        
        # Portfolio variance = w.T * cov * w
        weight_array = np.array(list(weights.values()))
        port_variance = np.dot(weight_array.T, np.dot(cov_matrix, weight_array))
        
        # Portfolio volatility (standard deviation)
        port_volatility = np.sqrt(port_variance)
        
        # Calculate concentration (Herfindahl-Hirschman Index)
        hhi = sum([w**2 for w in weights.values()])
        
        return {
            "portfolio_volatility_annualized": port_volatility,
            "concentration_hhi": hhi,
            "holdings_normalized": weights
        }
    except Exception as e:
        return {"error": str(e)}

def get_recent_news(query: str) -> dict:
    """
    Searches the web for recent news regarding a specific query (ticker or company name).
    Returns a list of relevant articles.
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {"error": "TAVILY_API_KEY not found in environment variables."}
            
        client = TavilyClient(api_key=api_key)
        # We use a search type optimized for news and recent events
        response = client.search(
            query=f"{query} stock news earnings sentiment",
            search_depth="advanced",
            include_images=False,
            include_answer=False,
            max_results=5,
            topic="news" 
        )
        
        # Extract relevant fields
        results = []
        for res in response.get('results', []):
            results.append({
                "title": res.get("title"),
                "url": res.get("url"),
                "content": res.get("content"),
                "score": res.get("score"),
                "published_date": res.get("published_date")
            })
            
        return {"query": query, "news_results": results}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Simple tests
    print("Testing get_price for RELIANCE...")
    print(get_price("RELIANCE"))
    
    print("\nTesting get_financials for TCS...")
    print(get_financials("TCS"))
    
    print("\nTesting calc_portfolio_risk for a sample portfolio...")
    print(calc_portfolio_risk({"TCS.NS": 0.4, "HDFCBANK.NS": 0.3, "RELIANCE.NS": 0.3}))
    
    print("\nTesting get_recent_news for Tata Motors...")
    print(get_recent_news("Tata Motors"))
