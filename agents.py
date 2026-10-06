import os
from typing import TypedDict, Annotated, Sequence, Any
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Import our custom tools and RAG pipeline
from tools import get_price, get_financials, calc_portfolio_risk, get_recent_news, generate_price_chart_base64
from rag import retrieve_financial_concept

load_dotenv()

# Define the State of the Graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda a, b: a + b]
    tickers: list[str]
    query_type: str # 'stock_research' or 'portfolio_risk'
    portfolio_holdings: dict
    fundamental_data: dict
    sentiment_data: dict
    risk_data: dict
    final_report: str

# Initialize the LLM
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# --- Agent Nodes ---

def supervisor_agent(state: AgentState) -> dict:
    """
    Analyzes the user's request and sets up the state with tickers or portfolio data.
    Decides the query_type.
    """
    print("--- SUPERVISOR AGENT ---")
    llm = get_llm()
    last_message = state['messages'][-1].content
    
    # We use a simple prompt to classify the request and extract tickers/holdings
    prompt = f"""
    Analyze this user request: "{last_message}"
    
    1. If the user is asking about a specific stock or comparing stocks, extract the tickers into a list (e.g., ['TCS', 'RELIANCE']).
    2. If the user is asking about portfolio risk, extract the holdings as a dictionary (e.g., {{"TCS": 0.4, "HDFCBANK": 0.6}}).
    
    Respond STRICTLY in the following JSON format without any markdown blocks:
    {{
      "type": "stock_research" or "portfolio_risk",
      "tickers": ["extracted_ticker1", "extracted_ticker2"] or [],
      "holdings": {{"ticker": weight}} or {{}}
    }}
    """
    
    response = llm.invoke([SystemMessage(content="You are a routing assistant. Respond ONLY in valid JSON."), HumanMessage(content=prompt)])
    
    import json
    try:
        # Strip potential markdown formatting
        raw_content = response.content.strip()
        if raw_content.startswith("```json"):
             raw_content = raw_content[7:-3]
        
        parsed = json.loads(raw_content)
        return {
            "query_type": parsed.get("type", "stock_research"),
            "tickers": parsed.get("tickers", []),
            "portfolio_holdings": parsed.get("holdings", {})
        }
    except Exception as e:
        print(f"Error parsing supervisor output: {e}")
        return {"query_type": "stock_research", "tickers": [], "portfolio_holdings": {}}

def fundamental_agent(state: AgentState) -> dict:
    """
    Gathers financials and explains them using the RAG database.
    """
    print("--- FUNDAMENTAL AGENT ---")
    if state.get("query_type") != "stock_research" or not state.get("tickers"):
         return {"fundamental_data": {}}
         
    tickers = state["tickers"]
    all_financials = {}
    charts = {}
    
    for ticker in tickers:
        financials = get_financials(ticker)
        all_financials[ticker] = financials
        
        price_data = get_price(ticker)
        if "1_month_history" in price_data:
            charts[ticker] = generate_price_chart_base64(ticker, price_data["1_month_history"])
    
    # Query RAG for context on a key metric (e.g., P/E ratio)
    pe_context = retrieve_financial_concept("P/E Ratio")
    
    llm = get_llm()
    if len(tickers) > 1:
        prompt = f"""
        You are a fundamental analysis agent.
        Here are the financials for {tickers}: {all_financials}
        
        Here is some educational context on P/E ratio: 
        {pe_context}
        
        Write a short, educational summary comparing the companies' fundamentals for a beginner investor. 
        Explain what their P/E ratios imply based on the educational context.
        DO NOT provide any buy/sell recommendations.
        """
    else:
        prompt = f"""
        You are a fundamental analysis agent.
        Here are the financials for {tickers[0]}: {all_financials[tickers[0]]}
        
        Here is some educational context on P/E ratio: 
        {pe_context}
        
        Write a short, educational summary of the company's fundamentals for a beginner investor. 
        Explain what their P/E ratio implies based on the educational context.
        DO NOT provide any buy/sell recommendations.
        """
        
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"fundamental_data": {"raw": all_financials, "summary": response.content, "charts": charts}}

def sentiment_agent(state: AgentState) -> dict:
    """
    Gathers recent news using Tavily and summarizes the sentiment.
    """
    print("--- SENTIMENT AGENT ---")
    if state.get("query_type") != "stock_research" or not state.get("tickers"):
         return {"sentiment_data": {}}
         
    tickers = state["tickers"]
    all_news = {}
    
    for ticker in tickers:
        news = get_recent_news(ticker)
        all_news[ticker] = news
        
    llm = get_llm()
    if len(tickers) > 1:
        prompt = f"""
        You are a sentiment analysis agent.
        Here is the recent news for {tickers}: {all_news}
        
        Compare and summarize the overall market sentiment (positive, negative, or neutral) for these companies based purely on these headlines and content. 
        Provide a 3-4 sentence summary.
        DO NOT provide any buy/sell recommendations.
        """
    else:
        prompt = f"""
        You are a sentiment analysis agent.
        Here is the recent news for {tickers[0]}: {all_news[tickers[0]]}
        
        Summarize the overall market sentiment (positive, negative, or neutral) based purely on these headlines and content. 
        Provide a 2-3 sentence summary.
        DO NOT provide any buy/sell recommendations.
        """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"sentiment_data": {"raw": all_news, "summary": response.content}}

def risk_agent(state: AgentState) -> dict:
    """
    Calculates portfolio risk and explains it.
    """
    print("--- RISK AGENT ---")
    if state.get("query_type") != "portfolio_risk" or not state.get("portfolio_holdings"):
         return {"risk_data": {}}
         
    holdings = state["portfolio_holdings"]
    risk_metrics = calc_portfolio_risk(holdings)
    
    if "error" in risk_metrics:
         return {"risk_data": {"error": risk_metrics["error"]}}
         
    # Query RAG for context
    risk_context = retrieve_financial_concept("Volatility and Portfolio Concentration HHI")
    
    llm = get_llm()
    prompt = f"""
    You are a risk analysis agent.
    Here are the risk metrics for the portfolio ({holdings}): {risk_metrics}
    
    Here is some educational context on Risk: 
    {risk_context}
    
    Write a short, educational summary of the portfolio's risk profile for a beginner investor. 
    Explain Volatility and Concentration based on the context.
    DO NOT provide any buy/sell recommendations.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"risk_data": {"raw": risk_metrics, "summary": response.content}}

def report_agent(state: AgentState) -> dict:
    """
    Compiles everything into a final research note and enforces guardrails.
    """
    print("--- REPORT AGENT ---")
    llm = get_llm()
    
    query_type = state.get("query_type")
    
    import datetime
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""
    You are the final Report Agent. Your job is to compile a structured, balanced research note based on the data provided by the specialized agents.
    
    Data provided:
    - Fundamental Data: {state.get("fundamental_data", {}).get("summary", "N/A")}
    - Sentiment Data: {state.get("sentiment_data", {}).get("summary", "N/A")}
    - Risk Data: {state.get("risk_data", {}).get("summary", "N/A")}
    
    Strict Constraints:
    1. You MUST NEVER say "buy", "sell", or give price targets.
    2. You MUST include this exact disclaimer at the very end: "DISCLAIMER: This report is for educational purposes only and does not constitute investment advice."
    3. You MUST state that the data is current as of {today}.
    
    Format the report nicely using Markdown. Use clear headings (e.g., Business Overview, Fundamentals, Sentiment, Risk Profile).
    """
    
    response = llm.invoke([SystemMessage(content="You compile safe, educational financial reports."), HumanMessage(content=prompt)])
    
    final_text = response.content
    charts = state.get("fundamental_data", {}).get("charts", {})
    if charts:
        final_text += "\n\n## Price Charts\n"
        for ticker, chart_md in charts.items():
            final_text += f"\n### {ticker}\n{chart_md}\n"
            
    return {"final_report": final_text}


# --- Routing Logic ---
def route_to_agents(state: AgentState):
    """
    Routes from Supervisor to the appropriate parallel agents based on query type.
    """
    if state["query_type"] == "stock_research":
        return ["fundamental_agent", "sentiment_agent"]
    elif state["query_type"] == "portfolio_risk":
        return ["risk_agent"]
    else:
        return ["report_agent"] # Fallback

# --- Build the Graph ---
def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_node("fundamental_agent", fundamental_agent)
    workflow.add_node("sentiment_agent", sentiment_agent)
    workflow.add_node("risk_agent", risk_agent)
    workflow.add_node("report_agent", report_agent)
    
    # Edges
    workflow.set_entry_point("supervisor_agent")
    
    # Conditional edge from supervisor to the specialized agents
    workflow.add_conditional_edges(
        "supervisor_agent",
        route_to_agents,
        {
            "fundamental_agent": "fundamental_agent",
            "sentiment_agent": "sentiment_agent",
            "risk_agent": "risk_agent",
            "report_agent": "report_agent"
        }
    )
    
    # All specialized agents flow into the report agent
    workflow.add_edge("fundamental_agent", "report_agent")
    workflow.add_edge("sentiment_agent", "report_agent")
    workflow.add_edge("risk_agent", "report_agent")
    
    # End
    workflow.add_edge("report_agent", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_graph()
    
    print("--- Testing Stock Research ---")
    test_state_1 = {"messages": [HumanMessage(content="Give me a research summary of Infosys.")]}
    result_1 = app.invoke(test_state_1)
    print("\nFINAL REPORT:\n")
    print(result_1['final_report'])
    
    # print("\n--- Testing Portfolio Risk ---")
    # test_state_2 = {"messages": [HumanMessage(content="My portfolio is 40% TCS, 30% HDFC Bank and 30% Reliance. How risky is it?")]}
    # result_2 = app.invoke(test_state_2)
    # print("\nFINAL REPORT:\n")
    # print(result_2['final_report'])
