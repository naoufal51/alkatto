"""Financial analysis graph for processing stock-related questions."""

from typing import Dict, List, Any
from datetime import datetime
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from shared.utils import load_chat_model
from agent.interview_graph.configuration import InterviewConfiguration
from shared.retrieval import StockRetriever
from agent.interview_graph.utils import (
    calculate_confidence_score,
    generate_executive_summary,
    combine_risk_analysis
)
from .state import FinancialState, MessagesState

def extract_stock_symbol(messages: List[dict], *, config: RunnableConfig = None) -> str:
    """Extract stock symbol from the conversation messages using LLM."""
    if not messages:
        return ""
        
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.analysis_model)
    
    # Get the last message content
    last_message = messages[-1].content if messages else ""
    
    # Use LLM to extract stock symbol
    prompt = [
        SystemMessage(content="""You are a financial assistant that extracts stock symbols from text.
Extract the stock symbol from the given text. Return ONLY the stock symbol in uppercase.
If multiple symbols are found, return the most relevant one. If no symbol is found, return an empty string.
Examples:
- Input: "What's the current price of Apple stock?"
  Output: AAPL
- Input: "How is Tesla performing today?"
  Output: TSLA
- Input: "Tell me about the weather"
  Output: ""
"""),
        HumanMessage(content=last_message)
    ]
    
    response = llm.invoke(prompt)
    return response.content.strip()

def analyze_stock_data(state: FinancialState, *, config: RunnableConfig = None):
    """Analyze stock data and news"""
    messages = state["messages"]
    
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.analysis_model)
    
    # Extract stock symbol from messages
    stock_symbol = extract_stock_symbol(messages, config=config)
    
    # Get stock data
    stock_retriever = StockRetriever()
    stock_data = stock_retriever.get_stock_data(stock_symbol)
    
    # Analyze data
    system_message = SystemMessage(content=configuration.stock_analysis_instructions)
    analysis = llm.invoke([system_message, HumanMessage(content=json.dumps(stock_data))])
    stock_data["analysis"] = analysis.content
    
    return {"stock_data": stock_data}

def analyze_market_sentiment(state: FinancialState, *, config: RunnableConfig = None):
    """Analyze market sentiment"""
    messages = state["messages"]
    
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.analysis_model)
    
    # Extract stock symbol from messages
    stock_symbol = extract_stock_symbol(messages, config=config)
    
    # Get stock data
    stock_retriever = StockRetriever()
    sentiment_data = stock_retriever.get_market_sentiment(stock_symbol)
    
    # Analyze sentiment
    system_message = SystemMessage(content=configuration.market_sentiment_instructions)
    analysis = llm.invoke([system_message, HumanMessage(content=json.dumps(sentiment_data))])
    sentiment_data["analysis"] = analysis.content
    
    return {"market_sentiment": sentiment_data}

def analyze_technical_indicators(state: FinancialState, *, config: RunnableConfig = None):
    """Analyze technical indicators"""
    stock_data = state.get("stock_data", {})
    
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.analysis_model)
    
    # Get technical data
    stock_retriever = StockRetriever()
    technical_data = stock_retriever.get_technical_indicators(stock_data.get("symbol", ""))
    
    # Analyze indicators
    system_message = SystemMessage(content=configuration.technical_analysis_instructions)
    analysis = llm.invoke([system_message, HumanMessage(content=json.dumps(technical_data))])
    technical_data["analysis"] = analysis.content
    
    return {"technical_analysis": technical_data}

def generate_price_targets(state: FinancialState, *, config: RunnableConfig = None):
    """Generate price targets"""
    stock_data = state.get("stock_data", {})
    market_sentiment = state.get("market_sentiment", {})
    technical_analysis = state.get("technical_analysis", {})
    
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.analysis_model)
    
    # Combine data for target generation
    target_data = {
        "stock_data": stock_data,
        "market_sentiment": market_sentiment,
        "technical_analysis": technical_analysis
    }
    
    # Generate targets
    system_message = SystemMessage(content=configuration.price_target_instructions)
    targets = llm.invoke([system_message, HumanMessage(content=json.dumps(target_data))])
    target_data["analysis"] = targets.content
    
    return {"price_targets": target_data}

def write_financial_article(state: FinancialState, *, config: RunnableConfig = None):
    """Write a comprehensive financial analysis article based on parallel analysis results"""
    stock_data = state.get("stock_data", {})
    market_sentiment = state.get("market_sentiment", {})
    technical_analysis = state.get("technical_analysis", {})
    price_targets = state.get("price_targets", {})

    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.article_model)

    # Prepare article data
    article_data = {
        "title": f"Financial Analysis Report: {stock_data.get('symbol', 'Unknown')}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "executive_summary": generate_executive_summary({
            "stock_data": stock_data,
            "market_sentiment": market_sentiment,
            "technical_analysis": technical_analysis,
            "price_targets": price_targets
        }, config),
        "company_overview": stock_data.get("company_overview", ""),
        "market_analysis": market_sentiment.get("market_analysis", ""),
        "technical_analysis": technical_analysis.get("analysis", ""),
        "financial_metrics": stock_data.get("financial_metrics", ""),
        "investment_thesis": price_targets.get("investment_thesis", ""),
        "risk_analysis": combine_risk_analysis({
            "stock_risks": stock_data.get("risks", []),
            "market_risks": market_sentiment.get("risks", []),
            "technical_risks": technical_analysis.get("risks", [])
        }, config),
        "price_targets": {
            "base_case": price_targets.get("targets", {}).get("base_case", 0),
            "bull_case": price_targets.get("targets", {}).get("bull_case", 0),
            "bear_case": price_targets.get("targets", {}).get("bear_case", 0)
        },
        "confidence_score": calculate_confidence_score([
            stock_data.get("confidence", 0),
            market_sentiment.get("confidence", 0),
            technical_analysis.get("confidence", 0),
            price_targets.get("confidence", 0)
        ], config)
    }

    # Generate article using LLM
    system_message = SystemMessage(content=configuration.financial_article_instructions)
    article = llm.invoke([
        system_message,
        HumanMessage(content=json.dumps(article_data))
    ])

    return {"messages": article.content}



def handle_financial_question(state: MessagesState, *, config: RunnableConfig = None) -> Dict[str, Any]:
    """Transform MessagesState to FinancialState and handle financial questions."""
    # Transform to financial state
    financial_state = FinancialState(
        messages=state["messages"],
        stock_data={},
        market_sentiment={},
        technical_analysis={},
        price_targets={},
        article={}
    )
    
    # Process with financial graph
    result = financial_app.invoke(financial_state)
    
    # Transform back to messages state
    return {"messages": result.get("messages", [])}


"""Create and return the compiled financial analysis graph."""
financial_graph = StateGraph(FinancialState)

# Add nodes
financial_graph.add_node("analyze_stock_data", analyze_stock_data)
financial_graph.add_node("analyze_market_sentiment", analyze_market_sentiment)
financial_graph.add_node("analyze_technical_indicators", analyze_technical_indicators)
financial_graph.add_node("generate_price_targets", generate_price_targets)
financial_graph.add_node("write_article", write_financial_article)

# Add parallel edges for analysis
financial_graph.add_edge("analyze_stock_data", "write_article")
financial_graph.add_edge("analyze_market_sentiment", "write_article")
financial_graph.add_edge("analyze_technical_indicators", "write_article")
financial_graph.add_edge("generate_price_targets", "write_article")

# Set the entry point
financial_graph.set_entry_point("analyze_stock_data")
financial_graph.set_entry_point("analyze_market_sentiment")
financial_graph.set_entry_point("analyze_technical_indicators")
financial_graph.set_entry_point("generate_price_targets")

# Compile the graph
financial_app = financial_graph.compile()
financial_app.name = "FinancialGraph"
