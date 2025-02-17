"""Market analysis graph implementation."""

from typing import Dict, Any, List
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from .state import MarketState
from .configuration import MarketConfiguration
from ..utils import load_chat_model
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import END, Graph, StateGraph

def analyze_market_trends(state: MarketState, *, config: RunnableConfig = None):
    """Analyze market trends and sector performance."""
    messages = state["messages"]
    
    # Get configuration and LLM
    configuration = MarketConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.trend_model)
    
    # Get the last message content
    last_message = messages[-1].content if messages else ""
    
    # Get current date from config
    current_date = config.get("current_date") if config else None
    
    # Generate market trends analysis
    system_message = SystemMessage(content=configuration.market_trends_template.format(current_date=current_date))
    
    analysis = llm.invoke([
        system_message,
        HumanMessage(content=last_message)
    ])
    
    return {"analysis": {"market_trends": analysis.content}}


def analyze_market_sentiment(state: MarketState, *, config: RunnableConfig = None):
    """Analyze market sentiment and investor behavior."""
    messages = state["messages"]
    
    # Get configuration and LLM
    configuration = MarketConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.trend_model)
    
    # Get the last message content
    last_message = messages[-1].content if messages else ""
    
    # Get current date from config
    current_date = config.get("current_date") if config else None
    
    # Generate market sentiment analysis
    system_message = SystemMessage(content=configuration.market_sentiment_template.format(current_date=current_date))
    
    analysis = llm.invoke([
        system_message,
        HumanMessage(content=last_message)
    ])
    
    return {"analysis": {"market_sentiment": analysis.content}}


def write_market_report(state: MarketState, *, config: RunnableConfig = None):
    """Write a comprehensive market analysis report."""
    messages = state["messages"]
    
    # Get configuration and LLM
    configuration = MarketConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.analysis_model)
    
    # Get the last message content
    last_message = messages[-1].content if messages else ""
    
    # Get current date from config
    current_date = config.get("current_date") if config else None
    
    # Get analysis results
    analysis = state.get("analysis", {})
    market_trends = analysis.get("market_trends", "No market trends analysis available.")
    market_sentiment = analysis.get("market_sentiment", "No market sentiment analysis available.")
    
    # Generate market report
    system_message = SystemMessage(content=configuration.market_report_template.format(current_date=current_date))
    
    report = llm.invoke([
        system_message,
        HumanMessage(content=f"""
Question: {last_message}

Market Trends Analysis:
{market_trends}

Market Sentiment Analysis:
{market_sentiment}
""")
    ])
    
    return {"messages": report.content}


def check_research_quality(state: MarketState, *, config: RunnableConfig = None):
    """Evaluate if the research properly addresses the original query."""
    messages = state["messages"]
    
    # Get configuration and LLM
    configuration = MarketConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.quality_control_model)
    
    # Get the original query (from first message)
    original_query = messages[0].content if messages else ""
    
    # Get the latest research report (from last message)
    latest_report = messages[-1].content if len(messages) > 1 else ""
    
    # Generate quality assessment
    system_message = SystemMessage(content=configuration.quality_control_template)
    
    assessment = llm.invoke([
        system_message,
        HumanMessage(content=f"""Original Query: {original_query}

Research to Evaluate:
{latest_report}""")
    ])
    
    # Parse the evaluation results
    evaluation = json.loads(assessment.content.strip())
    
    # Store the quality assessment in state
    return {
        "analysis": {
            "quality_control": evaluation
        }
    }


def needs_revision(state: MarketState) -> str:
    """Determine if the research needs revision based on quality assessment."""
    # Get quality control results
    analysis = state.get("analysis", {})
    quality_control = analysis.get("quality_control", {})
    
    # Check if research is approved
    return "end" if quality_control.get("approved", False) else "revise_report"


# Create the market analysis graph
market_graph = StateGraph(MarketState)

# Add nodes for market analysis
market_graph.add_node("analyze_trends", analyze_market_trends)
market_graph.add_node("analyze_sentiment", analyze_market_sentiment)
market_graph.add_node("write_report", write_market_report)
market_graph.add_node("check_quality", check_research_quality)

# Add edges for the analysis flow
market_graph.add_edge("analyze_trends", "analyze_sentiment")
market_graph.add_edge("analyze_sentiment", "write_report")
market_graph.add_edge("write_report", "check_quality")

# Add conditional edges for quality control
market_graph.add_conditional_edges(
    "check_quality",
    needs_revision,
    {
        "revise_report": "write_report",
        "end": END
    }
)

# Set entry point
market_graph.set_entry_point("analyze_trends")

# Compile the graph
market_app = market_graph.compile()
market_app.name = "MarketGraph"
