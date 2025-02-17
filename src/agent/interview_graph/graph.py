from typing import Optional, List, Dict
from shared.utils import (
    format_docs, 
    load_chat_model,
    Message,
    RunnableConfig,
    StateGraph,
    END,
    START
)
from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    SystemMessage,
    get_buffer_string
)
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState

from agent.interview_graph.state import MessagesState
from agent.interview_graph.configuration import InterviewConfiguration
from shared.retrieval import WebRetriever, StockRetriever
from langgraph.checkpoint.memory import MemorySaver
import json
import numpy as np
from datetime import datetime
from agent.interview_graph.financial.graph import financial_app
from agent.interview_graph.general.graph import general_app
from agent.interview_graph.market.graph import market_app

from agent.interview_graph.utils import (
    calculate_confidence_score,
    generate_executive_summary,
    combine_risk_analysis
)

def classify_question(state: MessagesState):
    """Classify if the question is financial, market, or general."""
    messages = state.get("messages", [])
    if not messages:
        return "general_question"

    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(None)
    llm = load_chat_model(configuration.classification_model)

    # Get the last message content
    last_message = messages[-1].content if messages else ""

    # Classify the question
    system_message = SystemMessage(content="""Classify this question into one of these categories. Be very specific in distinguishing between financial and market questions:

FINANCIAL questions focus on:
- Stock prices, technical analysis, and trading
- Company financial metrics (P/E ratio, EPS, revenue, profit margins)
- Financial statements and accounting
- Investment analysis and portfolio management
- Dividend policies and stock splits
- Company-specific financial performance
- Valuation metrics and methodologies

MARKET questions focus on:
- Industry/sector analysis and trends
- Competitive landscape and market share
- Market size and growth projections
- Consumer behavior and preferences
- Market entry strategies
- Geographic market differences
- Product market fit
- Market segmentation
- Industry regulations and policies
- Innovation and disruption in markets
- Supply chain and distribution analysis

GENERAL questions:
- Any question not specifically about financial metrics or market analysis
- General knowledge, facts, history
- Technology explanations
- Company history or background
- Product features or specifications
- News and current events
- How-to questions

Return EXACTLY one of these words: 'financial_question', 'market_question', or 'general_question'""")
    
    classification = llm.invoke([system_message, HumanMessage(content=last_message)])

    return classification.content.strip()

def route_messages(state: MessagesState, name: str = "expert"):
    """Route between financial, market, and general knowledge questions."""
    if not state["messages"]:
        return END
        
    # Classify the question
    question_type = classify_question(state)
    
    # Route based on classification
    if question_type == "financial_question":
        return "handle_financial"
    elif question_type == "market_question":
        return "handle_market"
    else:
        return "handle_general"

def handle_financial(state: MessagesState, *, config: RunnableConfig = None):
    """Handle financial questions using the financial subgraph."""
    return financial_app.invoke(state)

def handle_market(state: MessagesState, *, config: RunnableConfig = None):
    """Handle market questions using the market subgraph."""
    return market_app.invoke(state)

def handle_general(state: MessagesState, *, config: RunnableConfig = None):
    """Handle general knowledge questions using the general subgraph."""
    return general_app.invoke(state)

checkpointer = MemorySaver()

router = StateGraph(MessagesState)

# Add nodes
router.add_node("handle_financial", handle_financial)
router.add_node("handle_market", handle_market)
router.add_node("handle_general", handle_general)

# Add conditional edges
router.add_conditional_edges(
    START,
    route_messages,
    {
        "handle_financial": "handle_financial",
        "handle_market": "handle_market",
        "handle_general": "handle_general"
    }
)

# Set the entry point
# router.set_entry_point("root")

# Add edges to end
router.add_edge("handle_financial", END)
router.add_edge("handle_market", END)
router.add_edge("handle_general", END)

# Compile the graph
graph = router.compile(checkpointer=checkpointer)
# graph = router.compile()
graph.name = "InterviewRouterGraph"