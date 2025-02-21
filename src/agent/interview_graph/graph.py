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
from agent.interview_graph.research.graph import research_app

from agent.interview_graph.utils import (
    calculate_confidence_score,
    generate_executive_summary,
    combine_risk_analysis
)

def classify_question(state: MessagesState):
    """Classify if the question is financial, market, general, or research."""
    messages = state.get("messages", [])
    if not messages:
        return "general_question"

    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(None)
    llm = load_chat_model(configuration.classification_model)

    # Get the last message content
    last_message = messages[-1].content if messages else ""

    # Classify the question
    system_message = SystemMessage(content="""Classify this question into one of these categories:

RESEARCH questions focus on:
- Requests for research papers, academic articles, or scientific publications
- Questions asking for literature or papers on a specific topic
- Queries about academic research in any field
- Questions containing phrases like "research papers", "papers about", "articles on", "studies about"
- Scientific discoveries and innovations from academic literature
- Technical papers and methodologies
- Research trends and developments
- Academic literature reviews
- Scientific methodologies
- Research findings and implications
- Academic contributions
- Experimental results and analysis

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
- Any question not specifically about research papers, financial metrics, or market analysis
- General knowledge, facts, history
- Technology explanations
- Company history or background
- Product features or specifications
- News and current events
- How-to questions

Return EXACTLY one of these words: 'financial_question', 'market_question', 'research_question', or 'general_question'""")
    
    classification = llm.invoke([system_message, HumanMessage(content=last_message)])

    return classification.content.strip()

def calculate_confidence_score(state: MessagesState):
    """Calculate a confidence score using numpy for testing purposes."""
    messages = state.get("messages", [])
    if not messages:
        return 0.0
    
    # Simple test using numpy - create random confidence score
    random_score = np.random.uniform(0.7, 1.0)
    weighted_score = np.mean([random_score, 0.85])  # Combine with baseline
    
    return np.round(weighted_score, 2)

def route_messages(state: MessagesState, name: str = "expert"):
    """Route between financial, market, research, and general knowledge questions."""
    if not state["messages"]:
        return END
        
    # Add confidence scoring
    confidence = calculate_confidence_score(state)
    state["confidence_score"] = confidence
    
    # Classify the question
    question_type = classify_question(state)
    
    # Route based on classification
    if question_type == "financial_question":
        return "handle_financial"
    elif question_type == "market_question":
        return "handle_market"
    elif question_type == "research_question":
        return "handle_research"
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

def handle_research(state: MessagesState, *, config: RunnableConfig = None):
    """Handle research questions using the research subgraph."""
    return research_app.invoke(state)

checkpointer = MemorySaver()

router = StateGraph(MessagesState)

# Add nodes
router.add_node("handle_financial", handle_financial)
router.add_node("handle_market", handle_market)
router.add_node("handle_general", handle_general)
router.add_node("handle_research", handle_research)

# Add conditional edges
router.add_conditional_edges(
    START,
    route_messages,
    {
        "handle_financial": "handle_financial",
        "handle_market": "handle_market",
        "handle_general": "handle_general",
        "handle_research": "handle_research"
    }
)

# Set the entry point
# router.set_entry_point("root")

# Add edges to end
router.add_edge("handle_financial", END)
router.add_edge("handle_market", END)
router.add_edge("handle_general", END)
router.add_edge("handle_research", END)

# Compile the graph
graph = router.compile(checkpointer=checkpointer)
# graph = router.compile()
graph.name = "InterviewRouterGraph"