"""State definitions for financial analysis graph."""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

class SearchQuery:
    """Search query for stock data"""
    query: str
    company_info: dict
    recent_news: List[dict]

class FinancialState(MessagesState):
    """State for financial analysis graph."""
    stock_data: Dict[str, Any] = Field(default_factory=dict)
    market_sentiment: Dict[str, Any] = Field(default_factory=dict)
    technical_analysis: Dict[str, Any] = Field(default_factory=dict)
    price_targets: Dict[str, Any] = Field(default_factory=dict)
    article: Dict[str, Any] = Field(default_factory=dict)
