from typing import Annotated, List, Dict, Any, TypedDict, NotRequired
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
import operator


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class StockData(TypedDict):
    symbol: str
    current_price: float
    daily_change: float
    volume: int
    sma20: float
    sma50: float
    rsi: float
    market_cap: float
    high_52w: float
    low_52w: float
    company_info: dict
    recent_news: List[dict]

# Base state remains in the main module
class MessagesState(MessagesState):
    """Base state for all graphs."""
    pass