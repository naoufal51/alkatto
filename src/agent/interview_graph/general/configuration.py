"""Configuration for general knowledge graph."""
from typing import Optional
from pydantic import BaseModel

class GeneralConfiguration(BaseModel):
    """Configuration for general knowledge analysis."""
    search_model: str = "gpt-4"
    article_model: str = "gpt-4"
    max_search_results: int = 5
    max_wiki_results: int = 3
    tavily_api_key: Optional[str] = None
    min_confidence_threshold: float = 0.7
