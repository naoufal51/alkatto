"""State definitions for market analysis graph."""

from typing import Dict, Any, List
from pydantic import BaseModel, Field, field_serializer
from langgraph.graph import MessagesState
from langchain_core.documents import Document

class MarketState(MessagesState):
    """State for market analysis graph."""
    market_data: List[Document] = Field(default_factory=list)
    competitor_data: List[Document] = Field(default_factory=list)
    trend_data: List[Document] = Field(default_factory=list)
    context: List[Document] = Field(default_factory=list)
    analysis: Dict[str, Any] = Field(default_factory=dict)
    
    @field_serializer('market_data', 'competitor_data', 'trend_data', 'context')
    def serialize_documents(self, documents: List[Document], _info):
        """Serialize Document objects to dict format."""
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in documents
        ]
