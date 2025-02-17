"""State definitions for general knowledge graph."""

from typing import Dict, Any, List
from pydantic import BaseModel, Field, field_serializer
from langgraph.graph import MessagesState
from langchain_core.documents import Document

class GeneralState(MessagesState):
    """State for general knowledge graph."""
    web_results: List[Document] = Field(default_factory=list)
    wiki_results: List[Document] = Field(default_factory=list)
    context: List[Document] = Field(default_factory=list)
    analysis: Dict[str, Any] = Field(default_factory=dict)
    
    @field_serializer('web_results', 'wiki_results', 'context')
    def serialize_documents(self, documents: List[Document], _info):
        """Serialize Document objects to dict format."""
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in documents
        ]
