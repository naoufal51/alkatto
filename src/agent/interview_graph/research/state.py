"""State definitions for research analysis graph."""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """State for research analysis."""
    messages: Annotated[Sequence[BaseMessage], add_messages]