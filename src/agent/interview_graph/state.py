from typing import Annotated, List, ClassVar, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
import operator

class Analyst(BaseModel):
    # Default values with generic persona
    DEFAULT_NAME: ClassVar[str] = "Generic Analyst"
    DEFAULT_ROLE: ClassVar[str] = "Research Analyst"
    DEFAULT_AFFILIATION: ClassVar[str] = "Research Organization"
    DEFAULT_DESCRIPTION: ClassVar[str] = "Focused on gathering and analyzing information objectively. Interested in collecting factual data and understanding key concepts."

    affiliation: str = Field(
        default=DEFAULT_AFFILIATION,
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(
        default=DEFAULT_NAME,
        description="Name of the analyst."
    )
    role: str = Field(
        default=DEFAULT_ROLE,
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        default=DEFAULT_DESCRIPTION,
        description="Description of the analyst focus, concerns, and motives.",
    )
    
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

    @classmethod
    def get_default(cls) -> "Analyst":
        return cls(
            name=cls.DEFAULT_NAME,
            role=cls.DEFAULT_ROLE,
            affiliation=cls.DEFAULT_AFFILIATION,
            description=cls.DEFAULT_DESCRIPTION
        )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class InterviewState(MessagesState):
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, operator.add] # Source docs
    analyst: Analyst = Field(default_factory=Analyst) # Analyst asking questions with default instance
    interview: str # Interview transcript
    sections: list # Final key we duplicate in outer state for Send() API

    def get(self, key: str, default=None):
        """Override get to ensure analyst has default value"""
        if key == "analyst":
            return super().get(key, Analyst())
        return super().get(key, default)

class OutputState(TypedDict):
    sections: list[str]