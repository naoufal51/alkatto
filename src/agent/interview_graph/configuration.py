"""Define the configurable parameters for the interview agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated
import os

from agent.interview_graph import prompts
from shared.configuration import BaseConfiguration

@dataclass(kw_only=True)
class InterviewConfiguration(BaseConfiguration):
    """The configuration for the interview agent."""

    # API Keys
    tavily_api_key: str = field(
        default_factory=lambda: os.environ.get("TAVILY_API_KEY", ""),
        metadata={
            "description": "The API key for Tavily search service"
        },
    )

    # models
    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name."
        },
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name."
        },
    )

    # prompts
    question_instructions: str = field(
        default=prompts.QUESTION_INSTRUCTIONS,
        metadata={"description": "The system prompt used for generating analyst questions."},
    )

    search_instructions: str = field(
        default=prompts.SEARCH_INSTRUCTIONS,
        metadata={"description": "The system prompt used for generating search queries."},
    )

    answer_instructions: str = field(
        default=prompts.ANSWER_INSTRUCTIONS,
        metadata={"description": "The system prompt used for generating expert answers."},
    )

    section_writer_instructions: str = field(
        default=prompts.SECTION_WRITER_INSTRUCTIONS,
        metadata={"description": "The system prompt used for writing report sections."},
    )