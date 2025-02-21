"""Configuration for the research subgraph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from agent.interview_graph.research import prompts
from shared.configuration import BaseConfiguration

@dataclass(kw_only=True)
class ResearchConfiguration(BaseConfiguration):
    """Configuration for the research analysis subgraph."""

    # LLM Models
    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4-turbo",
        metadata={"description": "Model for query processing and refinement"},
    )
    analysis_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4-turbo",
        metadata={"description": "Model for research paper analysis"},
    )
    summary_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4-turbo",
        metadata={"description": "Model for research summary generation"},
    )

    # System Prompts
    focus_instructions: str = field(
        default=prompts.FOCUS_INSTRUCTIONS,
        metadata={"description": "Instructions for focusing research questions"},
    )
    query_instructions: str = field(
        default=prompts.QUERY_INSTRUCTIONS,
        metadata={"description": "Instructions for generating arXiv search queries"},
    )
    analysis_instructions: str = field(
        default=prompts.ANALYSIS_INSTRUCTIONS,
        metadata={"description": "Instructions for research paper analysis"},
    )
    summary_instructions: str = field(
        default=prompts.SUMMARY_INSTRUCTIONS,
        metadata={"description": "Instructions for research summary generation"},
    )
    confidence_scoring_instructions: str = field(
        default=prompts.CONFIDENCE_SCORING_INSTRUCTIONS,
        metadata={"description": "Instructions for scoring research paper confidence"},
    )

    # Search Parameters
    max_arxiv_results: int = field(
        default=10,
        metadata={"description": "Maximum number of arXiv papers to retrieve"},
    )
    arxiv_categories: list[str] = field(
        default_factory=lambda: [
            "cs.AI",    # Artificial Intelligence
            "cs.LG",    # Machine Learning
            "cs.CL",    # Computational Linguistics
            "cs.CV",    # Computer Vision
            "cs.NE",    # Neural and Evolutionary Computing
            "cs.RO",    # Robotics
            "cs.SE",    # Software Engineering
            "quant-ph", # Quantum Computing
            "stat.ML"   # Statistics - Machine Learning
        ],
        metadata={"description": "Default arXiv categories to search in"},
    )
    research_recency_days: int = field(
        default=365,
        metadata={"description": "Maximum age of research papers in days"},
    )
    min_citation_count: int = field(
        default=5,
        metadata={"description": "Minimum citation count for considering papers"},
    )
    include_preprints: bool = field(
        default=True,
        metadata={"description": "Whether to include preprints in search results"},
    )
    max_paper_length: int = field(
        default=30,
        metadata={"description": "Maximum paper length in pages to consider"},
    )

    # Ranking Parameters
    citation_weight: float = field(
        default=0.3,
        metadata={"description": "Weight given to citation count in paper ranking"},
    )
    recency_weight: float = field(
        default=0.4,
        metadata={"description": "Weight given to paper recency in ranking"},
    )
    relevance_weight: float = field(
        default=0.3,
        metadata={"description": "Weight given to search relevance in ranking"},
    )

    # Output Parameters
    max_summary_length: int = field(
        default=2000,
        metadata={"description": "Maximum length of research summaries in words"},
    )
    include_citations: bool = field(
        default=True,
        metadata={"description": "Whether to include citation information in summaries"},
    )
    include_abstracts: bool = field(
        default=True,
        metadata={"description": "Whether to include paper abstracts in summaries"},
    )

    # System Parameters
    parallel_execution: bool = field(
        default=True,
        metadata={"description": "Enable parallel execution of analysis nodes"},
    )
    cache_timeout: int = field(
        default=3600,
        metadata={"description": "Cache timeout in seconds for API responses"},
    )