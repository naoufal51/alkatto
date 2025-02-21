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
        metadata={"description": "The API key for Tavily search service"},
    )
    alpha_vantage_key: str = field(
        default_factory=lambda: os.environ.get("ALPHA_VANTAGE_KEY", ""),
        metadata={"description": "The API key for Alpha Vantage financial data"},
    )
    finnhub_key: str = field(
        default_factory=lambda: os.environ.get("FINNHUB_KEY", ""),
        metadata={"description": "The API key for Finnhub financial data"},
    )

    # LLM Models
    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4-turbo",
        metadata={"description": "Model for query processing and refinement"},
    )
    analysis_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4-turbo",
        metadata={"description": "Model for data analysis and insights"},
    )
    article_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4-turbo",
        metadata={"description": "Model for article generation"},
    )
    classification_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4-turbo",
        metadata={"description": "Model for question classification"},
    )

    # System Prompts
    classification_instructions: str = field(
        default=prompts.CLASSIFICATION_INSTRUCTIONS,
        metadata={"description": "Instructions for classifying question type"},
    )
    financial_focus_instructions: str = field(
        default=prompts.FINANCIAL_FOCUS_INSTRUCTIONS,
        metadata={"description": "Instructions for focusing financial questions"},
    )
    general_focus_instructions: str = field(
        default=prompts.GENERAL_FOCUS_INSTRUCTIONS,
        metadata={"description": "Instructions for focusing general questions"},
    )
    search_instructions: str = field(
        default=prompts.SEARCH_INSTRUCTIONS,
        metadata={"description": "Instructions for search query generation"},
    )
    stock_analysis_instructions: str = field(
        default=prompts.STOCK_ANALYSIS_INSTRUCTIONS,
        metadata={"description": "Instructions for stock analysis"},
    )
    market_sentiment_instructions: str = field(
        default=prompts.MARKET_SENTIMENT_INSTRUCTIONS,
        metadata={"description": "Instructions for market sentiment analysis"},
    )
    technical_analysis_instructions: str = field(
        default=prompts.TECHNICAL_ANALYSIS_INSTRUCTIONS,
        metadata={"description": "Instructions for technical analysis"},
    )
    price_target_instructions: str = field(
        default=prompts.PRICE_TARGET_INSTRUCTIONS,
        metadata={"description": "Instructions for price target generation"},
    )
    financial_article_instructions: str = field(
        default=prompts.FINANCIAL_ARTICLE_INSTRUCTIONS,
        metadata={"description": "Instructions for financial article writing"},
    )
    general_article_instructions: str = field(
        default=prompts.GENERAL_ARTICLE_INSTRUCTIONS,
        metadata={"description": "Instructions for general article writing"},
    )
    answer_instructions: str = field(
        default=prompts.ANSWER_INSTRUCTIONS,
        metadata={"description": "Instructions for answer generation"},
    )
    executive_summary_instructions: str = field(
        default=prompts.EXECUTIVE_SUMMARY_INSTRUCTIONS,
        metadata={"description": "Instructions for executive summary generation"},
    )
    risk_analysis_instructions: str = field(
        default=prompts.RISK_ANALYSIS_INSTRUCTIONS,
        metadata={"description": "Instructions for risk analysis"},
    )
    confidence_scoring_instructions: str = field(
        default=prompts.CONFIDENCE_SCORING_INSTRUCTIONS,
        metadata={"description": "Instructions for confidence scoring"},
    )
    research_analysis_instructions: str = field(
        default=prompts.RESEARCH_ANALYSIS_INSTRUCTIONS,
        metadata={"description": "Instructions for research paper analysis"},
    )
    research_summary_instructions: str = field(
        default=prompts.RESEARCH_SUMMARY_INSTRUCTIONS,
        metadata={"description": "Instructions for research summary generation"},
    )
    research_focus_instructions: str = field(
        default=prompts.RESEARCH_FOCUS_INSTRUCTIONS,
        metadata={"description": "Instructions for focusing research questions"},
    )
    research_query_instructions: str = field(
        default=prompts.RESEARCH_QUERY_INSTRUCTIONS,
        metadata={"description": "Instructions for generating arXiv search queries"},
    )
    research_confidence_scoring_instructions: str = field(
        default=prompts.RESEARCH_CONFIDENCE_SCORING_INSTRUCTIONS,
        metadata={"description": "Instructions for scoring research paper confidence"},
    )

    # Financial Analysis Parameters
    stock_history_days: int = field(
        default=365,
        metadata={"description": "Number of days of historical stock data to analyze"},
    )
    technical_indicators: list[str] = field(
        default_factory=lambda: ["SMA", "EMA", "RSI", "MACD"],
        metadata={"description": "Technical indicators to include in analysis"},
    )
    sentiment_sources: list[str] = field(
        default_factory=lambda: ["news", "social", "analyst"],
        metadata={"description": "Sources to consider for sentiment analysis"},
    )

    # Search Parameters
    max_search_results: int = field(
        default=5,
        metadata={"description": "Maximum number of search results to process"},
    )
    search_recency_days: int = field(
        default=30,
        metadata={"description": "Maximum age of search results in days"},
    )

    # Article Generation Parameters
    max_article_length: int = field(
        default=2000,
        metadata={"description": "Maximum length of generated articles in words"},
    )
    include_charts: bool = field(
        default=True,
        metadata={"description": "Whether to include charts in financial articles"},
    )

    # Research Parameters
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

    # System Parameters
    parallel_execution: bool = field(
        default=True,
        metadata={"description": "Enable parallel execution of analysis nodes"},
    )
    cache_timeout: int = field(
        default=3600,
        metadata={"description": "Cache timeout in seconds for API responses"},
    )