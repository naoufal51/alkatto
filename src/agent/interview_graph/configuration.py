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

    # System Parameters
    parallel_execution: bool = field(
        default=True,
        metadata={"description": "Enable parallel execution of analysis nodes"},
    )
    cache_timeout: int = field(
        default=3600,
        metadata={"description": "Cache timeout in seconds for API responses"},
    )