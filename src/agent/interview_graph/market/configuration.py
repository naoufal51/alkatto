"""Configuration settings for market analysis graph."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from .prompts import (
    MARKET_TRENDS_TEMPLATE,
    MARKET_SENTIMENT_TEMPLATE,
    MARKET_REPORT_TEMPLATE
)

class MarketConfiguration(BaseModel):
    """Configuration for market analysis processing."""
    market_data_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for market data analysis"
    )
    competitor_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for competitor analysis"
    )
    trend_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for trend analysis"
    )
    analysis_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for comprehensive market analysis"
    )
    report_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for report generation"
    )
    market_data_instructions: str = Field(
        default="market_data_instructions",
        description="Instructions for market data analysis"
    )
    competitor_instructions: str = Field(
        default="competitor_analysis_instructions",
        description="Instructions for competitor analysis"
    )
    trend_instructions: str = Field(
        default="trend_analysis_instructions",
        description="Instructions for trend analysis"
    )
    analysis_instructions: str = Field(
        default="market_analysis_instructions",
        description="Instructions for comprehensive market analysis"
    )
    report_instructions: str = Field(
        default="market_report_instructions",
        description="Instructions for report generation"
    )
    
    # Prompt templates
    market_trends_template: str = Field(
        default=MARKET_TRENDS_TEMPLATE,
        description="Template for market trends analysis"
    )

    market_sentiment_template: str = Field(
        default=MARKET_SENTIMENT_TEMPLATE,
        description="Template for market sentiment analysis"
    )

    market_report_template: str = Field(
        default=MARKET_REPORT_TEMPLATE,
        description="Template for market report generation"
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[Dict[str, Any]] = None) -> "MarketConfiguration":
        """Create MarketConfiguration from a runnable config dictionary."""
        if config is None:
            config = {}
        return cls(**config.get("market_config", {}))
