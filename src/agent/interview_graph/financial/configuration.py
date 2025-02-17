"""Configuration for financial analysis graph."""
from typing import Optional
from pydantic import BaseModel

class FinancialConfiguration(BaseModel):
    """Configuration for financial analysis."""
    stock_analysis_model: str = "gpt-4"
    sentiment_analysis_model: str = "gpt-4"
    technical_analysis_model: str = "gpt-4"
    price_target_model: str = "gpt-4"
    article_model: str = "gpt-4"
    max_stock_history_days: int = 30
    max_news_items: int = 10
    technical_indicators: list[str] = ["SMA", "EMA", "RSI", "MACD"]
    alpha_vantage_key: Optional[str] = None
