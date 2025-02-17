"""Shared utility functions for interview graphs."""

from typing import List, Dict, Any
from datetime import datetime
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from shared.utils import load_chat_model
from agent.interview_graph.configuration import InterviewConfiguration

def calculate_confidence_score(confidence_scores: List[float], config: RunnableConfig = None) -> float:
    """Calculate overall confidence score by weighted average."""
    if not confidence_scores:
        return 0.0
        
    # Simple average of all confidence scores
    avg_score = sum(confidence_scores) / len(confidence_scores)
    
    # Round to 3 decimal places
    return round(avg_score, 3)

def generate_executive_summary(data: Dict[str, Any], config: RunnableConfig = None) -> str:
    """Generate executive summary from analysis components"""
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.article_model)
    
    # Generate summary
    system_message = SystemMessage(content=configuration.executive_summary_instructions)
    summary = llm.invoke([system_message, HumanMessage(content=json.dumps(data))])
    
    return summary.content

def combine_risk_analysis(data: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
    """Combine risk analysis from multiple sources"""
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.analysis_model)
    
    # Generate combined risk analysis
    system_message = SystemMessage(content=configuration.risk_analysis_instructions)
    analysis = llm.invoke([system_message, HumanMessage(content=json.dumps(data))])
    
    return {
        "analysis": analysis.content,
        "risk_factors": data.get("risk_factors", []),
        "risk_score": calculate_confidence_score([
            data.get("market_risk", 0),
            data.get("technical_risk", 0),
            data.get("fundamental_risk", 0)
        ])
    }
