"""General knowledge graph for processing non-financial questions."""

from typing import Dict, Any, List
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from shared.utils import load_chat_model
from agent.interview_graph.configuration import InterviewConfiguration
from shared.retrieval import WebRetriever, WikipediaRetriever
from agent.interview_graph.utils import calculate_confidence_score
from .state import GeneralState, MessagesState

def search_web(state: GeneralState, *, config: RunnableConfig = None) -> Dict[str, Any]:
    """Search web for relevant information."""
    messages = state["messages"]
    
    # Get configuration
    configuration = InterviewConfiguration.from_runnable_config(config)
    
    # Get the last message content
    last_message = messages[-1].content if messages else ""
    
    # Search web
    web_retriever = WebRetriever()
    web_results = web_retriever.search(last_message)
    
    return {"web_results": web_results}

def search_wikipedia(state: GeneralState, *, config: RunnableConfig = None) -> Dict[str, Any]:
    """Search Wikipedia for relevant information."""
    messages = state["messages"]
    
    # Get the last message content
    last_message = messages[-1].content if messages else ""
    
    # Search Wikipedia
    wiki_retriever = WikipediaRetriever()
    wiki_results = wiki_retriever.search(last_message)
    
    return {"wiki_results": wiki_results}

def combine_results(state: GeneralState, *, config: RunnableConfig = None) -> Dict[str, Any]:
    """Combine web and Wikipedia search results."""
    web_results = state.get("web_results", [])
    wiki_results = state.get("wiki_results", [])
    
    # Combine all results
    all_results = web_results + wiki_results
    
    return {"context": all_results}

def analyze_search_results(state: GeneralState, *, config: RunnableConfig = None) -> Dict[str, Any]:
    """Analyze and summarize search results."""
    messages = state["messages"]
    context = state.get("context", [])
    
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.analysis_model)
    
    # Get the last message content
    last_message = messages[-1].content if messages else ""
    
    # Prepare analysis data
    analysis_data = {
        "question": last_message,
        "sources": [
            {
                "content": doc.page_content,
                "url": doc.metadata.get("url", ""),
                "type": "wikipedia" if "wikipedia" in doc.metadata.get("url", "").lower() else "web"
            } for doc in context
        ]
    }
    
    # Generate analysis using LLM
    system_message = SystemMessage(content="""You are an expert analyst. Analyze the search results to:
1. Identify key themes and concepts
2. Extract relevant facts and data points
3. Note any conflicting information
4. Assess the reliability of sources
5. Highlight gaps in the information

Return a JSON object with the following structure:
{
    "key_themes": list[str],
    "facts": list[str],
    "conflicts": list[str],
    "source_reliability": dict[str, float],
    "information_gaps": list[str],
    "analysis_summary": str
}""")
    
    analysis = llm.invoke([
        system_message,
        HumanMessage(content=json.dumps(analysis_data))
    ])
    
    try:
        analysis_result = json.loads(analysis.content)
    except json.JSONDecodeError:
        analysis_result = {
            "key_themes": [],
            "facts": [],
            "conflicts": [],
            "source_reliability": {},
            "information_gaps": [],
            "analysis_summary": analysis.content
        }
    
    return {"analysis": analysis_result}

def write_general_article(state: GeneralState, *, config: RunnableConfig = None):
    """Write a general analysis article"""
    messages = state["messages"]
    context = state.get("context", [])
    analysis = state.get("analysis", {})
    
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.article_model)
    
    # Get the last message content
    last_message = messages[-1].content if messages else ""
    
    # Prepare article data
    article_data = {
        "question": last_message,
        "context": [doc.page_content for doc in context],
        "sources": [doc.metadata.get("url", "") for doc in context],
        "confidence_score": calculate_confidence_score([0.8] * len(context), config),
        "analysis": analysis
    }
    
    # Generate article using LLM
    system_message = SystemMessage(content=configuration.general_article_instructions)
    article = llm.invoke([
        system_message,
        HumanMessage(content=json.dumps(article_data))
    ])
    
    return {"messages": article.content}

def handle_general_question(state: MessagesState, *, config: RunnableConfig = None) -> Dict[str, Any]:
    """Transform MessagesState to GeneralState and handle general knowledge questions."""
    # Transform to general state
    general_state = GeneralState(
        messages=state["messages"],
        context=[],
        analysis={}
    )
    
    # Process with general graph
    result = general_app.invoke(general_state)
    
    # Transform back to messages state
    return {"messages": result.get("messages", [])}

"""Create and return the compiled general knowledge graph."""
general_graph = StateGraph(GeneralState)

# Add nodes
general_graph.add_node("search_web", search_web)
general_graph.add_node("search_wikipedia", search_wikipedia)
general_graph.add_node("combine_results", combine_results)
general_graph.add_node("analyze_results", analyze_search_results)
general_graph.add_node("write_article", write_general_article)

# Add parallel edges for search
general_graph.add_edge("search_web", "combine_results")
general_graph.add_edge("search_wikipedia", "combine_results")

# Add sequential edges for analysis and writing
general_graph.add_edge("combine_results", "analyze_results")
general_graph.add_edge("analyze_results", "write_article")

# Set multiple entry points for parallel execution
general_graph.set_entry_point("search_web")
general_graph.set_entry_point("search_wikipedia")

# Compile the graph
general_app = general_graph.compile()
general_app.name = "GeneralGraph"
