"""Define a simple chatbot agent.

This agent returns a predefined response without using an actual LLM.
"""

from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph


from agent.configuration import Configuration
from agent.state import AgentState
from agent.interview_graph.graph import graph as interview_graph


# async def my_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
#     """Each node does work."""
#     configuration = Configuration.from_runnable_config(config)
#     # configuration = Configuration.from_runnable_config(config)
#     # You can use runtime configuration to alter the behavior of your
#     # graph.
#     return {
#         "changeme": "output from my_node. "
#         f"Configured with {configuration.my_configurable_param}"
#     }

async def conduct_interview(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    result = await interview_graph.ainvoke({"question": state.messages[-1].content})
    return {"sections": result["sections"]}


# Define a new graph
workflow = StateGraph(AgentState, config_schema=Configuration)

# Add the node to the graph
workflow.add_node("conduct_interview", conduct_interview)

# Set the entrypoint as `call_model`
workflow.add_edge(START, "conduct_interview")
workflow.add_edge("conduct_interview", END)

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "New Graph"  # This defines the custom name in LangSmith
