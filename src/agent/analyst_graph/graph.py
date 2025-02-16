
from agent.analyst_graph.state import GenerateAnalystsState, Perspectives
from agent.analyst_graph.configuration import AnalystConfiguration
from typing import Optional, List
from shared.utils import (
    format_docs, 
    load_chat_model,
    Message,
    RunnableConfig,
    StateGraph,
    END,
    START
)
from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    SystemMessage,
    get_buffer_string
)
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver




def create_analysts(state: GenerateAnalystsState, *, config: RunnableConfig = None):
    
    """ Create analysts """
    
    topic=state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback=state.get('human_analyst_feedback', '')
    configuration = AnalystConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.query_model)

        
    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = SystemMessage(content=configuration.analyst_instructions.format(topic=topic,
                                                            human_analyst_feedback=human_analyst_feedback, 
                                                            max_analysts=max_analysts))

    # Generate question 
    analysts = structured_llm.invoke([system_message] + [HumanMessage(content="Generate the set of analysts.")])
    
    # Write the list of analysis to state
    return {"analysts": analysts.analysts}


def human_review_node(state: GenerateAnalystsState):
    """Pauses the graph for human review of the generated analysts list."""
    # The interrupt call sends a JSON payload to the client.
    # On resume, the value passed via Command(resume=...) is returned here.
    analysts_info = [analyst.persona for analyst in state["analysts"]]

    human_feedback = interrupt({
         "generated_analysts": analysts_info,
         "instruction": (
             "Please review the generated analysts list. "
             "If any changes are needed, provide an updated list under the key 'updated_analysts'."
         )
    })
    return {"human_analyst_feedback": human_feedback}


def initiate_all_interviews(state: GenerateAnalystsState):

    """ Conditional edge to initiate all interviews via Send() API or return to create_analysts """    

    # Check if human feedback
    human_analyst_feedback=state.get("human_analyst_feedback", "appove")
    if human_analyst_feedback.lower() != 'approve':
        # Return to create_analysts
        return "create_analysts"

    else:
        return END
    
checkpointer = MemorySaver()

# Define a new graph
analyst_graph = StateGraph(GenerateAnalystsState, config_schema=AnalystConfiguration)
analyst_graph.add_node("create_analysts", create_analysts)
analyst_graph.add_node("human_review_node", human_review_node)

# Flow
analyst_graph.add_edge(START, "create_analysts")
analyst_graph.add_edge("create_analysts", "human_review_node")
analyst_graph.add_conditional_edges("human_review_node", initiate_all_interviews, ['create_analysts',END])


# Compile the interview graph
graph = analyst_graph.compile(checkpointer=checkpointer)
graph.name = "AnalystGraph"