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
from langchain_core.output_parsers import StrOutputParser

from agent.interview_graph.state import InterviewState, Analyst, SearchQuery, OutputState
from agent.interview_graph.configuration import InterviewConfiguration
from shared.retrieval import WebRetriever, WikipediaRetriever

def generate_question(state: InterviewState, *, config: RunnableConfig = None):
    """ Node to generate a question """
    # Use default analyst if none provided
    analyst = state.get("analyst", Analyst.get_default())
    messages = state["messages"]
    
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.query_model)
    
    # Generate question 
    system_message = SystemMessage(content=configuration.question_instructions.format(goals=analyst.persona))
    question = llm.invoke([system_message] + messages)
        
    # Write messages to state
    return {"messages": [question]}

def search_web(state: InterviewState, *, config: RunnableConfig = None):
    """ Retrieve docs from web search """
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.query_model)
    
    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([SystemMessage(content=configuration.search_instructions)] + state['messages'])
    
    # Search
    retriever = WebRetriever(max_results=3)
    search_docs = retriever.search(search_query.search_query, config=config)
    formatted_search_docs = format_docs(search_docs)

    return {"context": [formatted_search_docs]} 

def search_wikipedia(state: InterviewState, *, config: RunnableConfig = None):
    """ Retrieve docs from wikipedia """
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.query_model)
    
    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([SystemMessage(content=configuration.search_instructions)] + state['messages'])
    
    # Search
    retriever = WikipediaRetriever(max_docs=2)
    search_docs = retriever.search(search_query.search_query)
    formatted_search_docs = format_docs(search_docs)

    return {"context": [formatted_search_docs]} 

def generate_answer(state: InterviewState, *, config: RunnableConfig = None):
    """ Node to answer a question """
    analyst = state.get("analyst", Analyst())  # Use get with default
    messages = state["messages"]
    context = state["context"]

    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.response_model)
    
    # Answer question
    system_message = SystemMessage(content=configuration.answer_instructions.format(goals=analyst.persona, context=context))
    answer = llm.invoke([system_message] + messages)
            
    # Name the message as coming from the expert
    answer.name = "expert"
    
    # Append it to state
    return {"messages": [answer]}

# def save_interview(state: InterviewState):
#     """ Save interviews """
#     messages = state["messages"]
#     interview = "\n".join(f"{m.role}: {m.content}" for m in messages)
#     return {"interview": interview}

def save_interview(state: InterviewState):
    
    """ Save interviews """

    # Get messages
    messages = state["messages"]
    
    # Convert interview to a string
    interview = get_buffer_string(messages)
    
    # Save to interviews key
    return {"interview": interview}

def route_messages(state: InterviewState, 
                   name: str = "expert"):

    """ Route between question and answer """
    
    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # Check the number of expert answers 
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair 
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

def write_section(state: InterviewState, *, config: RunnableConfig = None):
    """ Node to write a section """
    interview = state["interview"]
    context = state["context"]
    # analyst = state["analyst"]
    analyst = state.get("analyst", Analyst())
   
    # Get configuration and LLM
    configuration = InterviewConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.response_model)
    
    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = SystemMessage(content=configuration.section_writer_instructions.format(focus=analyst.description))
    section = llm.invoke([
        system_message,
        HumanMessage(content=f"Use this source to write your section: {context}")
    ]) 
                
    # Append it to state
    return {"sections": [section.content]}

# Add nodes and edges 
interview_builder = StateGraph(InterviewState,  input=InterviewState, output=OutputState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages,['ask_question','save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# Compile the interview graph
graph = interview_builder.compile()
graph.name = "InterviewGraph"
