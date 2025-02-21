"""Research graph for processing arXiv paper requests."""

import os
from typing import Annotated, Literal, Sequence, TypedDict
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain.tools.retriever import create_retriever_tool
import arxiv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
import logging
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)

# Initialize text splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
embeddings = OpenAIEmbeddings()

# Define agent state
class AgentState(TypedDict):
    """State for research analysis."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str

# Initialize empty vectorstore
vectorstore = Chroma(
    collection_name="research-papers",
    embedding_function=embeddings,
    persist_directory="./chroma_db"  # Add persist directory
)
# Configure retriever with specific search parameters
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        'k': 10, 
        'fetch_k': 50
    }
)

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_papers",
    "Search and retrieve information from research papers"
)

def format_arxiv_query(query: str) -> str:
    """Format a natural language query into a simple ArXiv search query using LLM."""
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    prompt = PromptTemplate(
        template="""Convert this natural language query into a simple ArXiv search query.

Guidelines:
- Keep the query simple and direct
- Use only the most important keywords
- Add quotes around multi-word phrases
- Remove unnecessary words (like "papers about", "research on", etc)
- Do not use any special operators (no AND, OR, etc)
- Do not use any field specifiers (no ti:, abs:, etc)
- Do not use category filters

Examples:
"Latest papers about transformers in NLP" -> "transformer neural language processing"
"Quantum computing error correction" -> "quantum error correction"
"Deep learning for computer vision" -> "deep learning computer vision"

Natural language query: {query}

ArXiv search query:""",
        input_variables=["query"]
    )
    
    chain = prompt | model | StrOutputParser()
    arxiv_query = chain.invoke({"query": query})
    return arxiv_query.strip()

def index_papers(state):
    """Node that indexes ArXiv papers based on the query."""
    print("---INDEX PAPERS---")
    state["query"] = state["messages"][0].content
    query = state["query"]
    
    # Format query for ArXiv using LLM
    arxiv_query = format_arxiv_query(query)
    print(f"Original query: {query}")
    print(f"ArXiv query: {arxiv_query}")
    
    # Initialize ArXiv client
    logging.info("Initializing ArXiv client")
    client = arxiv.Client()
    
    # Create search parameters
    search = arxiv.Search(
        query=arxiv_query,
        max_results=100,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    # Load papers from ArXiv
    logging.info("Starting ArXiv search")
    results = list(client.results(search))
    logging.info(f"Loaded {len(results)} documents from ArXiv")

    if not results:
        print("No documents found for query:", arxiv_query)
        return {"messages": [HumanMessage(content="I couldn't find any research papers matching your query. Could you try rephrasing it or being more specific?")]}
    
    # Print metadata for debugging
    print("\nPaper Summaries:")
    for i, result in enumerate(results):
        print(f"\nPaper {i+1}:")
        print(f"Title: {result.title}")
        print(f"Authors: {', '.join(str(author) for author in result.authors)}")
        print(f"Summary: {result.summary}")
        print(f"Published: {result.published}")
    
    # Create document splits from metadata summaries
    summary_docs = []
    for result in results:
        summary_text = f"""Title: {result.title}
Authors: {', '.join(str(author) for author in result.authors)}
Published: {result.published}
Summary: {result.summary}"""
        
        summary_doc = Document(
            page_content=summary_text,
            metadata={
                'Title': result.title,
                'Authors': ', '.join(str(author) for author in result.authors),
                'Published': str(result.published),
                'entry_id': result.entry_id,
                'pdf_url': result.pdf_url,
                'primary_category': result.primary_category
            }
        )
        summary_docs.append(summary_doc)
    
    try:
        global vectorstore, retriever
        
        # Log initial state
        logging.info(f"Initial Chroma collection count: {vectorstore._collection.count()}")
        logging.info(f"Initial Chroma collection name: {vectorstore._collection.name}")
        
        # Add summaries to vectorstore
        print(f"Adding {len(summary_docs)} new documents to vectorstore...")
        vectorstore.add_documents(summary_docs)
        logging.info(f"After adding documents - Collection count: {vectorstore._collection.count()}")
        print(f"Successfully indexed {len(summary_docs)} paper summaries")
        
        # Update retriever to use vectorstore
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                'k': 10, 
                'fetch_k': 50
            }
        )
        
        # Log final state
        logging.info("Chroma collection details:")
        logging.info(f"- Collection name: {vectorstore._collection.name}")
        logging.info(f"- Document count: {vectorstore._collection.count()}")
        
        return state
    except Exception as e:
        logging.error(f"Error adding documents to vectorstore: {str(e)}")
        logging.error(f"Chroma collection state at error: {vectorstore._collection.count()} documents")
        raise
        
 
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """Grade the relevance of retrieved documents."""
    print("---CHECK RELEVANCE---")
    
    class Grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    
    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    llm_with_tool = model.with_structured_output(Grade)
    
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a research question.

Here are the retrieved documents:
{context}

Here is the research question:
{question}

If the documents contain relevant information, methodology, findings, or technical details related to the research question,
grade it as relevant. Consider both direct keyword matches and semantic/conceptual relevance.

Give a binary score 'yes' or 'no' to indicate whether the documents are relevant to the question.""",
        input_variables=["context", "question"]
    )
    
    chain = prompt | llm_with_tool
    
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score
    
    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"

def agent(state):
    """Agent node that decides what to do next."""
    print("---CALL AGENT---")
    messages = state["messages"]
    
    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    model = model.bind_tools([retriever_tool])
    response = model.invoke(messages)
    return {"messages": [response]}

def rewrite(state):
    """Transform the query to produce a better research question."""
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    
    msg = [
        HumanMessage(
            content=f"""Look at the research question and try to reason about the underlying semantic intent and research goals.

Here is the initial question:
---
{question}
---

Formulate an improved research question that will help find more relevant papers:"""
        )
    ]
    
    model = ChatOpenAI(temperature=0, model="gpt-4o")
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    """Generate research analysis and summary using RAG with APA citations."""
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    
    # RAG-optimized research analysis prompt with APA citations
    prompt = PromptTemplate(
        template="""You are a research analyst tasked with answering questions based on the provided research papers. Use APA style for all citations and references.

CONTEXT PAPERS:
{context}

QUESTION:
{question}

GUIDELINES FOR ANALYSIS:

1. APA Citation Guidelines:
- In-text citations: (Author et al., Year) for 3+ authors, (Author & Author, Year) for 2
- For direct quotes: (Author et al., Year, p. X) or section name if no page number
- Multiple sources: List citations alphabetically (Author et al., Year; Author & Author, Year)
- First citation: Include all authors up to 3, then et al.
- Subsequent citations: Use et al. for 3+ authors

2. Source Attribution:
- Every claim must have an in-text citation
- Group multiple related citations in one parenthetical
- For synthesis, cite all relevant sources
- Use author-date format consistently

3. Direct Quotations (APA Style):
Short quotes (< 40 words):
"Quote" (Author et al., Year, p. X)

Block quotes (≥ 40 words):
Indent and cite: (Author et al., Year, p. X)

4. Multiple Source Citations:
- For similar findings:
Several studies (Author et al., Year; Author & Author, Year) found...
- For contrasting findings:
While Author et al. (Year) found X, Author and Author (Year) demonstrated Y...

5. Reference List Format (APA 7th Edition):
REQUIRED components for each reference:
a) Authors:
   - List all authors (up to 20)
   - Use last name, followed by initials
   - Use & for last author
   - Example: Smith, J. D., Johnson, R. M., & Williams, K. L.

b) Publication Date:
   - (YYYY, Month DD)
   - For preprints: (YYYY, Month DD). [Preprint]

c) Title:
   - Article title in sentence case
   - No quotation marks
   - Only capitalize first word and proper nouns

d) Source Information:
   - arXiv section/category
   - arXiv identifier
   - DOI (if available)

Complete Reference Format:
Author, A. A., Author, B. B., & Author, C. C. (YYYY, Month DD). Title of the article in sentence case. arXiv [Category]. https://doi.org/[DOI] or https://arxiv.org/abs/[identifier]

Example Reference:
Smith, J. D., Johnson, R. M., & Williams, K. L. (2023, March 15). Advances in transformer architectures for natural language processing. arXiv [cs.CL]. https://arxiv.org/abs/2303.12345

RESPONSE FORMAT:

Executive Summary:
[Concise answer with appropriate in-text citations]

Detailed Analysis:
1. Main Findings:
   - Finding with citation (Author et al., Year)
   - Supporting evidence with page numbers/sections
   - Integration of multiple sources

2. Methodological Approaches:
   - Methods used (Author et al., Year)
   - Implementation details with citations
   - Comparative analysis of approaches

3. Synthesis of Evidence:
   - Patterns across studies
   - Conflicting findings
   - Chronological development

4. Limitations and Future Directions:
   - Study-specific limitations (Author et al., Year)
   - General research gaps
   - Proposed future work

References:
[Full APA-style reference list]
- Must include ALL cited works
- Must be alphabetically ordered by first author's surname
- Must include ALL required components (authors, date, title, source)
- Must follow exact APA 7th Edition format
- Must include proper URLs/DOIs
- Must include arXiv categories
- Must be properly indented with hanging indent

Remember:
- EVERY paper cited in the text MUST have a complete reference entry
- References MUST include ALL required components
- References MUST be in perfect APA 7th Edition format
- References section MUST be at the end
- References MUST be alphabetically ordered
- Each reference MUST include proper arXiv identifiers and categories

Analysis:""",
        input_variables=["context", "question"]
    )
    
    # LLM setup with specific RAG parameters
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        streaming=True
    )
    
    # Chain setup
    chain = prompt | llm | StrOutputParser()
    
    # Generate response
    response = chain.invoke({"context": docs, "question": question})
    return {"messages": [HumanMessage(content=response)]}

# Create the research graph
workflow = StateGraph(AgentState)

# Define nodes
workflow.add_node("index", index_papers)
workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

# Add edges
workflow.add_edge(START, "index")
workflow.add_edge("index", "agent")

# Add conditional edges for agent decision
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END
    }
)

# Add conditional edges for document grading
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile the graph
research_app = workflow.compile()
research_app.name = "ResearchGraph"

