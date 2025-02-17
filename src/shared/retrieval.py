"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.
"""

import os
from contextlib import contextmanager
from typing import Generator, List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from shared.configuration import BaseConfiguration

class WebRetriever:
    """Web search retriever using Tavily."""
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        
    def search(self, query: str, *, config: RunnableConfig = None) -> List[Document]:
        from langchain_community.tools.tavily_search import TavilySearchResults
        from agent.interview_graph.configuration import InterviewConfiguration
        
        if config:
            configuration = InterviewConfiguration.from_runnable_config(config)
            os.environ["TAVILY_API_KEY"] = configuration.tavily_api_key
            
        tavily_search = TavilySearchResults(max_results=self.max_results)
        search_results = tavily_search.invoke(query)
        return [
            Document(
                page_content=doc["content"],
                metadata={"url": doc["url"]}
            ) for doc in search_results
        ]

class WikipediaRetriever:
    """Wikipedia retriever."""
    def __init__(self, max_docs: int = 2):
        self.max_docs = max_docs
        
    def search(self, query: str) -> List[Document]:
        from langchain_community.document_loaders import WikipediaLoader
        return WikipediaLoader(query=query, load_max_docs=self.max_docs).load()

class StockRetriever:
    """Stock data retriever using Alpha Vantage and Finnhub."""
    
    def __init__(self):
        self.alpha_vantage_key = None
        self.finnhub_key = None
        
    def _get_keys(self, config: RunnableConfig = None):
        """Get API keys from configuration."""
        from agent.interview_graph.configuration import InterviewConfiguration
        
        if config:
            configuration = InterviewConfiguration.from_runnable_config(config)
            self.alpha_vantage_key = configuration.alpha_vantage_key
            self.finnhub_key = configuration.finnhub_key
    
    def get_stock_data(self, symbol: str) -> dict:
        """Get stock data from Alpha Vantage."""
        # Mock data for now
        return {
            "symbol": symbol,
            "current_price": 150.00,
            "daily_change": 2.5,
            "volume": 1000000,
            "market_cap": 2000000000,
            "company_overview": f"Mock company overview for {symbol}",
            "financial_metrics": {
                "pe_ratio": 20.5,
                "eps": 7.5,
                "revenue": 1000000000
            },
            "confidence": 0.8
        }
    
    def get_market_sentiment(self, symbol: str) -> dict:
        """Get market sentiment data from Finnhub."""
        # Mock data for now
        return {
            "symbol": symbol,
            "market_analysis": f"Mock market analysis for {symbol}",
            "sentiment_summary": "Bullish",
            "risks": ["Market Risk 1", "Market Risk 2"],
            "confidence": 0.75
        }
    
    def get_technical_indicators(self, symbol: str) -> dict:
        """Get technical indicators from Alpha Vantage."""
        # Mock data for now
        return {
            "symbol": symbol,
            "indicators": {
                "rsi": 65,
                "macd": "bullish",
                "sma_20": 145.00,
                "sma_50": 140.00
            },
            "analysis": f"Mock technical analysis for {symbol}",
            "risks": ["Technical Risk 1", "Technical Risk 2"],
            "confidence": 0.7
        }

## Encoder constructors


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case "cohere":
            from langchain_cohere import CohereEmbeddings

            return CohereEmbeddings(model=model)  # type: ignore
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@contextmanager
def make_elastic_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    connection_options = {}
    if configuration.retriever_provider == "elastic-local":
        connection_options = {
            "es_user": os.environ["ELASTICSEARCH_USER"],
            "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
        }

    else:
        connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}

    vstore = ElasticsearchStore(
        **connection_options,  # type: ignore
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )

    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_pinecone_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_pinecone import PineconeVectorStore

    vstore = PineconeVectorStore.from_existing_index(
        os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
    )
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_mongodb_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="langgraph_retrieval_agent.default",
        embedding=embedding_model,
    )
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    match configuration.retriever_provider:
        case "elastic" | "elastic-local":
            with make_elastic_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "mongodb":
            with make_mongodb_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
