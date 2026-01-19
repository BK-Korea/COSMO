"""LangGraph workflow for financial Q&A (following NOVA pattern)"""
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agents.state import FinancialQAState
from src.yahoo_finance import YahooFinanceService
from src.vectorstore.chroma_store import ChromaStore
from src.llm.glm_client import GLMClient
from config.config import settings


class FinancialQAGraph:
    """LangGraph-based financial Q&A workflow"""

    def __init__(self):
        self.yf_service = YahooFinanceService()
        self.vector_store = ChromaStore()
        self.llm_client = GLMClient()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def fetch_ticker_data(self, state: FinancialQAState) -> Dict[str, Any]:
        """Node: Fetch ticker data from Yahoo Finance"""
        ticker = state["ticker"]

        try:
            # Fetch comprehensive ticker info
            ticker_info = self.yf_service.get_ticker_info(ticker)
            historical_data = self.yf_service.get_historical_data(ticker, period="1mo")
            news = self.yf_service.get_news(ticker)

            return {
                "ticker_info": ticker_info,
                "historical_data": historical_data,
                "news": news,
                "error": None,
            }
        except Exception as e:
            return {
                "error": f"Failed to fetch ticker data: {str(e)}",
            }

    def index_ticker_data(self, state: FinancialQAState) -> Dict[str, Any]:
        """Node: Index ticker data into vector store"""
        if state.get("error"):
            return {}

        ticker = state["ticker"]
        ticker_info = state.get("ticker_info", {})
        news = state.get("news", [])

        try:
            documents = []

            # Create document from ticker info
            if ticker_info:
                ticker_text = f"""
Company: {ticker_info.get('name', 'N/A')} ({ticker_info.get('symbol', ticker)})
Sector: {ticker_info.get('sector', 'N/A')}
Industry: {ticker_info.get('industry', 'N/A')}
Current Price: ${ticker_info.get('current_price', 'N/A')}
Market Cap: ${ticker_info.get('market_cap', 'N/A')}
P/E Ratio: {ticker_info.get('pe_ratio', 'N/A')}
Description: {ticker_info.get('description', 'N/A')}
"""
                documents.append(
                    Document(
                        page_content=ticker_text,
                        metadata={"ticker": ticker, "type": "company_info"},
                    )
                )

            # Create documents from news
            for i, article in enumerate(news[:10]):  # Limit to 10 most recent
                news_text = f"""
Title: {article.get('title', 'N/A')}
Publisher: {article.get('publisher', 'N/A')}
Summary: {article.get('summary', 'N/A')}
"""
                documents.append(
                    Document(
                        page_content=news_text,
                        metadata={
                            "ticker": ticker,
                            "type": "news",
                            "link": article.get("link", ""),
                        },
                    )
                )

            # Split and add to vector store
            if documents:
                split_docs = self.text_splitter.split_documents(documents)
                self.vector_store.add_documents(split_docs)

            return {}
        except Exception as e:
            return {
                "error": f"Failed to index data: {str(e)}",
            }

    def retrieve_context(self, state: FinancialQAState) -> Dict[str, Any]:
        """Node: Retrieve relevant context from vector store"""
        if state.get("error"):
            return {}

        query = state["query"]
        ticker = state["ticker"]

        try:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(
                query=query,
                k=settings.retrieval_top_k,
                filter={"ticker": ticker},
            )

            # Combine documents into context
            context = "\n\n".join([doc.page_content for doc in docs])

            return {
                "retrieved_documents": docs,
                "context": context,
            }
        except Exception as e:
            return {
                "error": f"Failed to retrieve context: {str(e)}",
            }

    def generate_response(self, state: FinancialQAState) -> Dict[str, Any]:
        """Node: Generate response using LLM"""
        if state.get("error"):
            return {"response": f"Error: {state['error']}"}

        query = state["query"]
        context = state.get("context", "")
        ticker = state["ticker"]

        system_prompt = f"""You are a financial analyst assistant specializing in stock market analysis.
You have access to recent information about {ticker} including company data and news.

Your task is to answer the user's question based on the provided context.
Be specific, cite sources when available, and provide actionable insights.
If the context doesn't contain enough information, say so clearly.
"""

        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""

        try:
            response = self.llm_client.simple_query(
                query=user_prompt,
                system_prompt=system_prompt,
            )

            return {"response": response}
        except Exception as e:
            return {"response": f"Error generating response: {str(e)}"}

    def evaluate_quality(self, state: FinancialQAState) -> Dict[str, Any]:
        """Node: Evaluate response quality (NOVA pattern)"""
        if state.get("error"):
            return {}

        response = state.get("response", "")
        query = state["query"]

        # Simple quality evaluation (can be enhanced with LLM-based scoring like NOVA)
        evaluation_prompt = f"""Evaluate the quality of this financial analysis response on a scale of 0-10.

Question: {query}
Response: {response}

Provide a score (0-10) and brief feedback. Format:
SCORE: [number]
FEEDBACK: [your feedback]
"""

        try:
            evaluation = self.llm_client.simple_query(evaluation_prompt)

            # Parse score (simplified)
            score = 8.0  # Default score
            feedback = evaluation

            if "SCORE:" in evaluation:
                try:
                    score_line = evaluation.split("SCORE:")[1].split("\n")[0].strip()
                    score = float(score_line)
                except:
                    pass

            return {
                "quality_score": score,
                "quality_feedback": feedback,
            }
        except Exception as e:
            return {
                "quality_score": 5.0,
                "quality_feedback": f"Evaluation failed: {str(e)}",
            }

    def should_return_response(self, state: FinancialQAState) -> str:
        """Edge: Decide whether to return response based on quality"""
        quality_score = state.get("quality_score", 0.0)

        if quality_score >= settings.quality_threshold:
            return "end"
        else:
            return "regenerate"

    def build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(FinancialQAState)

        # Add nodes
        workflow.add_node("fetch_data", self.fetch_ticker_data)
        workflow.add_node("index_data", self.index_ticker_data)
        workflow.add_node("retrieve", self.retrieve_context)
        workflow.add_node("generate", self.generate_response)
        workflow.add_node("evaluate", self.evaluate_quality)

        # Define edges
        workflow.set_entry_point("fetch_data")
        workflow.add_edge("fetch_data", "index_data")
        workflow.add_edge("index_data", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "evaluate")

        # Conditional edge based on quality
        workflow.add_conditional_edges(
            "evaluate",
            self.should_return_response,
            {
                "end": END,
                "regenerate": END,  # For now, just end (could regenerate)
            },
        )

        return workflow.compile()

    def run(self, ticker: str, query: str) -> Dict[str, Any]:
        """Run the workflow"""
        graph = self.build_graph()

        initial_state: FinancialQAState = {
            "ticker": ticker,
            "query": query,
            "ticker_info": None,
            "historical_data": None,
            "financials": None,
            "news": None,
            "retrieved_documents": None,
            "context": None,
            "response": None,
            "quality_score": None,
            "quality_feedback": None,
            "error": None,
        }

        result = graph.invoke(initial_state)
        return result


def create_qa_graph() -> FinancialQAGraph:
    """Factory function to create FinancialQAGraph"""
    return FinancialQAGraph()
