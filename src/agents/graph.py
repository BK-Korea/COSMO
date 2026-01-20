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
        # Cache for indexed tickers to avoid re-indexing
        self.indexed_tickers = set()

    def resolve_ticker(self, state: FinancialQAState) -> Dict[str, Any]:
        """Node: Resolve company name or ticker to ticker symbol using LLM"""
        user_input = state.get("original_input") or state["ticker"]

        # Check if it looks like a valid ticker (2-5 chars, uppercase, no spaces)
        if 2 <= len(user_input) <= 5 and user_input.isupper() and ' ' not in user_input:
            # Likely already a ticker
            return {"ticker": user_input}

        try:
            # Use LLM to resolve company name to ticker
            resolve_prompt = f"""You are a US stock market ticker symbol resolver.

User input: "{user_input}"

Task: Convert this to the correct NYSE/NASDAQ ticker symbol.

Important rules:
- Ticker symbols are typically 1-5 uppercase letters (e.g., AAPL, TSLA, MSFT, GOOGL)
- If given a partial company name, find the most well-known matching company
- Return ONLY the ticker symbol, nothing else (no company name, no explanation)

Examples:
Input: "Apple" → Output: AAPL
Input: "apple inc" → Output: AAPL
Input: "Tesla" → Output: TSLA
Input: "tesla motors" → Output: TSLA
Input: "Microsoft" → Output: MSFT
Input: "microsoft corporation" → Output: MSFT
Input: "Archer Aviation" → Output: ACHR
Input: "archer" → Output: ACHR
Input: "nvidia" → Output: NVDA
Input: "amazon" → Output: AMZN
Input: "meta" → Output: META
Input: "facebook" → Output: META
Input: "GOOGL" → Output: GOOGL
Input: "tsla" → Output: TSLA

Now resolve this input: "{user_input}"

Ticker:"""

            ticker_response = self.llm_client.simple_query(resolve_prompt).strip().upper()

            # Extract ticker (remove any extra text, keep only alphanumeric)
            # Split by whitespace or newline, take first token
            ticker = ticker_response.split()[0] if ticker_response else user_input.upper()
            ticker = ''.join(c for c in ticker if c.isalnum())

            # Validate ticker length (reasonable range)
            if not (1 <= len(ticker) <= 5):
                # Fallback to uppercase input
                ticker = user_input.upper().replace(" ", "")[:5]

            return {"ticker": ticker}

        except Exception as e:
            # Fallback: use input as-is
            return {
                "ticker": user_input.upper().replace(" ", "")[:5],
                "error": f"Ticker resolution warning: {str(e)}"
            }

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

        # Skip if already indexed
        if ticker in self.indexed_tickers:
            return {}

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
                # Mark as indexed
                self.indexed_tickers.add(ticker)

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
        """Node: Evaluate response quality at CEO reporting level (NOVA pattern)"""
        if state.get("error"):
            return {}

        response = state.get("response", "")
        query = state["query"]
        ticker = state["ticker"]
        context = state.get("context", "")

        # CEO-level quality evaluation
        evaluation_prompt = f"""You are evaluating a financial analysis report for C-level executives.
Assess whether this response meets CEO reporting standards on a scale of 0-10.

Company: {ticker}
Question: {query}
Response: {response}

Available Context:
{context[:500]}...

Evaluation Criteria for CEO-Level Reports:
1. **Accuracy** (0-2): Data correctness, no misleading claims
2. **Completeness** (0-2): Answers the question fully, covers key aspects
3. **Clarity** (0-2): Clear language, well-structured, no jargon without explanation
4. **Actionability** (0-2): Provides insights executives can act on
5. **Professionalism** (0-2): Appropriate tone, proper citations, executive-ready

Provide detailed evaluation in this exact format:
SCORE: [total score 0-10]
ACCURACY: [score 0-2]
COMPLETENESS: [score 0-2]
CLARITY: [score 0-2]
ACTIONABILITY: [score 0-2]
PROFESSIONALISM: [score 0-2]
FEEDBACK: [specific improvement suggestions if score < {settings.quality_threshold}]
"""

        try:
            evaluation = self.llm_client.simple_query(evaluation_prompt)

            # Parse score
            score = 5.0  # Default score
            feedback = evaluation

            if "SCORE:" in evaluation:
                try:
                    score_line = evaluation.split("SCORE:")[1].split("\n")[0].strip()
                    # Extract just the number
                    score_str = ''.join(c for c in score_line if c.isdigit() or c == '.')
                    if score_str:
                        score = float(score_str)
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

    def regenerate_response(self, state: FinancialQAState) -> Dict[str, Any]:
        """Node: Regenerate response with feedback (max 3 attempts)"""
        query = state["query"]
        context = state.get("context", "")
        ticker = state["ticker"]
        feedback = state.get("quality_feedback", "")
        regenerate_count = state.get("regenerate_count", 0) + 1

        system_prompt = f"""You are a financial analyst assistant specializing in stock market analysis.
You have access to recent information about {ticker} including company data and news.

Your task is to answer the user's question based on the provided context.
This is attempt #{regenerate_count + 1}. Previous attempt was evaluated and found lacking.

FEEDBACK FROM PREVIOUS ATTEMPT:
{feedback}

IMPORTANT: Address the feedback above and improve your response to meet CEO reporting standards:
- Be accurate with data
- Be complete in covering all aspects
- Be clear and professional
- Provide actionable insights
"""

        user_prompt = f"""Context:
{context}

Question: {query}

Please provide an improved, CEO-level answer based on the context and feedback above."""

        try:
            response = self.llm_client.simple_query(
                query=user_prompt,
                system_prompt=system_prompt,
            )

            return {
                "response": response,
                "regenerate_count": regenerate_count,
            }
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "regenerate_count": regenerate_count,
            }

    def should_return_response(self, state: FinancialQAState) -> str:
        """Edge: Decide whether to return response or regenerate (max 3 attempts)"""
        quality_score = state.get("quality_score", 0.0)
        regenerate_count = state.get("regenerate_count", 0)

        # If quality is good enough, return
        if quality_score >= settings.quality_threshold:
            return "end"

        # If we've tried 3 times, give up and return
        if regenerate_count >= 3:
            return "end"

        # Otherwise, regenerate
        return "regenerate"

    def build_graph(self, skip_fetch_index: bool = False, enable_evaluation: bool = False) -> StateGraph:
        """Build the LangGraph workflow

        Args:
            skip_fetch_index: If True, skip fetch_data and index_data nodes (for cached queries)
            enable_evaluation: If True, include quality evaluation step with regeneration loop
        """
        workflow = StateGraph(FinancialQAState)

        if skip_fetch_index:
            # Fast path: only retrieve and generate
            workflow.add_node("retrieve", self.retrieve_context)
            workflow.add_node("generate", self.generate_response)
            workflow.set_entry_point("retrieve")
            workflow.add_edge("retrieve", "generate")

            if enable_evaluation:
                workflow.add_node("evaluate", self.evaluate_quality)
                workflow.add_node("regenerate", self.regenerate_response)
                workflow.add_edge("generate", "evaluate")
                workflow.add_conditional_edges(
                    "evaluate",
                    self.should_return_response,
                    {
                        "end": END,
                        "regenerate": "regenerate",  # Loop back with feedback
                    },
                )
                # After regenerate, go back to evaluate
                workflow.add_edge("regenerate", "evaluate")
            else:
                workflow.add_edge("generate", END)
        else:
            # Full path: resolve ticker → fetch → index → retrieve → generate
            workflow.add_node("resolve_ticker", self.resolve_ticker)
            workflow.add_node("fetch_data", self.fetch_ticker_data)
            workflow.add_node("index_data", self.index_ticker_data)
            workflow.add_node("retrieve", self.retrieve_context)
            workflow.add_node("generate", self.generate_response)

            workflow.set_entry_point("resolve_ticker")
            workflow.add_edge("resolve_ticker", "fetch_data")
            workflow.add_edge("fetch_data", "index_data")
            workflow.add_edge("index_data", "retrieve")
            workflow.add_edge("retrieve", "generate")

            if enable_evaluation:
                workflow.add_node("evaluate", self.evaluate_quality)
                workflow.add_node("regenerate", self.regenerate_response)
                workflow.add_edge("generate", "evaluate")
                workflow.add_conditional_edges(
                    "evaluate",
                    self.should_return_response,
                    {
                        "end": END,
                        "regenerate": "regenerate",  # Loop back with feedback
                    },
                )
                # After regenerate, go back to evaluate
                workflow.add_edge("regenerate", "evaluate")
            else:
                workflow.add_edge("generate", END)

        return workflow.compile()

    def run(self, ticker: str, query: str, enable_evaluation: bool = False) -> Dict[str, Any]:
        """Run the workflow

        Args:
            ticker: Stock ticker symbol or company name (will be resolved)
            query: User question
            enable_evaluation: If True, enable quality evaluation with regeneration (max 3 attempts)
        """
        # Check if ticker is already indexed (only for resolved tickers)
        # For first run with company name, we need to resolve it first
        ticker_upper = ticker.upper()
        skip_fetch_index = ticker_upper in self.indexed_tickers

        graph = self.build_graph(
            skip_fetch_index=skip_fetch_index,
            enable_evaluation=enable_evaluation
        )

        initial_state: FinancialQAState = {
            "ticker": ticker,  # May be company name, will be resolved
            "original_input": ticker,  # Keep original for resolution
            "query": query,
            "ticker_info": None,
            "historical_data": None,
            "financials": None,
            "news": None,
            "retrieved_documents": None,
            "context": None,
            "response": None,
            "regenerate_count": 0,  # Initialize regeneration counter
            "quality_score": None,
            "quality_feedback": None,
            "error": None,
        }

        result = graph.invoke(initial_state)
        return result

    def is_ticker_cached(self, ticker: str) -> bool:
        """Check if ticker data is already cached"""
        return ticker in self.indexed_tickers

    def clear_cache(self):
        """Clear the ticker cache"""
        self.indexed_tickers.clear()


def create_qa_graph() -> FinancialQAGraph:
    """Factory function to create FinancialQAGraph"""
    return FinancialQAGraph()
