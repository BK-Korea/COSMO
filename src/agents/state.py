"""State definitions for LangGraph agents (NOVA pattern)"""
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.documents import Document


class FinancialQAState(TypedDict):
    """State for financial Q&A workflow"""

    # User input
    ticker: str
    query: str

    # Retrieved data
    ticker_info: Optional[Dict[str, Any]]
    historical_data: Optional[Any]
    financials: Optional[Dict[str, Any]]
    news: Optional[List[Dict[str, Any]]]

    # Vector store context
    retrieved_documents: Optional[List[Document]]
    context: Optional[str]

    # LLM response
    response: Optional[str]

    # Quality evaluation
    quality_score: Optional[float]
    quality_feedback: Optional[str]

    # Error handling
    error: Optional[str]
