"""LangGraph workflow for financial Q&A (following NOVA pattern)"""
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

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
- If given a partial company name (in any language including Korean), find the most well-known matching company
- Return ONLY the ticker symbol, nothing else (no company name, no explanation)

Examples (English and Korean):
Input: "Apple" → Output: AAPL
Input: "애플" → Output: AAPL
Input: "apple inc" → Output: AAPL
Input: "Tesla" → Output: TSLA
Input: "테슬라" → Output: TSLA
Input: "tesla motors" → Output: TSLA
Input: "Microsoft" → Output: MSFT
Input: "마이크로소프트" → Output: MSFT
Input: "microsoft corporation" → Output: MSFT
Input: "Archer Aviation" → Output: ACHR
Input: "archer" → Output: ACHR
Input: "nvidia" → Output: NVDA
Input: "엔비디아" → Output: NVDA
Input: "amazon" → Output: AMZN
Input: "아마존" → Output: AMZN
Input: "meta" → Output: META
Input: "메타" → Output: META
Input: "facebook" → Output: META
Input: "페이스북" → Output: META
Input: "GOOGL" → Output: GOOGL
Input: "구글" → Output: GOOGL
Input: "google" → Output: GOOGL
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
            news = self.yf_service.get_news(ticker, limit=10)

            # Fetch financial statements (income statement, balance sheet, cash flow)
            try:
                financials = self.yf_service.get_financials(ticker)
            except Exception as fin_error:
                financials = None
                print(f"Warning: Could not fetch financials: {fin_error}")

            # Fetch analyst recommendations
            try:
                recommendations = self.yf_service.get_recommendations(ticker)
            except Exception as rec_error:
                recommendations = None
                print(f"Warning: Could not fetch recommendations: {rec_error}")

            # Fetch major shareholders
            try:
                major_holders = self.yf_service.get_major_holders(ticker)
            except Exception as holder_error:
                major_holders = None
                print(f"Warning: Could not fetch major holders: {holder_error}")

            # Fetch institutional holders
            try:
                institutional_holders = self.yf_service.get_institutional_holders(ticker)
            except Exception as inst_error:
                institutional_holders = None
                print(f"Warning: Could not fetch institutional holders: {inst_error}")

            # Fetch insider transactions
            try:
                insider_transactions = self.yf_service.get_insider_transactions(ticker)
            except Exception as insider_error:
                insider_transactions = None
                print(f"Warning: Could not fetch insider transactions: {insider_error}")

            return {
                "ticker_info": ticker_info,
                "historical_data": historical_data,
                "news": news,
                "financials": financials,
                "recommendations": recommendations,
                "major_holders": major_holders,
                "institutional_holders": institutional_holders,
                "insider_transactions": insider_transactions,
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
        financials = state.get("financials", {})
        recommendations = state.get("recommendations")
        major_holders = state.get("major_holders")
        institutional_holders = state.get("institutional_holders")
        insider_transactions = state.get("insider_transactions")

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

            # Create documents from financial statements
            if financials:
                # Income Statement (Revenue, Expenses, Net Income)
                income_stmt = financials.get('income_statement')
                if income_stmt is not None and not income_stmt.empty:
                    # Get most recent and year-over-year data
                    financial_text = f"\n=== Income Statement for {ticker} ===\n"

                    for col in income_stmt.columns[:4]:  # Last 4 years
                        year = col.strftime('%Y') if hasattr(col, 'strftime') else str(col)
                        financial_text += f"\nYear: {year}\n"

                        # Key metrics from income statement
                        for row_name in income_stmt.index:
                            value = income_stmt.loc[row_name, col]
                            if pd.notna(value):
                                # Format large numbers
                                if abs(value) > 1e9:
                                    formatted = f"${value/1e9:.2f}B"
                                elif abs(value) > 1e6:
                                    formatted = f"${value/1e6:.2f}M"
                                else:
                                    formatted = f"${value:,.0f}"
                                financial_text += f"  {row_name}: {formatted}\n"

                    documents.append(
                        Document(
                            page_content=financial_text,
                            metadata={"ticker": ticker, "type": "income_statement"},
                        )
                    )

                # Balance Sheet
                balance_sheet = financials.get('balance_sheet')
                if balance_sheet is not None and not balance_sheet.empty:
                    balance_text = f"\n=== Balance Sheet for {ticker} ===\n"

                    for col in balance_sheet.columns[:4]:
                        year = col.strftime('%Y') if hasattr(col, 'strftime') else str(col)
                        balance_text += f"\nYear: {year}\n"

                        for row_name in balance_sheet.index:
                            value = balance_sheet.loc[row_name, col]
                            if pd.notna(value):
                                if abs(value) > 1e9:
                                    formatted = f"${value/1e9:.2f}B"
                                elif abs(value) > 1e6:
                                    formatted = f"${value/1e6:.2f}M"
                                else:
                                    formatted = f"${value:,.0f}"
                                balance_text += f"  {row_name}: {formatted}\n"

                    documents.append(
                        Document(
                            page_content=balance_text,
                            metadata={"ticker": ticker, "type": "balance_sheet"},
                        )
                    )

                # Cash Flow Statement
                cash_flow = financials.get('cash_flow')
                if cash_flow is not None and not cash_flow.empty:
                    cf_text = f"\n=== Cash Flow Statement for {ticker} ===\n"

                    for col in cash_flow.columns[:4]:
                        year = col.strftime('%Y') if hasattr(col, 'strftime') else str(col)
                        cf_text += f"\nYear: {year}\n"

                        for row_name in cash_flow.index:
                            value = cash_flow.loc[row_name, col]
                            if pd.notna(value):
                                if abs(value) > 1e9:
                                    formatted = f"${value/1e9:.2f}B"
                                elif abs(value) > 1e6:
                                    formatted = f"${value/1e6:.2f}M"
                                else:
                                    formatted = f"${value:,.0f}"
                                cf_text += f"  {row_name}: {formatted}\n"

                    documents.append(
                        Document(
                            page_content=cf_text,
                            metadata={"ticker": ticker, "type": "cash_flow"},
                        )
                    )

            # Create documents from news (최신 뉴스 10개, 한글 포맷팅)
            if news:
                news_summary = f"\n=== 최신 뉴스 ({ticker}) ===\n\n"
                for i, article in enumerate(news[:10], 1):
                    title = article.get('title', 'N/A')
                    publisher = article.get('publisher', 'N/A')
                    link = article.get('link', '')
                    summary = article.get('summary', 'N/A')

                    news_summary += f"""
뉴스 #{i}
제목: {title}
출처: {publisher}
링크: {link}
요약: {summary}
---
"""

                documents.append(
                    Document(
                        page_content=news_summary,
                        metadata={
                            "ticker": ticker,
                            "type": "news",
                        },
                    )
                )

            # Create document from analyst recommendations
            if recommendations is not None and not recommendations.empty:
                rec_text = f"\n=== 애널리스트 추천 ({ticker}) ===\n\n"

                # Get most recent recommendations (last 10)
                recent_recs = recommendations.tail(10)

                for idx, row in recent_recs.iterrows():
                    date = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                    firm = row.get('Firm', 'N/A')
                    to_grade = row.get('To Grade', 'N/A')
                    from_grade = row.get('From Grade', '')
                    action = row.get('Action', 'N/A')

                    rec_text += f"날짜: {date}\n"
                    rec_text += f"  증권사: {firm}\n"
                    rec_text += f"  등급: {from_grade} → {to_grade}\n" if from_grade else f"  등급: {to_grade}\n"
                    rec_text += f"  조치: {action}\n\n"

                documents.append(
                    Document(
                        page_content=rec_text,
                        metadata={"ticker": ticker, "type": "analyst_recommendations"},
                    )
                )

            # Create document from major holders
            if major_holders is not None and not major_holders.empty:
                holders_text = f"\n=== 주요 주주 ({ticker}) ===\n\n"

                for idx, row in major_holders.iterrows():
                    holders_text += f"{idx}: {row.iloc[0]}\n"

                documents.append(
                    Document(
                        page_content=holders_text,
                        metadata={"ticker": ticker, "type": "major_holders"},
                    )
                )

            # Create document from institutional holders
            if institutional_holders is not None and not institutional_holders.empty:
                inst_text = f"\n=== 기관 투자자 ({ticker}) ===\n\n"

                # Top 10 institutional holders
                for idx, row in institutional_holders.head(10).iterrows():
                    holder = row.get('Holder', 'N/A')
                    shares = row.get('Shares', 0)
                    date_reported = row.get('Date Reported', 'N/A')
                    pct_out = row.get('% Out', 0)
                    value = row.get('Value', 0)

                    inst_text += f"기관명: {holder}\n"
                    inst_text += f"  보유주식수: {shares:,}\n"
                    inst_text += f"  지분율: {pct_out:.2%}\n" if isinstance(pct_out, (int, float)) else f"  지분율: {pct_out}\n"
                    inst_text += f"  가치: ${value:,}\n" if isinstance(value, (int, float)) else f"  가치: {value}\n"
                    inst_text += f"  보고일: {date_reported}\n\n"

                documents.append(
                    Document(
                        page_content=inst_text,
                        metadata={"ticker": ticker, "type": "institutional_holders"},
                    )
                )

            # Create document from insider transactions
            if insider_transactions is not None and not insider_transactions.empty:
                insider_text = f"\n=== 내부자 거래 ({ticker}) ===\n\n"

                # Most recent 15 transactions
                for idx, row in insider_transactions.head(15).iterrows():
                    insider = row.get('Insider', 'N/A')
                    relation = row.get('Relation', 'N/A')
                    transaction = row.get('Transaction', 'N/A')
                    shares = row.get('Shares', 0)
                    value = row.get('Value', 0)
                    date = row.get('Start Date', 'N/A')

                    insider_text += f"날짜: {date}\n"
                    insider_text += f"  내부자: {insider} ({relation})\n"
                    insider_text += f"  거래유형: {transaction}\n"
                    insider_text += f"  주식수: {shares:,}\n" if isinstance(shares, (int, float)) else f"  주식수: {shares}\n"
                    insider_text += f"  거래금액: ${value:,}\n" if isinstance(value, (int, float)) else f"  거래금액: {value}\n"
                    insider_text += f"\n"

                documents.append(
                    Document(
                        page_content=insider_text,
                        metadata={"ticker": ticker, "type": "insider_transactions"},
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
            return {"response": f"오류: {state['error']}"}

        query = state["query"]
        context = state.get("context", "")
        ticker = state["ticker"]

        system_prompt = f"""당신은 주식 시장 분석을 전문으로 하는 금융 애널리스트 어시스턴트입니다.
{ticker}에 대한 최신 정보(기업 데이터, 재무제표, 애널리스트 추천, 주주 정보, 내부자 거래, 뉴스 등)에 접근할 수 있습니다.

당신의 임무는 제공된 컨텍스트를 기반으로 사용자의 질문에 답변하는 것입니다.

중요한 지침:
- **반드시 한국어로 답변하세요**
- 구체적이고 정확한 숫자와 데이터를 인용하세요
- 출처가 있다면 명시하세요
- **뉴스를 언급할 때는 반드시 원문 링크를 마크다운 형식으로 포함하세요** (예: [뉴스 제목](링크))
- 애널리스트 추천, 주요 주주, 내부자 거래 정보가 있다면 적극 활용하세요
- 경영진이 실행할 수 있는 인사이트를 제공하세요
- 컨텍스트에 충분한 정보가 없다면 명확히 밝히세요
- 표와 목록을 활용하여 가독성을 높이세요
"""

        user_prompt = f"""컨텍스트:
{context}

질문: {query}

위 컨텍스트를 기반으로 한국어로 포괄적인 답변을 제공해주세요."""

        try:
            response = self.llm_client.simple_query(
                query=user_prompt,
                system_prompt=system_prompt,
            )

            return {"response": response}
        except Exception as e:
            return {"response": f"응답 생성 오류: {str(e)}"}

    def evaluate_quality(self, state: FinancialQAState) -> Dict[str, Any]:
        """Node: Evaluate response quality at CEO reporting level (NOVA pattern)"""
        if state.get("error"):
            return {}

        response = state.get("response", "")
        query = state["query"]
        ticker = state["ticker"]
        context = state.get("context", "")

        # CEO-level quality evaluation
        evaluation_prompt = f"""You are evaluating a Korean-language financial analysis report for C-level executives.
Assess whether this response meets CEO reporting standards on a scale of 0-10.

Company: {ticker}
Question: {query}
Response (in Korean): {response}

Available Context:
{context[:500]}...

Evaluation Criteria for CEO-Level Reports:
1. **Accuracy** (0-2): Data correctness, no misleading claims, proper use of financial data
2. **Completeness** (0-2): Answers the question fully, covers key aspects
3. **Clarity** (0-2): Clear Korean language, well-structured, professional formatting
4. **Actionability** (0-2): Provides insights executives can act on
5. **Professionalism** (0-2): Appropriate tone for Korean business context, proper citations, executive-ready

IMPORTANT: The response MUST be in Korean. If it's in English, deduct 2 points from PROFESSIONALISM.

Provide detailed evaluation in this exact format:
SCORE: [total score 0-10]
ACCURACY: [score 0-2]
COMPLETENESS: [score 0-2]
CLARITY: [score 0-2]
ACTIONABILITY: [score 0-2]
PROFESSIONALISM: [score 0-2]
FEEDBACK: [specific improvement suggestions in Korean if score < {settings.quality_threshold}]
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

        system_prompt = f"""당신은 주식 시장 분석을 전문으로 하는 금융 애널리스트 어시스턴트입니다.
{ticker}에 대한 최신 정보(기업 데이터, 재무제표, 뉴스 등)에 접근할 수 있습니다.

당신의 임무는 제공된 컨텍스트를 기반으로 사용자의 질문에 답변하는 것입니다.
현재 시도 횟수: #{regenerate_count + 1}. 이전 답변이 평가되었고 개선이 필요합니다.

이전 시도의 피드백:
{feedback}

중요 지침:
- **반드시 한국어로 답변하세요** (이것은 필수입니다!)
- 위 피드백을 반영하여 CEO 보고 수준으로 답변을 개선하세요
- 데이터를 정확하게 사용하세요
- 모든 측면을 완전하게 다루세요
- 명확하고 전문적인 한국어를 사용하세요
- 경영진이 실행할 수 있는 인사이트를 제공하세요
- 표와 목록을 활용하여 가독성을 높이세요
"""

        user_prompt = f"""컨텍스트:
{context}

질문: {query}

위 컨텍스트와 피드백을 기반으로 개선된 CEO 수준의 한국어 답변을 제공해주세요."""

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
                "response": f"응답 생성 오류: {str(e)}",
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
            "recommendations": None,
            "major_holders": None,
            "institutional_holders": None,
            "insider_transactions": None,
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
