# COSMO 🌟

**C**onversational **O**perational **S**tock **M**arket **O**racle

AI-powered Yahoo Finance assistant for intelligent stock market analysis using GLM-4.7 and RAG (Retrieval Augmented Generation).

## Overview

COSMO is an intelligent financial assistant that combines Yahoo Finance data with advanced LLM capabilities to answer questions about stocks, companies, and market trends. Built following the NOVA architecture pattern, it uses LangGraph for workflow orchestration and ChromaDB for semantic search.

## Features

- 📊 **Real-time Stock Data**: Fetch comprehensive ticker information from Yahoo Finance
- 🤖 **AI-Powered Analysis**: Leverages GLM-4.7 for intelligent responses
- 🔍 **Semantic Search**: ChromaDB-backed vector store for relevant context retrieval
- 📰 **News Integration**: Access to latest financial news and market updates
- 💬 **Interactive CLI**: User-friendly command-line interface with rich formatting
- ⚡ **Quality Gates**: Automatic evaluation of response quality (NOVA pattern)
- 🔄 **RAG Pipeline**: Retrieval Augmented Generation for accurate, context-aware answers

## Architecture

```
COSMO/
├── src/
│   ├── agents/          # LangGraph workflow orchestration
│   │   ├── graph.py     # Main Q&A workflow
│   │   └── state.py     # State definitions
│   ├── llm/             # LLM client integrations
│   │   └── glm_client.py # GLM-4.7 client
│   ├── vectorstore/     # Vector database
│   │   └── chroma_store.py # ChromaDB integration
│   ├── yahoo_finance.py # Yahoo Finance API wrapper
│   └── main.py          # CLI entry point
├── config/
│   └── config.py        # Pydantic settings
├── data/                # Data storage (auto-created)
│   ├── raw/
│   ├── processed/
│   └── vectordb/        # ChromaDB storage
└── .env                 # API keys (not in git)
```

## Installation

### Prerequisites

- Python 3.9+
- API Keys:
  - GLM API Key (from Zhipu AI)
  - OpenAI API Key (for embeddings)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/BK-Korea/COSMO.git
cd COSMO
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
# Or install in development mode:
pip install -e .
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

Create a `.env` file with the following:

```env
# GLM API Configuration
GLM_API_KEY=your_glm_api_key_here
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/

# OpenAI API Configuration (for embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
CHAT_MODEL=glm-4-flash
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_PROVIDER=openai
```

## Usage

### Interactive Mode (NOVA-style) - Default

Simply run `cosmo` to enter interactive mode:

```bash
cosmo
```

Or if not installed globally:
```bash
python -m src.main
```

This will display the COSMO banner and guide you through:

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ██████╗ ███████╗███╗   ███╗ ██████╗                ║
║  ██╔════╝██╔═══██╗██╔════╝████╗ ████║██╔═══██╗               ║
║  ██║     ██║   ██║███████╗██╔████╔██║██║   ██║               ║
║  ██║     ██║   ██║╚════██║██║╚██╔╝██║██║   ██║               ║
║  ╚██████╗╚██████╔╝███████║██║ ╚═╝ ██║╚██████╔╝               ║
║   ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝                ║
║                                                               ║
║        Conversational Operational Stock Market Oracle        ║
║           AI-Powered Yahoo Finance Assistant                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
         Powered by GLM-4.7 & Yahoo Finance API

┌─ Step 1: Stock Selection ─┐
🏢 Enter stock ticker symbol (e.g., AAPL, TSLA, NVDA): AAPL

✓ Selected: AAPL

┌─ Step 2: Ask Questions ─┐
Type your questions about the stock. Enter 'exit' or 'quit' to end.

💬 Question #1: What's the current stock price?
💬 Question #2: How has the stock performed this month?
💬 Question #3: exit
```

### View System Information

```bash
cosmo info
```

### Clear Vector Store Cache

```bash
cosmo clear-cache --yes
```

### Version Information

```bash
cosmo --version
```

## Examples

### Example 1: Full Interactive Session
```bash
$ cosmo
[COSMO Banner appears]

🏢 Enter stock ticker symbol: AAPL
✓ Selected: AAPL

💬 Question #1: What's the current price and P/E ratio?
[AI Response with stock data]

💬 Question #2: Any recent news about Apple?
[AI Response with news summary]

💬 Question #3: exit
Goodbye! 👋
```

### Example 2: Different Stocks
```bash
$ cosmo
🏢 Enter stock ticker symbol: NVDA
✓ Selected: NVDA

💬 Question #1: Analyze NVIDIA's recent performance
[AI Response]
```

### Example 3: Quick System Check
```bash
$ cosmo info
              COSMO Configuration
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Setting            ┃ Value                  ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Chat Model         │ glm-4-flash            │
│ Embedding Model    │ text-embedding-3-small │
│ Embedding Provider │ openai                 │
│ Vector Store       │ chroma                 │
│ Quality Threshold  │ 8.0/10                 │
│ Chunk Size         │ 1000                   │
│ Retrieval Top-K    │ 10                     │
└────────────────────┴────────────────────────┘
```

## Technical Details

### LLM Models

- **Chat Model**: GLM-4-Flash (Zhipu AI)
  - Fast inference for conversational responses
  - Cost-effective for production use

- **Embedding Model**: OpenAI text-embedding-3-small
  - Used for semantic search in vector store
  - High quality embeddings for financial context

### Workflow (LangGraph)

1. **Fetch Data**: Retrieve ticker info, historical data, and news from Yahoo Finance
2. **Index Data**: Process and store in ChromaDB vector database
3. **Retrieve Context**: Semantic search for relevant information
4. **Generate Response**: GLM-4.7 generates answer based on context
5. **Evaluate Quality**: Score response quality (0-10 scale)

### Quality Threshold

Responses are evaluated on a 0-10 scale. Default threshold is 8.0 (configurable in `.env`).

## Configuration

All configuration is managed through Pydantic Settings in `config/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `chat_model` | glm-4-flash | LLM for chat responses |
| `embedding_model` | text-embedding-3-small | Embedding model |
| `chunk_size` | 1000 | Token count per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `retrieval_top_k` | 10 | Documents to retrieve |
| `quality_threshold` | 8.0 | Minimum quality score |

## Development

### Project Structure

Following the NOVA pattern:
- **Modular Design**: Clear separation of concerns
- **Pydantic Settings**: Type-safe configuration
- **LangGraph Workflows**: Orchestrated agent pipelines
- **Quality Gates**: Automatic response evaluation

### Adding New Features

1. Define state in `src/agents/state.py`
2. Add nodes to workflow in `src/agents/graph.py`
3. Update CLI commands in `src/main.py`

## Limitations

- Requires active internet connection for Yahoo Finance API
- API rate limits apply for GLM and OpenAI
- Historical data limited by Yahoo Finance availability
- Response quality depends on data availability and LLM capabilities

## Roadmap

- [ ] Support for multiple LLM providers
- [ ] Enhanced financial metrics analysis
- [ ] Export results to various formats (PDF, Excel)
- [ ] Web interface (Streamlit/Gradio)
- [ ] Portfolio tracking and analysis
- [ ] Custom indicators and alerts

## License

MIT License

## Acknowledgments

- Built following the [NOVA](https://github.com/BK-Korea/NOVA) architecture pattern
- Powered by [Yahoo Finance API](https://github.com/ranaroussi/yfinance)
- Uses [Zhipu AI's GLM](https://open.bigmodel.cn/) for LLM capabilities
- Leverages [LangChain](https://www.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

Made with ❤️ by BK-Korea
