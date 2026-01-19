"""COSMO - Yahoo Finance AI Assistant CLI (following NOVA pattern)"""
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from typing import Optional

from src.agents.graph import create_qa_graph
from config.config import settings

app = typer.Typer(
    name="cosmo",
    help="COSMO - AI-powered Yahoo Finance assistant for stock market analysis",
    add_completion=False,
)
console = Console()


@app.command()
def query(
    ticker: str = typer.Argument(..., help="Stock ticker symbol (e.g., AAPL, TSLA)"),
    question: Optional[str] = typer.Option(
        None,
        "--question", "-q",
        help="Question to ask about the ticker"
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive", "-i/-n",
        help="Enable interactive mode for multiple questions"
    ),
):
    """
    Query financial information about a stock ticker using AI

    Examples:
        cosmo query AAPL -q "What's the current stock price?"
        cosmo query TSLA --interactive
    """
    console.print(Panel.fit(
        f"[bold cyan]COSMO Financial Assistant[/bold cyan]\n"
        f"Ticker: [yellow]{ticker.upper()}[/yellow]",
        border_style="cyan"
    ))

    # Initialize the graph
    console.print("\n[dim]Initializing AI assistant...[/dim]")
    qa_graph = create_qa_graph()

    # Fetch and index initial data
    console.print(f"[dim]Fetching data for {ticker.upper()}...[/dim]")

    if question and not interactive:
        # Single question mode
        _process_question(qa_graph, ticker, question)
    else:
        # Interactive mode
        console.print("\n[bold green]Interactive mode enabled.[/bold green]")
        console.print("[dim]Type 'exit' or 'quit' to end the session.[/dim]\n")

        # Process initial question if provided
        if question:
            _process_question(qa_graph, ticker, question)

        # Interactive loop
        while True:
            try:
                user_question = typer.prompt("\n💬 Your question")

                if user_question.lower() in ["exit", "quit", "q"]:
                    console.print("\n[yellow]Goodbye! 👋[/yellow]")
                    break

                _process_question(qa_graph, ticker, user_question)

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Session interrupted. Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")


def _process_question(qa_graph, ticker: str, question: str):
    """Process a single question"""
    console.print(f"\n[dim]Processing question...[/dim]")

    try:
        # Run the workflow
        result = qa_graph.run(ticker=ticker.upper(), query=question)

        # Display response
        response = result.get("response", "No response generated")
        quality_score = result.get("quality_score", 0.0)

        console.print("\n" + "="*80)
        console.print(Panel(
            Markdown(response),
            title="[bold green]Response[/bold green]",
            border_style="green",
        ))

        # Display quality info if available
        if quality_score:
            quality_color = "green" if quality_score >= settings.quality_threshold else "yellow"
            console.print(f"\n[dim]Quality Score: [{quality_color}]{quality_score:.1f}/10[/{quality_color}][/dim]")

    except Exception as e:
        console.print(f"\n[red]Error processing question: {str(e)}[/red]")


@app.command()
def info():
    """Display COSMO system information"""
    table = Table(title="COSMO Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Chat Model", settings.chat_model)
    table.add_row("Embedding Model", settings.embedding_model)
    table.add_row("Embedding Provider", settings.embedding_provider)
    table.add_row("Vector Store", settings.vector_store_type)
    table.add_row("Quality Threshold", f"{settings.quality_threshold}/10")
    table.add_row("Chunk Size", str(settings.chunk_size))
    table.add_row("Retrieval Top-K", str(settings.retrieval_top_k))

    console.print(table)


@app.command()
def clear_cache(
    confirm: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Skip confirmation prompt"
    )
):
    """Clear the vector store cache"""
    if not confirm:
        confirm = typer.confirm("Are you sure you want to clear the vector store cache?")

    if confirm:
        try:
            from src.vectorstore.chroma_store import ChromaStore
            store = ChromaStore()
            store.delete_collection()
            console.print("[green]✓[/green] Cache cleared successfully")
        except Exception as e:
            console.print(f"[red]Error clearing cache: {str(e)}[/red]")
    else:
        console.print("[yellow]Cache clear cancelled[/yellow]")


def main():
    """Main entry point"""
    try:
        # Configuration is auto-validated by Pydantic on initialization
        app()
    except ValueError as e:
        console.print(f"[red]Configuration Error: {str(e)}[/red]")
        console.print("\n[yellow]Please check your .env file and ensure all required API keys are set.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected Error: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    main()
