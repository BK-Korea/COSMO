"""COSMO - Yahoo Finance AI Assistant CLI (NOVA-style)"""
import sys
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

COSMO_BANNER = """
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
"""


def show_banner():
    """Display COSMO banner - 우주 테마 (군청색)"""
    console.print(COSMO_BANNER, style="bold blue")
    console.print(
        "[dim bright_blue]Powered by GLM-4.7 & Yahoo Finance API[/dim bright_blue]\n",
        justify="center"
    )


def run_interactive():
    """Run interactive mode (NOVA-style)"""
    show_banner()

    try:
        # Step 1: Get ticker symbol or company name
        console.print(Panel.fit(
            "[bold bright_blue]Step 1: Stock Selection[/bold bright_blue]",
            border_style="blue"
        ))

        ticker = typer.prompt(
            "\n🌌 Enter ticker or company name (e.g., AAPL, Apple, Tesla, Archer Aviation)"
        ).strip()

        if not ticker:
            console.print("[red]Invalid input. Exiting.[/red]")
            raise typer.Exit(1)

        console.print(f"\n✓ Input: [bold bright_blue]{ticker}[/bold bright_blue]")
        console.print("[dim bright_blue]AI will resolve this to the correct ticker symbol...[/dim bright_blue]")

        # Initialize the graph
        console.print("\n[dim bright_blue]Initializing AI assistant with RAG pipeline...[/dim bright_blue]")
        qa_graph = create_qa_graph()

        # Step 2: Q&A Loop
        console.print(Panel.fit(
            "[bold bright_blue]Step 2: Ask Questions[/bold bright_blue]\n"
            "[dim bright_blue]Type your questions about the stock. Enter 'exit' or 'quit' to end.[/dim bright_blue]",
            border_style="blue"
        ))

        question_count = 0

        while True:
            try:
                question_count += 1
                user_question = typer.prompt(f"\n⭐ Question #{question_count}")

                if user_question.lower() in ["exit", "quit", "q"]:
                    console.print("\n[bright_blue]Goodbye! 🌙[/bright_blue]")
                    break

                _process_question(qa_graph, ticker, user_question, question_count)

            except KeyboardInterrupt:
                console.print("\n\n[bright_blue]Session interrupted. Goodbye! 🌙[/bright_blue]")
                break
            except EOFError:
                console.print("\n\n[bright_blue]Session ended. Goodbye! 🌙[/bright_blue]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")

    except KeyboardInterrupt:
        console.print("\n\n[bright_blue]Session cancelled. Goodbye! 🌙[/bright_blue]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


def _process_question(qa_graph, ticker: str, question: str, question_count: int):
    """Process a single question with CEO-level quality evaluation"""
    # Don't uppercase - let resolver handle it
    is_cached = qa_graph.is_ticker_cached(ticker.upper())

    if question_count == 1 and not is_cached:
        console.print(f"\n[dim bright_blue]Resolving ticker and fetching data... (first query may take 10-20s)[/dim bright_blue]")
    else:
        console.print(f"\n[dim bright_blue]Processing with quality evaluation... (may take 5-15s with regeneration)[/dim bright_blue]")

    try:
        # Run the workflow with CEO-level quality evaluation enabled
        result = qa_graph.run(ticker=ticker, query=question, enable_evaluation=True)

        # Get response and quality info
        response = result.get("response", "No response generated")
        quality_score = result.get("quality_score")
        quality_feedback = result.get("quality_feedback", "")
        regenerate_count = result.get("regenerate_count", 0)
        resolved_ticker = result.get("ticker", ticker)

        # Display response
        console.print("\n" + "="*80)
        console.print(Panel(
            Markdown(response),
            title=f"[bold bright_blue]🌟 Response - {resolved_ticker}[/bold bright_blue]",
            border_style="blue",
        ))

        # Display quality score and metrics
        if quality_score is not None:
            quality_color = "bright_blue" if quality_score >= settings.quality_threshold else "blue"
            console.print(f"\n[bold bright_blue]Quality Score:[/bold bright_blue] [{quality_color}]{quality_score:.1f}/10[/{quality_color}]", end="")

            if regenerate_count > 0:
                console.print(f" [dim bright_blue](Regenerated {regenerate_count} time{'s' if regenerate_count > 1 else ''})[/dim bright_blue]")
            else:
                console.print()

            # Show detailed breakdown if available
            if "ACCURACY:" in quality_feedback:
                console.print("\n[dim bright_blue]Quality Breakdown:[/dim bright_blue]")
                for line in quality_feedback.split("\n"):
                    if any(metric in line for metric in ["ACCURACY:", "COMPLETENESS:", "CLARITY:", "ACTIONABILITY:", "PROFESSIONALISM:"]):
                        console.print(f"  [dim bright_blue]{line.strip()}[/dim bright_blue]")

    except Exception as e:
        console.print(f"\n[red]Error processing question: {str(e)}[/red]")


@app.command()
def info():
    """Display COSMO system information"""
    table = Table(title="🌌 COSMO Configuration", show_header=True, header_style="bold bright_blue")
    table.add_column("Setting", style="bright_blue")
    table.add_column("Value", style="blue")

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
            console.print("[bright_blue]✓[/bright_blue] Cache cleared successfully")
        except Exception as e:
            console.print(f"[red]Error clearing cache: {str(e)}[/red]")
    else:
        console.print("[bright_blue]Cache clear cancelled[/bright_blue]")


def main():
    """Main entry point"""
    try:
        # Check if no arguments provided (just running 'cosmo' or 'python -m src.main')
        # sys.argv[0] is the script name
        if len(sys.argv) == 1:
            # No arguments, run interactive mode
            run_interactive()
        elif len(sys.argv) == 2 and sys.argv[1] in ['--version', '-v']:
            # Handle version flag
            console.print("[bright_blue]🌌 COSMO v0.1.0[/bright_blue]")
        else:
            # Run Typer app for subcommands
            app()
    except ValueError as e:
        console.print(f"[red]Configuration Error: {str(e)}[/red]")
        console.print("\n[bright_blue]Please check your .env file and ensure all required API keys are set.[/bright_blue]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected Error: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    main()
