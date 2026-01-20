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
    """Display COSMO banner"""
    console.print(COSMO_BANNER, style="bold cyan")
    console.print(
        "[dim]Powered by GLM-4.7 & Yahoo Finance API[/dim]\n",
        justify="center"
    )


def run_interactive():
    """Run interactive mode (NOVA-style)"""
    show_banner()

    try:
        # Step 1: Get ticker symbol
        console.print(Panel.fit(
            "[bold yellow]Step 1: Stock Selection[/bold yellow]",
            border_style="yellow"
        ))

        ticker = typer.prompt(
            "\n🏢 Enter stock ticker symbol (e.g., AAPL, TSLA, NVDA)"
        ).strip().upper()

        if not ticker:
            console.print("[red]Invalid ticker. Exiting.[/red]")
            raise typer.Exit(1)

        console.print(f"\n✓ Selected: [bold green]{ticker}[/bold green]")

        # Initialize the graph
        console.print("\n[dim]Initializing AI assistant...[/dim]")
        qa_graph = create_qa_graph()

        # Fetch initial data
        console.print(f"[dim]Fetching data for {ticker}...[/dim]")

        # Step 2: Q&A Loop
        console.print(Panel.fit(
            "[bold yellow]Step 2: Ask Questions[/bold yellow]\n"
            "[dim]Type your questions about the stock. Enter 'exit' or 'quit' to end.[/dim]",
            border_style="yellow"
        ))

        question_count = 0

        while True:
            try:
                user_question = typer.prompt(f"\n💬 Question #{question_count + 1}")

                if user_question.lower() in ["exit", "quit", "q"]:
                    console.print("\n[yellow]Goodbye! 👋[/yellow]")
                    break

                question_count += 1
                _process_question(qa_graph, ticker, user_question)

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Session interrupted. Goodbye! 👋[/yellow]")
                break
            except EOFError:
                console.print("\n\n[yellow]Session ended. Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Session cancelled. Goodbye! 👋[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


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
        # Check if no arguments provided (just running 'cosmo' or 'python -m src.main')
        # sys.argv[0] is the script name
        if len(sys.argv) == 1:
            # No arguments, run interactive mode
            run_interactive()
        elif len(sys.argv) == 2 and sys.argv[1] in ['--version', '-v']:
            # Handle version flag
            console.print("[cyan]COSMO v0.1.0[/cyan]")
        else:
            # Run Typer app for subcommands
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
