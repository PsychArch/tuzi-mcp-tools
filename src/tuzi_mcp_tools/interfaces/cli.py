#!/usr/bin/env python3
"""
Tuzi Image Generator CLI - Generate images with Tu-zi.com API
A command-line interface with rich progress indicators and automatic model fallback
"""

import os
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from ..infrastructure.container import get_cli_container

# Initialize Typer app and Rich console
app = typer.Typer(
    help="Generate images using Tu-zi.com API with automatic model fallback"
)
console = Console()

# Get the service container for CLI mode
container = get_cli_container()

# Get application services
image_service = container.get_image_service()
survey_service = container.get_survey_service()


def check_api_key() -> str:
    """Check for API key and display error panel if missing"""
    api_key = os.getenv("TUZI_API_KEY")
    if not api_key:
        console.print(
            Panel.fit(
                "[bold red]❌ Error: TUZI_API_KEY environment variable not set[/bold red]\n"
                "Please set your Tu-zi.com API key:\n"
                "[dim]export TUZI_API_KEY='your_api_key_here'[/dim]",
                title="API Key Required",
                border_style="red",
            )
        )
        raise typer.Exit(1)
    return api_key


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text prompt for image generation"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (default: auto-generated)"
    ),
    quality: str = typer.Option(
        "auto", "--quality", "-q", help="Image quality: auto, low, medium, high"
    ),
    size: str = typer.Option(
        "auto", "--size", "-s", help="Image size: auto, 1024x1024, 1536x1024, 1024x1536"
    ),
    format: str = typer.Option(
        "png", "--format", "-f", help="Output format: png, jpeg, webp"
    ),
    background: str = typer.Option(
        "opaque", "--background", "-b", help="Background type: opaque, transparent"
    ),
    compression: Optional[int] = typer.Option(
        None, "--compression", "-c", help="Compression level (0-100) for JPEG/WebP"
    ),
    conversation_id: Optional[str] = typer.Option(
        None, "--conversation-id", help="Continue existing conversation"
    ),
    input_image: Optional[str] = typer.Option(
        None, "--input-image", help="Path to reference image"
    ),
):
    """Generate images using GPT model with automatic fallback"""
    check_api_key()

    try:
        # Read input image if provided
        input_image_data = None
        if input_image:
            import base64

            with open(input_image, "rb") as f:
                input_image_data = base64.b64encode(f.read()).decode()

        # Generate output path if not provided
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"generated_image_{timestamp}.{format}"

        # Use image service to generate
        result = image_service.generate_gpt_image(
            prompt=prompt,
            output_path=output,
            quality=quality,
            size=size,
            format=format,
            background=background,
            output_compression=compression,
            input_image=input_image_data,
            conversation_id=conversation_id,
        )

        if result.success:
            console.print(
                f"[green]✓[/green] Image generated successfully: {result.output_path}"
            )
        else:
            console.print(f"[red]✗[/red] Failed to generate image: {result.error}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def flux(
    prompt: str = typer.Argument(..., help="Text prompt for FLUX image generation"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (default: auto-generated)"
    ),
    aspect_ratio: str = typer.Option(
        "1:1",
        "--aspect-ratio",
        "-a",
        help="Aspect ratio: 1:1, 16:9, 9:16, 4:3, 3:4, 21:9, 9:21",
    ),
    format: str = typer.Option(
        "png", "--format", "-f", help="Output format: png, jpg, jpeg, webp"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Seed for reproducible generation"
    ),
    conversation_id: Optional[str] = typer.Option(
        None, "--conversation-id", help="Continue existing conversation"
    ),
    input_image: Optional[str] = typer.Option(
        None, "--input-image", help="Path to reference image"
    ),
):
    """Generate images using FLUX model"""
    check_api_key()

    try:
        # Read input image if provided
        input_image_data = None
        if input_image:
            import base64

            with open(input_image, "rb") as f:
                input_image_data = base64.b64encode(f.read()).decode()

        # Generate output path if not provided
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"flux_image_{timestamp}.{format}"

        # Use image service to generate
        result = image_service.generate_flux_image(
            prompt=prompt,
            output_path=output,
            aspect_ratio=aspect_ratio,
            output_format=format,
            seed=seed,
            input_image=input_image_data,
            conversation_id=conversation_id,
        )

        if result.success:
            console.print(
                f"[green]✓[/green] FLUX image generated successfully: {result.output_path}"
            )
        else:
            console.print(f"[red]✗[/red] Failed to generate FLUX image: {result.error}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def survey(
    query: str = typer.Argument(..., help="Survey question or topic to research"),
    conversation_id: Optional[str] = typer.Option(
        None, "--conversation-id", help="Continue existing conversation"
    ),
    enable_web_search: bool = typer.Option(
        True, "--web-search/--no-web-search", help="Enable web search capabilities"
    ),
):
    """Conduct a survey using o3-all model with web search"""
    check_api_key()

    try:
        # Use survey service
        result = survey_service.conduct_survey(
            query=query,
            conversation_id=conversation_id,
            enable_web_search=enable_web_search,
        )

        if result.success:
            console.print(
                Panel(
                    Markdown(result.content),
                    title=f"Survey Results: {query[:50]}...",
                    border_style="green",
                )
            )
        else:
            console.print(f"[red]✗[/red] Survey failed: {result.error}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def conversations():
    """List available conversations"""
    try:
        conversation_service = container.get_conversation_service()
        conversations = conversation_service.list_conversations()

        if not conversations:
            console.print("[dim]No conversations found[/dim]")
            return

        console.print("[bold]Available Conversations:[/bold]")
        for conv in conversations:
            message_count = len(conv.messages)
            last_message = conv.messages[-1] if conv.messages else None
            last_time = (
                last_message.timestamp.strftime("%Y-%m-%d %H:%M")
                if last_message
                else "Unknown"
            )

            console.print(
                f"  [cyan]{conv.id}[/cyan] ({conv.type.value}) - {message_count} messages - Last: {last_time}"
            )

    except Exception as e:
        console.print(f"[red]✗[/red] Error listing conversations: {str(e)}")
        raise typer.Exit(1)


@app.command()
def clear_conversations():
    """Clear all conversation history"""
    try:
        conversation_service = container.get_conversation_service()
        conversation_service.clear_all_conversations()
        console.print("[green]✓[/green] All conversations cleared")

    except Exception as e:
        console.print(f"[red]✗[/red] Error clearing conversations: {str(e)}")
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI application"""
    app()


if __name__ == "__main__":
    main()
