#!/usr/bin/env python3
"""
Tuzi Image Generator CLI - Generate images with Tu-zi.com API
A command-line interface with rich progress indicators and automatic model fallback
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from .core import (
    TuZiImageGenerator,
    TuZiSurvey,
    get_api_key,
    validate_parameters,
    QUALITY_OPTIONS,
    SIZE_OPTIONS,
    FORMAT_OPTIONS,
    BACKGROUND_OPTIONS,
    MODEL_FALLBACK_ORDER,
)

# Initialize Typer app and Rich console
app = typer.Typer(help="Generate images using Tu-zi.com API with automatic model fallback")
console = Console()


def check_api_key() -> str:
    """Check for API key and display error panel if missing"""
    try:
        return get_api_key()
    except ValueError as e:
        console.print(Panel.fit(
            f"[bold red]❌ Error: {e}[/bold red]\n"
            "Please set your Tu-zi.com API key:\n"
            "[dim]export TUZI_API_KEY='your_api_key_here'[/dim]",
            title="API Key Required",
            border_style="red"
        ))
        raise typer.Exit(1)



@app.command()
def image(
    prompt: str = typer.Argument(..., help="The prompt for image generation"),
    quality: str = typer.Option("auto", "--quality", "-q", help="Image quality: low, medium, high, auto"),
    size: str = typer.Option("auto", "--size", "-s", help="Image size: 1024x1024, 1536x1024, 1024x1536, auto"),
    format: str = typer.Option("png", "--format", "-f", help="Output format: png, jpeg, webp"),
    background: str = typer.Option("opaque", "--background", "-bg", help="Background: transparent, opaque"),
    compression: Optional[int] = typer.Option(None, "--compression", "-c", help="Output compression (0-100, for JPEG/WebP)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (can be filename or full path)"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming response"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full API response")
):
    """Generate an image using Tu-zi.com API with automatic model fallback"""
    
    # Validate parameters
    try:
        validate_parameters(quality, size, format, background, compression)
    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        raise typer.Exit(1)
    
    # Check for API key
    api_key = check_api_key()
    
    # Initialize generator with console for rich output
    generator = TuZiImageGenerator(api_key, console=console)
    
    # Build parameters
    params = {}
    if quality != "auto":
        params["quality"] = quality
    if size != "auto":
        params["size"] = size
    if format != "png":
        params["format"] = format
    if background == "transparent":
        params["background"] = background
    if compression is not None:
        params["output_compression"] = compression
    
    # Parse output path
    if output:
        output_path = Path(output)
        if output_path.is_absolute() or "/" in output or "\\" in output:
            # Full path provided
            output_dir = str(output_path.parent)
            base_name = output_path.stem
        else:
            # Just filename provided, use current directory
            output_dir = "."
            base_name = output
    else:
        # No output specified, use default
        output_dir = "."
        base_name = None
    
    # Display generation info
    info_lines = [
        f"[bold]Prompt:[/bold] {prompt}",
        f"[bold]Quality:[/bold] {quality}",
        f"[bold]Size:[/bold] {size}",
        f"[bold]Format:[/bold] {format}",
        f"[bold]Background:[/bold] {background}",
        f"[bold]Output:[/bold] {output or 'auto-generated filename'}",
        f"[bold]Model Fallback Order:[/bold] {' → '.join(MODEL_FALLBACK_ORDER)}"
    ]
    
    if compression is not None:
        info_lines.append(f"[bold]Compression:[/bold] {compression}%")
    
    console.print(Panel.fit(
        "\n".join(info_lines),
        title="🎨 Image Generation (Automatic Model Fallback)",
        border_style="cyan"
    ))
    
    try:
        # Generate the image with automatic model fallback
        result = generator.generate_image(
            prompt=prompt,
            stream=not no_stream,
            **params
        )
        
        # Extract and display the response
        content = generator.extract_response_content(result)
        
        # Extract image URLs
        image_urls = generator.extract_image_urls(content)
        
        if image_urls:
            console.print(f"[bold cyan]📸 Found {len(image_urls)} image(s) to download[/bold cyan]")
            
            # Download images
            downloaded_files = generator.download_images(
                image_urls, 
                output_dir=output_dir, 
                base_name=base_name
            )
            
            if downloaded_files:
                console.print(Panel(
                    "\n".join([f"✅ {file}" for file in downloaded_files]),
                    title="🖼️ Downloaded Images",
                    border_style="green"
                ))
                
                # Show which model was used
                if "model_used" in result:
                    console.print(f"[dim]Generated using model: {result['model_used']}[/dim]")
            else:
                console.print("[red]❌ No images were successfully downloaded[/red]")
        else:
            console.print("[yellow]⚠️ No images found in the response[/yellow]")
        
        # Display the response content in a panel only if verbose mode is enabled
        if verbose:
            console.print(Panel.fit(
                "",  # Empty content, we'll render below
                title="📝 Full Response",
                border_style="blue"
            ))
            console.print(Markdown(content))
        
    except Exception as e:
        console.print(Panel(
            f"[bold red]❌ Failed to generate image:[/bold red] {str(e)}",
            title="Error",
            border_style="red"
        ))
        raise typer.Exit(1)


@app.command()
def survey(
    prompt: str = typer.Argument(..., help="The query prompt for the survey"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming response"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed response information"),
    show_thinking: bool = typer.Option(False, "--show-thinking", help="Show the thinking process in addition to the final answer")
):
    """Conduct a survey/query using Tu-zi.com's o3-all model with web search capabilities"""
    
    # Check for API key
    api_key = check_api_key()
    
    
    # Initialize survey with console for rich output
    survey_obj = TuZiSurvey(api_key, console=console, show_thinking=show_thinking)
    
    # Display survey info
    info_lines = [
        f"[bold]Query:[/bold] {prompt}",
        f"[bold]Model:[/bold] o3-all (with web search)",
        f"[bold]Streaming:[/bold] {'No' if no_stream else 'Yes'}",
        f"[bold]Show Thinking:[/bold] {'Yes' if show_thinking else 'No (final answer only)'}"
    ]
    
    console.print(Panel.fit(
        "\n".join(info_lines),
        title="🔍 Survey Query (o3-all with Web Search)",
        border_style="cyan"
    ))
    
    try:
        # Conduct the survey
        result = survey_obj.survey(
            prompt=prompt,
            stream=not no_stream
        )
        
        # Extract and display the response
        content = survey_obj.extract_survey_content(result)
        
        if verbose:
            # Show usage information if available
            if "result" in result and "usage" in result["result"]:
                usage = result["result"]["usage"]
                console.print(Panel(
                    f"[dim]Tokens used: {usage.get('total_tokens', 'N/A')} "
                    f"(prompt: {usage.get('prompt_tokens', 'N/A')}, "
                    f"completion: {usage.get('completion_tokens', 'N/A')})[/dim]",
                    title="📊 Usage Information",
                    border_style="blue"
                ))
        
        # Display the content in a markdown panel if not already displayed during streaming
        if no_stream and content:
            console.print(Panel.fit(
                "",  # Empty content, we'll render below
                title="📝 Survey Response",
                border_style="green"
            ))
            console.print(Markdown(content))
        
    except Exception as e:
        console.print(Panel(
            f"[bold red]❌ Failed to conduct survey:[/bold red] {str(e)}",
            title="Error",
            border_style="red"
        ))
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI application"""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 