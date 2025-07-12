"""
Tuzi Image Generator MCP Server

This module provides a Model Context Protocol (MCP) server interface
for the Tuzi image generation service using FastMCP with automatic model fallback.
Supports both stdio and HTTP transport protocols.
"""

import argparse
import os
import sys
import time
from typing import Optional, Annotated, Literal

import typer

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .core import (
    TuZiImageGenerator,
    validate_parameters,
    get_api_key,
)

# Create the MCP server
mcp = FastMCP("Tuzi Image Generator")

# Pydantic model for structured response
class ImageGenerationResult(BaseModel):
    """Result of image generation"""
    success: bool = Field(description="Whether the generation was successful")
    message: str = Field(description="Status message")
    image_url: str = Field(description="URL of the generated image")
    downloaded_file: str = Field(description="Path to the downloaded image file")
    model_used: str = Field(description="Model used for generation")
    generation_time: float = Field(description="Time taken for generation in seconds")


@mcp.tool()
def generate_image(
    prompt: Annotated[str, Field(description="The text prompt for image generation")],
    quality: Annotated[
        Literal["auto", "low", "medium", "high"], 
        Field(description="Image quality setting")
    ] = "auto",
    size: Annotated[
        Literal["auto", "1024x1024", "1536x1024", "1024x1536"], 
        Field(description="Image dimensions")
    ] = "auto",
    format: Annotated[
        Literal["png", "jpeg", "webp"], 
        Field(description="Output image format")
    ] = "png",
    background: Annotated[
        Literal["opaque", "transparent"], 
        Field(description="Background type for the image")
    ] = "opaque",
    compression: Annotated[
        Optional[int], 
        Field(description="Output compression 0-100 for JPEG/WebP formats", ge=0, le=100)
    ] = None,
    output_path: Annotated[
        str, 
        Field(description="Full path where to save the generated image")
    ] = "images/generated_image.png"
) -> ImageGenerationResult:
    """Generate an image from a prompt using Text-To-Image model"""
    start_time = time.time()
    
    try:
        # Validate parameters
        validate_parameters(quality, size, format, background, compression)
        
        # Get API key
        api_key = get_api_key()
        
        # Initialize generator (without console output for MCP)
        generator = TuZiImageGenerator(api_key, console=None)
        
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
        
        # Generate the image (uses automatic model fallback)
        result = generator.generate_image(
            prompt=prompt,
            stream=True,
            **params
        )
        
        # Extract response content
        content = generator.extract_response_content(result)
        
        # Extract image URLs
        image_urls = generator.extract_image_urls(content)
        
        # Use only the first image URL to simplify
        if not image_urls:
            raise Exception("No images were generated")
        
        first_image_url = image_urls[0]
        
        # Parse output path
        output_dir = os.path.dirname(output_path) or "."
        base_name = os.path.splitext(os.path.basename(output_path))[0] or "generated_image"
        
        # Download only the first image
        downloaded_files = generator.download_images(
            [first_image_url], 
            output_dir=output_dir, 
            base_name=base_name
        )
        
        downloaded_file = downloaded_files[0] if downloaded_files else ""
        
        generation_time = time.time() - start_time
        model_used = result.get("model_used", "unknown")
        
        return ImageGenerationResult(
            success=True,
            message=f"Image generated successfully using model: {model_used}",
            image_url=first_image_url,
            downloaded_file=downloaded_file,
            model_used=model_used,
            generation_time=generation_time
        )
        
    except Exception as e:
        generation_time = time.time() - start_time
        return ImageGenerationResult(
            success=False,
            message=f"Failed to generate image: {str(e)}",
            image_url="",
            downloaded_file="",
            model_used="none",
            generation_time=generation_time
        )


def run_server(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport protocol to use (default: stdio)", show_default=True, case_sensitive=False),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to when using HTTP transport (default: 127.0.0.1)", show_default=True),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to when using HTTP transport (default: 8000)", show_default=True),
    path: str = typer.Option("/mcp", "--path", help="Path to bind to when using HTTP transport (default: /mcp)", show_default=True),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Log level (default: INFO)", show_default=True, case_sensitive=False),
):
    """Start the Tuzi Image Generator MCP Server (supports stdio and HTTP transport)."""
    import logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    # Print server information
    print(f"Starting Tuzi Image Generator MCP Server", file=sys.stderr)
    print(f"Transport: {transport}", file=sys.stderr)

    # Run the server with the specified transport
    try:
        if transport == "stdio":
            mcp.run(transport="stdio")
        elif transport == "streamable-http":
            mcp.run(
                transport="streamable-http",
                host=host,
                port=port,
                path=path,
                log_level=log_level.upper(),
            )
    except KeyboardInterrupt:
        print("\nServer stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"Error running server: {e}", file=sys.stderr)
        sys.exit(1)


# Remove epilog and simplify Typer app initialization
app = typer.Typer(
    help="Tuzi Image Generator MCP Server (supports stdio and HTTP transport)",
    add_completion=False,
)

app.command()(run_server)

def main():
    app()

if __name__ == "__main__":
    main() 