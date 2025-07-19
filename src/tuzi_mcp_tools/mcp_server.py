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
    TuZiSurvey,
    validate_parameters,
    get_api_key,
    FLUX_ASPECT_RATIOS,
    FLUX_OUTPUT_FORMATS,
)

# Create the MCP server
mcp = FastMCP("Tuzi Tools - Image Generator and Survey")

# Pydantic model for structured response
class ImageGenerationResult(BaseModel):
    """Result of image generation"""
    success: bool = Field(description="Whether the generation was successful")
    message: str = Field(description="Status message")
    image_url: str = Field(description="URL of the generated image")
    downloaded_file: str = Field(description="Path to the downloaded image file")
    model_used: str = Field(description="Model used for generation")
    generation_time: float = Field(description="Time taken for generation in seconds")


class SurveyResult(BaseModel):
    """Result of survey/query"""
    success: bool = Field(description="Whether the survey was successful")
    message: str = Field(description="Status message")
    content: str = Field(description="The survey response content")
    response_time: float = Field(description="Time taken for response in seconds")


class FluxImageGenerationResult(BaseModel):
    """Result of FLUX image generation"""
    success: bool = Field(description="Whether the generation was successful")
    message: str = Field(description="Status message")
    image_url: str = Field(description="URL of the generated FLUX image")
    downloaded_file: str = Field(description="Path to the downloaded image file")
    generation_time: float = Field(description="Time taken for generation in seconds")
    seed_used: Optional[int] = Field(description="Seed used for generation (if provided)")
    aspect_ratio: str = Field(description="Aspect ratio used for generation")
    output_format: str = Field(description="Output format used for generation")


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


@mcp.tool()
def survey(
    prompt: Annotated[str, Field(description="The natural language query/question for the survey")],
    show_thinking: Annotated[bool, Field(description="Whether to include the thinking process in the response (default: False)")] = False,
    deep: Annotated[bool, Field(description="Deep analysis for complicated mathemetical or logical questions (default: False)")] = False
) -> SurveyResult:
    """Survey/query a topic using LLM with web search capabilities"""
    start_time = time.time()
    
    try:
        # Get API key
        api_key = get_api_key()
        
        # Initialize survey (without console output for MCP)
        survey_obj = TuZiSurvey(api_key, console=None, show_thinking=show_thinking)
        
        # Conduct the survey
        result = survey_obj.survey(
            prompt=prompt,
            stream=True,
            deep=deep
        )
        
        # Extract response content
        content = survey_obj.extract_survey_content(result)
        
        response_time = time.time() - start_time
        
        return SurveyResult(
            success=True,
            message="Survey completed successfully",
            content=content,
            response_time=response_time
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        return SurveyResult(
            success=False,
            message=f"Failed to conduct survey: {str(e)}",
            content="",
            response_time=response_time
        )


@mcp.tool()
def generate_flux_image(
    prompt: Annotated[str, Field(description="The text prompt for FLUX image generation")],
    aspect_ratio: Annotated[
        str, 
        Field(description="Image aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4, 21:9, 9:21)")
    ] = "1:1",
    output_format: Annotated[
        str, 
        Field(description="Output image format (png, jpg, jpeg, webp)")
    ] = "png",
    seed: Annotated[
        Optional[int], 
        Field(description="Reproducible generation seed (optional)")
    ] = None,
    input_image_path: Annotated[
        Optional[str], 
        Field(description="Path to reference image file (optional)")
    ] = None,
    output_path: Annotated[
        str, 
        Field(description="Full path where to save the generated image")
    ] = "images/flux_generated_image.png"
) -> FluxImageGenerationResult:
    """Generate an image using FLUX model (flux-kontext-pro)"""
    start_time = time.time()
    
    try:
        # Validate parameters
        if aspect_ratio not in FLUX_ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect_ratio. Must be one of: {', '.join(FLUX_ASPECT_RATIOS)}")
        
        if output_format not in FLUX_OUTPUT_FORMATS:
            raise ValueError(f"Invalid output_format. Must be one of: {', '.join(FLUX_OUTPUT_FORMATS)}")
        
        # Get API key
        api_key = get_api_key()
        
        # Initialize generator (without console output for MCP)
        generator = TuZiImageGenerator(api_key, console=None)
        
        # Handle input image if provided
        input_image_b64 = None
        if input_image_path:
            try:
                import base64
                with open(input_image_path, 'rb') as f:
                    image_data = f.read()
                    input_image_b64 = base64.b64encode(image_data).decode('utf-8')
                    # Add data URL prefix based on file extension
                    ext = os.path.splitext(input_image_path)[1].lower()
                    if ext in ['.jpg', '.jpeg']:
                        input_image_b64 = f"data:image/jpeg;base64,{input_image_b64}"
                    elif ext == '.png':
                        input_image_b64 = f"data:image/png;base64,{input_image_b64}"
                    elif ext == '.webp':
                        input_image_b64 = f"data:image/webp;base64,{input_image_b64}"
                    else:
                        input_image_b64 = f"data:image/png;base64,{input_image_b64}"  # Default to PNG
            except Exception as e:
                raise Exception(f"Failed to read input image: {e}")
        
        # Generate the image using FLUX
        result = generator.generate_flux_image(
            prompt=prompt,
            input_image=input_image_b64,
            seed=seed,
            aspect_ratio=aspect_ratio,
            output_format=output_format
        )
        
        # Extract image URLs using FLUX-specific method
        image_urls = generator.extract_flux_image_urls(result)
        
        # Use only the first image URL to simplify
        if not image_urls:
            raise Exception("No images were generated by FLUX")
        
        first_image_url = image_urls[0]
        
        # Parse output path
        output_dir = os.path.dirname(output_path) or "."
        base_name = os.path.splitext(os.path.basename(output_path))[0] or "flux_generated_image"
        
        # Download only the first image
        downloaded_files = generator.download_images(
            [first_image_url], 
            output_dir=output_dir, 
            base_name=base_name
        )
        
        downloaded_file = downloaded_files[0] if downloaded_files else ""
        
        generation_time = time.time() - start_time
        
        return FluxImageGenerationResult(
            success=True,
            message="FLUX image generated successfully",
            image_url=first_image_url,
            downloaded_file=downloaded_file,
            generation_time=generation_time,
            seed_used=seed,
            aspect_ratio=aspect_ratio,
            output_format=output_format
        )
        
    except Exception as e:
        generation_time = time.time() - start_time
        return FluxImageGenerationResult(
            success=False,
            message=f"Failed to generate FLUX image: {str(e)}",
            image_url="",
            downloaded_file="",
            generation_time=generation_time,
            seed_used=seed,
            aspect_ratio=aspect_ratio,
            output_format=output_format
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