"""
Refactored MCP Server using Clean Architecture

This module provides a Model Context Protocol (MCP) server interface
using the new clean architecture with proper dependency injection.
"""

import argparse
from typing import Optional, Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ..infrastructure.container import get_mcp_container

# Create the MCP server
mcp = FastMCP("Tuzi Tools - Image Generator and Survey")

# Get the service container for MCP mode
container = get_mcp_container()

# Set up conversation service for memory storage (MCP mode)
conversation_service = container.get_conversation_service(storage_mode="memory")

# Get application services
image_service = container.get_image_service()
survey_service = container.get_survey_service()
task_service = container.get_task_service()


# Pydantic models for structured responses


@mcp.tool()
async def submit_gpt_image(
    prompt: Annotated[str, Field(description="The text prompt for image generation")],
    output_path: Annotated[
        str, Field(description="Absolute path where to save the generated image")
    ],
    quality: Annotated[
        Literal["auto", "low", "medium", "high"],
        Field(description="Image quality setting"),
    ] = "auto",
    size: Annotated[
        Literal["auto", "1024x1024", "1536x1024", "1024x1536"],
        Field(description="Image dimensions"),
    ] = "auto",
    format: Annotated[
        Literal["png", "jpeg", "webp"], Field(description="Output image format")
    ] = "png",
    background: Annotated[
        Literal["opaque", "transparent"],
        Field(description="Background type for the image"),
    ] = "opaque",
    compression: Annotated[
        Optional[int],
        Field(description="Output compression 0-100 for JPEG/WebP formats"),
    ] = None,
    conversation_id: Annotated[
        Optional[str],
        Field(
            description="Conversation ID to continue an existing conversation (optional)"
        ),
    ] = None,
    input_image: Annotated[
        Optional[str], Field(description="Base64 encoded reference image (optional)")
    ] = None,
) -> str:
    """Submit a GPT image generation task for async processing. Returns task ID immediately."""

    try:
        task_id = await task_service.submit_gpt_image_task(
            prompt=prompt,
            output_path=output_path,
            quality=quality,
            size=size,
            format=format,
            background=background,
            output_compression=compression,
            input_image=input_image,
            conversation_id=conversation_id,
        )

        return f"{task_id} task submitted"

    except Exception as e:
        return f"Failed to submit task: {str(e)}"


@mcp.tool()
async def submit_flux_image(
    prompt: Annotated[
        str, Field(description="The text prompt for FLUX image generation")
    ],
    output_path: Annotated[
        str, Field(description="Absolute path where to save the generated image")
    ],
    aspect_ratio: Annotated[
        Literal["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
        Field(description="Image aspect ratio"),
    ] = "1:1",
    output_format: Annotated[
        Literal["png", "jpg", "jpeg", "webp"], Field(description="Output image format")
    ] = "png",
    seed: Annotated[
        Optional[int], Field(description="Reproducible generation seed (optional)")
    ] = None,
    conversation_id: Annotated[
        Optional[str],
        Field(
            description="Conversation ID to continue an existing conversation (optional)"
        ),
    ] = None,
    input_image: Annotated[
        Optional[str], Field(description="Base64 encoded reference image (optional)")
    ] = None,
) -> str:
    """Submit a FLUX image generation task for async processing. Returns task ID immediately."""

    try:
        task_id = await task_service.submit_flux_image_task(
            prompt=prompt,
            output_path=output_path,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
            seed=seed,
            input_image=input_image,
            conversation_id=conversation_id,
        )

        return f"{task_id} task submitted"

    except Exception as e:
        return f"Failed to submit task: {str(e)}"


@mcp.tool()
async def task_barrier(
    output_dir: Annotated[
        str, Field(description="Directory where to save all generated images")
    ] = "images",
) -> str:
    """Wait for all submitted image generation tasks to complete and download their results."""

    try:
        # Wait for all tasks to complete
        results = await task_service.wait_for_all_tasks()

        # Format detailed results
        output_lines = []

        # Report successful tasks
        for task_id, task_result in results["results"].items():
            if task_result["success"]:
                conversation_id = task_result.get("conversation_id", "unknown")
                output_lines.append(
                    f"{task_id}: success (conversation_id: {conversation_id})"
                )
            else:
                error_msg = task_result["error"] or "Unknown error"
                conversation_id = task_result.get("conversation_id", "unknown")
                output_lines.append(
                    f"{task_id}: failed - {error_msg} (conversation_id: {conversation_id})"
                )

        return "\n".join(output_lines) if output_lines else "No tasks to report"

    except Exception as e:
        return f"Task barrier failed: {str(e)}"


def get_global_generator():
    """Get the global image generator instance for MCP server compatibility"""
    # This function maintains compatibility with the old MCP server
    # Returns the image service from the container
    return container.get_image_service()


def run_server():
    """Run the MCP server"""
    parser = argparse.ArgumentParser(description="Tuzi MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol to use",
    )
    parser.add_argument("--host", default="localhost", help="Host for HTTP transport")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP transport"
    )

    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", host=args.host, port=args.port)


if __name__ == "__main__":
    run_server()
