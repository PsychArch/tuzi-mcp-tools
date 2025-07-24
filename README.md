# Tuzi MCP Tools

[![PyPI version](https://badge.fury.io/py/tuzi-mcp-tools.svg)](https://badge.fury.io/py/tuzi-mcp-tools)

English | [简体中文](README_zh.md)

A Python package providing both **CLI** and **MCP server** interfaces for generating images and conducting surveys using the Tu-zi.com API.

## Features

- **Dual Interface**: CLI and MCP server
- **GPT Image Generation**: GPT text to image 
- **FLUX Image Generation**: FLUX text to image
- **Survey/Query**: Advanced AI-powered research with real-time web search capabilities
- **Multiple Formats**: PNG, JPEG, WebP with quality settings
- **Real-time Progress**: Streaming generation with progress tracking

## Installation

```bash
pipx install tuzi-mcp-tools
```

## Setup

Set your Tu-zi.com API key:

```bash
export TUZI_API_KEY='your_api_key_here'
```

## MCP Server Usage

```json
{
  "mcpServers": {
    "tuzi": {
      "command": "uvx",
      "args": ["tuzi-mcp-tools"],
      "env": {
        "TUZI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

### Available MCP Tools

The MCP server provides three tools that correspond to the CLI commands. All parameters are documented in the [Common Arguments](#common-arguments) section below.

#### `gpt_image`
Generate images using GPT with automatic model fallback.

#### `flux_image`
Generate images using FLUX for premium quality.

#### `survey`
Survey/query topics using advanced AI with real-time web search capabilities.

## CLI Usage

### Image Generation

```bash
# Generate image with intelligent model selection
tuzi gpt-image "A beautiful sunset over mountains"

# High quality with custom options
tuzi gpt-image "A cute cat" --quality high --size 1024x1536 --format png

# Transparent background
tuzi gpt-image "Company logo" --background transparent --output logo.png

# With reference image for style transfer or modification
tuzi gpt-image "Make this more colorful" --input-image reference.jpg

# Transform existing image with new prompt
tuzi gpt-image "Turn this into a cyberpunk scene" --input-image photo.png
```

### FLUX Image Generation

```bash
# Generate high-quality image with FLUX
tuzi flux-image "A beautiful sunset over mountains"

# Custom aspect ratio and format
tuzi flux-image "A futuristic cityscape" --aspect-ratio 16:9 --format webp

# With reference image for style transfer or modification
tuzi flux-image "Make this more colorful" --input-image reference.jpg --seed 42

# Transform existing image with new prompt
tuzi flux-image "Turn this into a cyberpunk scene" --input-image photo.png --aspect-ratio 16:9

# Ultra-wide panoramic image
tuzi flux-image "Mountain landscape panorama" --aspect-ratio 21:9 --output panorama.png
```

### Survey/Query

```bash
# Ask a question with web search capabilities
tuzi survey "What are the latest developments in AI?"

# Get current information
tuzi survey "What is the current weather in New York?"

# Show the thinking process
tuzi survey "Explain quantum computing" --show-thinking

# Enable deep analysis mode for complex topics
tuzi survey "Analyze the implications of quantum computing on cryptography" --deep
```

### Common Arguments

The following arguments are available for both CLI and MCP server interfaces:

#### GPT Image Generation Arguments

| Argument | CLI Option | MCP Parameter | Description | Default |
|----------|------------|---------------|-------------|---------|
| `prompt` | (positional) | `prompt` | Text prompt for image generation | - |
| `quality` | `--quality` | `quality` | Image quality (low, medium, high, auto) | `auto` |
| `size` | `--size` | `size` | Dimensions (1024x1024, 1536x1024, 1024x1536, auto) | `auto` |
| `format` | `--format` | `format` | Output format (png, jpeg, webp) | `png` |
| `background` | `--background` | `background` | Background (opaque, transparent) | `opaque` |
| `compression` | `--compression` | `compression` | Compression level 0-100 (JPEG/WebP) | `None` |
| `output_path` | `--output` | `output_path` | Output file path | auto-generated |
| `input_image` | `--input-image` | `input_image_path` | Path to reference image file | `None` |
| `conversation_id` | `--conversation-id` | `conversation_id` | Conversation ID for context | `None` |
| `close_conversation` | `--close-conversation` | `close_conversation` | Close conversation after request | `False` |

#### FLUX Image Generation Arguments

| Argument | CLI Option | MCP Parameter | Description | Default |
|----------|------------|---------------|-------------|---------|
| `prompt` | (positional) | `prompt` | Text prompt for FLUX image generation | - |
| `aspect_ratio` | `--aspect-ratio` | `aspect_ratio` | Image aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4, 21:9, 9:21) | `1:1` |
| `output_format` | `--format` | `output_format` | Output format (png, jpg, jpeg, webp) | `png` |
| `seed` | `--seed` | `seed` | Reproducible generation seed | `None` |
| `input_image` | `--input-image` | `input_image_path` | Path to reference image file | `None` |
| `output_path` | `--output` | `output_path` | Output file path | auto-generated |
| `conversation_id` | `--conversation-id` | `conversation_id` | Conversation ID for context | `None` |
| `close_conversation` | `--close-conversation` | `close_conversation` | Close conversation after request | `False` |

#### Survey/Query Arguments

| Argument | CLI Option | MCP Parameter | Description | Default |
|----------|------------|---------------|-------------|---------|
| `prompt` | (positional) | `prompt` | Natural language query/question | - |
| `show_thinking` | `--show-thinking` | `show_thinking` | Show thinking process in addition to final answer | `False` |
| `deep` | `--deep` | `deep` | Enable advanced analysis mode | `False` |
| `conversation_id` | `--conversation-id` | `conversation_id` | Conversation ID for context | `None` |
| `close_conversation` | `--close-conversation` | `close_conversation` | Close conversation after request | `False` |

#### CLI-Only Options

| Option | Description | Default |
|--------|-------------|---------|
| `--no-stream` | Disable streaming response | `False` |
| `--verbose` | Show detailed response information | `False` |