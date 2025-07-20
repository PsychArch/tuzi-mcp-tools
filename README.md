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

### CLI Options

#### Image Generation Options (`tuzi gpt-image`)

| Option | Description | Default |
|--------|-------------|---------|
| `--quality` | Image quality (low, medium, high, auto) | `auto` |
| `--size` | Dimensions (1024x1024, 1536x1024, 1024x1536, auto) | `auto` |
| `--format` | Output format (png, jpeg, webp) | `png` |
| `--background` | Background (opaque, transparent) | `opaque` |
| `--output` | Output file path | auto-generated |
| `--input-image` | Path to reference image file | `None` |
| `--compression` | Compression level 0-100 (JPEG/WebP) | `None` |
| `--no-stream` | Disable streaming response | `False` |
| `--verbose` | Show full API response | `False` |

#### Survey Options (`tuzi survey`)

| Option | Description | Default |
|--------|-------------|---------|
| `--no-stream` | Disable streaming response | `False` |
| `--verbose` | Show detailed response information | `False` |
| `--show-thinking` | Show thinking process in addition to final answer | `False` |
| `--deep` | Enable advanced analysis mode | `False` |

#### FLUX Generation Options (`tuzi flux-image`)

| Option | Description | Default |
|--------|-------------|---------|
| `--aspect-ratio` | Image aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4, 21:9, 9:21) | `1:1` |
| `--format` | Output format (png, jpg, jpeg, webp) | `png` |
| `--seed` | Reproducible generation seed | `None` |
| `--input-image` | Path to reference image file | `None` |
| `--output` | Output file path | auto-generated |
| `--verbose` | Show full API response | `False` |

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

#### `gpt_image`
Generate images using gpt.

**Parameters:**
- `prompt` (string): Text prompt for image generation
- `quality` (string): Image quality (auto, low, medium, high)
- `size` (string): Image dimensions (auto, 1024x1024, 1536x1024, 1024x1536)
- `format` (string): Output format (png, jpeg, webp)
- `background` (string): Background type (opaque, transparent)
- `compression` (integer): Compression level 0-100 for JPEG/WebP
- `output_path` (string): Full path where to save the image
- `input_image_path` (string, optional): Path to reference image file

#### `flux_image`
Generate images using flux.

**Parameters:**
- `prompt` (string): Text prompt for FLUX image generation
- `aspect_ratio` (string): Image aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4, 21:9, 9:21)
- `output_format` (string): Output format (png, jpg, jpeg, webp)
- `seed` (integer, optional): Reproducible generation seed
- `input_image_path` (string, optional): Path to reference image file
- `output_path` (string): Full path where to save the image

#### `survey`
Survey/query a topic using advanced AI with real-time web search capabilities.

**Parameters:**
- `prompt` (string): Natural language query/question
- `show_thinking` (boolean): Whether to include thinking process in response
- `deep` (boolean): Whether to enable advanced analysis mode (default: False)