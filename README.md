# Tuzi MCP Tools

A Python package providing both **CLI** and **MCP server** interfaces for generating images using the Tu-zi.com API.

## Features

- **Dual Interface**: CLI and MCP server
- **Automatic Model Fallback**: Tries models from low to high price
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

### Basic Commands

```bash
# Generate image with automatic model selection
tuzi "A beautiful sunset over mountains"

# High quality with custom options
tuzi "A cute cat" --quality high --size 1024x1536 --format png

# Transparent background
tuzi "Company logo" --background transparent --output logo.png
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--quality` | Image quality (low, medium, high, auto) | `auto` |
| `--size` | Dimensions (1024x1024, 1536x1024, 1024x1536, auto) | `auto` |
| `--format` | Output format (png, jpeg, webp) | `png` |
| `--background` | Background (opaque, transparent) | `opaque` |
| `--output` | Output file path | `images/generated_image.png` |
| `--compression` | Compression level 0-100 (JPEG/WebP) | `None` |

## MCP Server Usage

```json
{
  "mcpServers": {
    "tuzi-image-generator": {
      "command": "tuzi-mcp-server",
      "env": {
        "TUZI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```