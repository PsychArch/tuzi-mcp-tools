# Tuzi MCP Tools

A Python package providing both **CLI** and **MCP server** interfaces for generating images and conducting surveys using the Tu-zi.com API.

## Features

- **Dual Interface**: CLI and MCP server
- **Image Generation**: Automatic model fallback system (tries models from low to high price)
- **Survey/Query**: o3-all model with web search capabilities
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
# Generate image with automatic model selection
tuzi image "A beautiful sunset over mountains"

# High quality with custom options
tuzi image "A cute cat" --quality high --size 1024x1536 --format png

# Transparent background
tuzi image "Company logo" --background transparent --output logo.png
```

### Survey/Query

```bash
# Ask a question with web search capabilities
tuzi survey "What are the latest developments in AI?"

# Get current information
tuzi survey "What is the current weather in New York?"

# Show the thinking process
tuzi survey "Explain quantum computing" --show-thinking
```

### CLI Options

#### Image Generation Options (`tuzi image`)

| Option | Description | Default |
|--------|-------------|---------|
| `--quality` | Image quality (low, medium, high, auto) | `auto` |
| `--size` | Dimensions (1024x1024, 1536x1024, 1024x1536, auto) | `auto` |
| `--format` | Output format (png, jpeg, webp) | `png` |
| `--background` | Background (opaque, transparent) | `opaque` |
| `--output` | Output file path | auto-generated |
| `--compression` | Compression level 0-100 (JPEG/WebP) | `None` |
| `--no-stream` | Disable streaming response | `False` |
| `--verbose` | Show full API response | `False` |

#### Survey Options (`tuzi survey`)

| Option | Description | Default |
|--------|-------------|---------|
| `--no-stream` | Disable streaming response | `False` |
| `--verbose` | Show detailed response information | `False` |
| `--show-thinking` | Show thinking process in addition to final answer | `False` |

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