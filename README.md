# Tuzi MCP Tools

A Python package providing both **CLI** and **MCP server** interfaces for generating images and conducting surveys using the Tu-zi.com API.

## Features

- **Dual Interface**: CLI and MCP server
- **Image Generation**: Automatic model fallback system (tries models from low to high price)
- **FLUX Image Generation**: High-quality image generation using flux-kontext-pro model with reference image support
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

### FLUX Image Generation

```bash
# Generate high-quality image with FLUX model
tuzi flux "A beautiful sunset over mountains"

# Custom aspect ratio and format
tuzi flux "A futuristic cityscape" --aspect-ratio 16:9 --format webp

# With reference image for style transfer or modification
tuzi flux "Make this more colorful" --input-image reference.jpg --seed 42

# Transform existing image with new prompt
tuzi flux "Turn this into a cyberpunk scene" --input-image photo.png --aspect-ratio 16:9

# Ultra-wide panoramic image
tuzi flux "Mountain landscape panorama" --aspect-ratio 21:9 --output panorama.png
```

### Survey/Query

```bash
# Ask a question with web search capabilities
tuzi survey "What are the latest developments in AI?"

# Get current information
tuzi survey "What is the current weather in New York?"

# Show the thinking process
tuzi survey "Explain quantum computing" --show-thinking

# Use o3-pro for deeper analysis
tuzi survey "Analyze the implications of quantum computing on cryptography" --deep
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
| `--deep` | Use o3-pro for deeper analysis (default: o3-all) | `False` |

#### FLUX Generation Options (`tuzi flux`)

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
    "tuzi-image-generator": {
      "command": "tuzi-mcp",
      "env": {
        "TUZI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

### Available MCP Tools

#### `generate_image`
Generate images using automatic model fallback system.

**Parameters:**
- `prompt` (string): Text prompt for image generation
- `quality` (string): Image quality (auto, low, medium, high)
- `size` (string): Image dimensions (auto, 1024x1024, 1536x1024, 1024x1536)
- `format` (string): Output format (png, jpeg, webp)
- `background` (string): Background type (opaque, transparent)
- `compression` (integer): Compression level 0-100 for JPEG/WebP
- `output_path` (string): Full path where to save the image

#### `generate_flux_image`
Generate high-quality images using FLUX model (flux-kontext-pro).

**Parameters:**
- `prompt` (string): Text prompt for FLUX image generation
- `aspect_ratio` (string): Image aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4, 21:9, 9:21)
- `output_format` (string): Output format (png, jpg, jpeg, webp)
- `seed` (integer, optional): Reproducible generation seed
- `input_image_path` (string, optional): Path to reference image file
- `output_path` (string): Full path where to save the image

#### `survey`
Conduct surveys/queries using o3-all or o3-pro model with web search capabilities.

**Parameters:**
- `prompt` (string): Natural language query/question
- `show_thinking` (boolean): Whether to include thinking process in response
- `deep` (boolean): Whether to use o3-pro for deeper analysis (default: False, uses o3-all)

## Model Information

### Standard Image Generation Models
- **gpt-image-1**: $0.04 per image
- **gpt-4o-image**: $0.04 per image
- **gpt-4o-image-vip**: $0.10 per image
- **gpt-image-1-vip**: $0.10 per image

### FLUX Model
- **flux-kontext-pro**: High-quality image generation with advanced features
  - Support for multiple aspect ratios
  - Reference image input
  - Reproducible generation with seeds
  - Enhanced prompt processing

### Survey Models
- **o3-all**: Advanced reasoning model with web search capabilities (default)
- **o3-pro**: Enhanced reasoning model for deeper analysis with web search capabilities