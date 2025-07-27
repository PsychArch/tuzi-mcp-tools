# Tuzi MCP Tools

[![PyPI version](https://badge.fury.io/py/tuzi-mcp-tools.svg)](https://badge.fury.io/py/tuzi-mcp-tools)

English | [简体中文](README_zh.md)

A Python package providing both **CLI** and **MCP server** interfaces for generating images and conducting surveys using the Tu-zi.com API.

## Features

- **Dual Interface**: CLI and MCP server with clean architecture
- **Async Task Management**: Submit/barrier pattern for efficient parallel processing
- **GPT Image Generation**: GPT text to image with conversation continuity
- **FLUX Image Generation**: FLUX text to image with conversation tracking
- **Conversation Management**: Continue image editing across multiple tasks

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

The MCP server provides **async task management** with submit/barrier pattern for efficient image generation.

#### `submit_gpt_image`
Submit a GPT image generation task for async processing. Returns task ID immediately.

#### `submit_flux_image` 
Submit a FLUX image generation task for async processing. Returns task ID immediately.

#### `task_barrier`
Wait for all submitted image generation tasks to complete and download their results. Reports conversation IDs for task tracking.

## CLI Usage

```
uvx --from tuzi-mcp-tools tuzi --help
```