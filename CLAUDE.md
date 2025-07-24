# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tuzi MCP Tools is a Python package that provides both CLI and MCP (Model Context Protocol) server interfaces for generating images and conducting surveys using the Tu-zi.com API. The project features automatic model fallback for image generation, o3-all model integration for surveys with web search capabilities, rich CLI output with progress tracking, automatic conversation management with circular buffering, and comprehensive error handling.

## Development Tools

- We use uv to manage the project.

## Development Commands

### Installation and Setup
```bash
# Install with pipx (recommended)
pipx install tuzi-mcp-tools

# Set API key
export TUZI_API_KEY='your_api_key_here'

# For development, create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .
```

[Rest of the file remains unchanged...]