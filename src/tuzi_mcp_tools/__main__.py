#!/usr/bin/env python3
"""
Main entry point for the tuzi-mcp-tools package.
This allows running the MCP server with: python -m tuzi_mcp_tools
"""

from .mcp_server import main

if __name__ == "__main__":
    main() 