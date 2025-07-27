#!/usr/bin/env python3
"""
Main entry point for the tuzi-mcp-tools package.
This allows running the MCP server with: python -m tuzi_mcp_tools
"""

from .interfaces.mcp_server import run_server

if __name__ == "__main__":
    run_server() 