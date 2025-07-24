"""
Tuzi Image Generator MCP Server

This module provides a Model Context Protocol (MCP) server interface
for the Tuzi image generation service using the new clean architecture.
This is a compatibility wrapper around the new interface.
"""

# Import the new MCP server interface
from .interfaces.mcp_server import mcp, run_server, get_global_generator

# Re-export everything for backward compatibility
__all__ = ["mcp", "run_server", "get_global_generator", "main"]

# Entry point function for the script
def main():
    """Main entry point for the MCP server."""
    run_server()

if __name__ == "__main__":
    main()