import os
import asyncio
from urllib.parse import urlencode

from dotenv import load_dotenv
from mcp import Client

load_dotenv()

TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"


async def _search_with_mcp(query: str,max_results: int = 3,) -> str:
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing from .env")

    # Build the authenticated remote MCP URL.
    url = TAVILY_MCP_URL + "?" + urlencode({
        "tavilyApiKey": api_key
    })

    # Connect to Tavily's hosted MCP server.
    async with Client(url) as client:

        # Discover the tools offered by the remote server.
        tools_response = await client.list_tools()

        available_tools = {
            tool.name: tool
            for tool in tools_response.tools
        }

        # Tavily's hosted server uses the tavily-search tool.
        tool_name = next(
            (
                name for name in available_tools
                if name in ("tavily-search", "tavily_search")
            ),
            None,
        )

        if tool_name is None:
            raise RuntimeError(
                "Tavily search tool not found. "
                f"Available tools: {list(available_tools)}"
            )

        # Call the remote MCP tool.
        result = await client.call_tool(
            tool_name,
            {
                "query": query,
                "max_results": max_results,
            },
        )

        if result.is_error:
            raise RuntimeError(
                f"Tavily MCP returned an error: {result.content}"
            )

        # Convert the MCP response into readable text.
        text_parts = [
            block.text
            for block in result.content
            if hasattr(block, "text")
        ]

        return "\n\n".join(text_parts) or "No search results returned."


def call_tavily_search(
    query: str,
    max_results: int = 3,
) -> str:
    """Synchronous wrapper for the LangGraph tool."""

    return asyncio.run(
        _search_with_mcp(query, max_results)
    )