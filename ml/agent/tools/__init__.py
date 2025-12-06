"""
Tools package for the TNTU assistant agent.

Each tool is stored in its own file for better organization and maintainability.
"""

from ml.agent.tools.tavily_search import tavily_search
from ml.agent.tools.vector_search import vector_search
from ml.agent.tools.web_fetch import web_fetch

# Export all available tools
AVAILABLE_TOOLS = [tavily_search, vector_search, web_fetch]

__all__ = ["AVAILABLE_TOOLS", "tavily_search", "vector_search", "web_fetch"]
