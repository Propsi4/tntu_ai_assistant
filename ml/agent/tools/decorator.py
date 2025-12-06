"""Decorator for converting Python functions into DSPy tools."""

from typing import Callable
import dspy


def dspy_tool(func: Callable) -> dspy.Tool:
    r"""
    Convert a Python function into a dspy.Tool.

    This decorator wraps a function with dspy.Tool, making it usable as both
    a regular Python function and as a DSPy tool for agent tool calling.

    Parameters
    ----------
    func : Callable
        The function to wrap as a DSPy tool. The function should have:
        - Clear docstrings describing what it does
        - Type hints for parameters and return values
        - Simple parameter types (str, int, bool, dict, list) or Pydantic models

    Returns
    -------
    dspy.Tool
        A dspy.Tool instance that wraps the original function.
        The tool can be called directly like a function and also used in DSPy agents.

    Examples
    --------
    >>> @dspy_tool
    ... def get_weather_by_city(city: str) -> dict:
    ...     \"\"\"Get the weather for a given city.\"\"\"
    ...     return get_weather_by_city(city)
    ...
    >>> tool = dspy_tool(get_weather_by_city)
    >>> tool(city="Kyiv")
    {"weather": "sunny", "temperature": 20}
    """
    return dspy.Tool(func)
