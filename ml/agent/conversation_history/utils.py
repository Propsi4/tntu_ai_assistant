"""Utility functions for message format conversion."""

from typing import List, Dict
from datetime import datetime, timezone
import dspy
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from ml.agent.schemas import AgentResponse


def langchain_to_dict(message: BaseMessage) -> Dict[str, str]:
    """
    Convert a LangChain message to a dictionary format.

    Parameters
    ----------
    message : BaseMessage
        LangChain message object (HumanMessage, AIMessage, or ToolMessage).

    Returns
    -------
    Dict[str, str]
        Dictionary with 'role' and 'content' keys.
        Format: {"role": "user"|"assistant"|"tool", "content": "..."}

    Examples
    --------
    >>> msg = HumanMessage(content="Hello")
    >>> dict_to_dspy_format([langchain_to_dict(msg)])
    """
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ToolMessage):
        role = "tool"
    else:
        # Default to user for unknown types
        role = "user"

    content = message.content if hasattr(message, "content") else str(message)
    return {"role": role, "content": content}


def dict_to_langchain(message_dict: Dict[str, str]) -> BaseMessage:
    """
    Convert a dictionary to a LangChain message object.

    Parameters
    ----------
    message_dict : Dict[str, str]
        Dictionary with 'role' and 'content' keys.

    Returns
    -------
    BaseMessage
        LangChain message object (HumanMessage, AIMessage, or ToolMessage).

    Examples
    --------
    >>> msg_dict = {"role": "user", "content": "Hello"}
    >>> msg = dict_to_langchain(msg_dict)
    """
    role = message_dict.get("role", "user")
    content = message_dict.get("content", "")

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "tool":
        return ToolMessage(content=content, tool_call_id="")
    else:
        # Default to user message
        return HumanMessage(content=content)


def langchain_messages_to_dicts(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """
    Convert a list of LangChain messages to dictionaries.

    Parameters
    ----------
    messages : List[BaseMessage]
        List of LangChain message objects.

    Returns
    -------
    List[Dict[str, str]]
        List of message dictionaries with 'role' and 'content' keys.
    """
    return [langchain_to_dict(msg) for msg in messages]


def dicts_to_langchain_messages(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    """
    Convert a list of dictionaries to LangChain messages.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        List of message dictionaries with 'role' and 'content' keys.

    Returns
    -------
    List[BaseMessage]
        List of LangChain message objects.
    """
    return [dict_to_langchain(msg) for msg in messages]


def dict_to_dspy_format(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert simple message dicts to DSPy-compatible format.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        List of message dictionaries with 'role' and 'content' keys.
        Format: [{"role": "user"|"assistant"|"tool", "content": "..."}, ...]

    Returns
    -------
    List[Dict[str, str]]
        List of DSPy-compatible message dictionaries.
        Format compatible with MessageLikeRepresentation.

    Examples
    --------
    >>> messages = [
    ...     {"role": "user", "content": "Hello"},
    ...     {"role": "assistant", "content": "Hi there!"}
    ... ]
    >>> dspy_format = dict_to_dspy_format(messages)
    """
    # DSPy expects messages in a format compatible with MessageLikeRepresentation
    # Include all three roles: user, assistant, and tool
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg.get("role") in ("user", "assistant", "tool") and msg.get("content")
    ]


def dspy_prediction_to_response(
    prediction: dspy.Prediction, session_id: str
) -> AgentResponse:
    """
    Convert a DSPy Prediction to an AgentResponse schema.

    Parameters
    ----------
    prediction : dspy.Prediction
        The prediction object from DSPy agent.
    session_id : str
        The session identifier.

    Returns
    -------
    AgentResponse
        AgentResponse object with the agent's response.

    Examples
    --------
    >>> prediction = dspy.Prediction(response="Hello, how can I help?")
    >>> response = dspy_prediction_to_response(prediction, "session123")
    """
    return AgentResponse(
        response=prediction.response,
        status="success",
        timestamp=datetime.now(timezone.utc),
        session_id=session_id,
    )
