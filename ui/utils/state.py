"""State management utilities for the UI."""

import uuid
import streamlit as st
from typing import List, Dict, Any


def init_session_state():
    """Initialize the session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "model_name" not in st.session_state:
        from ui.config.settings import settings
        st.session_state.model_name = settings.DEFAULT_LLM

    if "temperature" not in st.session_state:
        from ui.config.settings import settings
        st.session_state.temperature = settings.DEFAULT_LLM_TEMPERATURE


def get_session_id() -> str:
    """
    Get the current session ID.

    Returns
    -------
    str
        The session ID.
    """
    if "session_id" not in st.session_state:
        init_session_state()
    return st.session_state.session_id


def set_session_id(session_id: str):
    """
    Set the current session ID.

    Parameters
    ----------
    session_id : str
        The new session ID.
    """
    st.session_state.session_id = session_id
    # Clear messages when switching sessions to force a refresh
    st.session_state.messages = []


def add_message(role: str, content: str):
    """
    Add a message to the chat history.

    Parameters
    ----------
    role : str
        The sender role ('user' or 'assistant').
    content : str
        The message content.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": role, "content": content})


def get_messages() -> List[Dict[str, Any]]:
    """
    Get the chat history.

    Returns
    -------
    List[Dict[str, Any]]
        List of messages.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages


def set_messages(messages: List[Dict[str, Any]]):
    """
    Replace the stored chat history.

    Parameters
    ----------
    messages : List[Dict[str, Any]]
        List of message dictionaries to store in session state.

    Returns
    -------
    None
        This function updates ``st.session_state`` in place.
    """
    st.session_state.messages = list(messages)
