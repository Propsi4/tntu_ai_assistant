"""Chat component for rendering messages."""

import streamlit as st
from typing import Dict, Any


def render_message(message: Dict[str, Any]):
    """
    Render a single chat message.

    Parameters
    ----------
    message : Dict[str, Any]
        The message dictionary containing 'role' and 'content'.
    """
    role = message.get("role", "user")
    content = message.get("content", "")

    with st.chat_message(role):
        st.markdown(content)
