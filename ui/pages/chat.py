"""Chat page implementation."""

import asyncio
from typing import Any, Dict, List

import streamlit as st
from ui.api.client import TNTUClient
from ui.components.chat import render_message
from ui.components.sidebar import render_sidebar
from ui.config.settings import settings
from ui.utils.state import (
    add_message,
    get_messages,
    get_session_id,
    init_session_state,
    set_messages,
)

st.set_page_config(
    page_title=f"Chat - {settings.PAGE_TITLE}",
    page_icon="💬",
    layout=settings.LAYOUT
)

# Initialize state and client
init_session_state()
client = TNTUClient()

# Sidebar
render_sidebar()

# Main content
st.title("💬 Chat with TNTU Assistant")

session_id = get_session_id()


def load_history_if_needed(session_key: str) -> None:
    """
    Load chat history once per session and rerun to render it.

    Parameters
    ----------
    session_key : str
        The session identifier used to fetch history.

    Returns
    -------
    None
        Sets messages in session state and triggers ``st.rerun``.
    """
    if st.session_state.get("last_loaded_session") == session_key:
        return

    messages: List[Dict[str, Any]] = []
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        history = loop.run_until_complete(client.get_history(session_key))
        messages = [msg for msg in history.get("messages", []) if msg.get("role") != "tool"]
    except Exception as exc:
        st.warning(f"Could not load history for session {session_key}: {exc}")
        return
    finally:
        loop.close()

    set_messages(messages)
    st.session_state.last_loaded_session = session_key
    st.rerun()


load_history_if_needed(session_id)

# Display chat history
messages = get_messages()
for msg in messages:
    render_message(msg)

# Chat input
if prompt := st.chat_input("Ask a question about TNTU..."):
    # Add user message
    add_message("user", prompt)
    render_message({"role": "user", "content": prompt})

    # Get assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        async def stream_response() -> str:
            """Stream and assemble the response from the API."""
            result_text = ""
            async for event in client.chat_stream(
                message=prompt,
                session_id=get_session_id(),
                model_name=st.session_state.get("model_name"),
                temperature=st.session_state.get("temperature")
            ):
                event_type = event.get("type")
                data = event.get("data", {}) if isinstance(event, dict) else {}

                if event_type == "token":
                    token = data.get("token", "")
                    result_text += token
                    response_placeholder.markdown(result_text + "▌")
                elif event_type == "complete":
                    # When complete, optionally use final response if provided
                    final_response = data.get("response", result_text)
                    result_text = final_response
                    response_placeholder.markdown(result_text)
                elif event_type == "error":
                    error_msg = data.get("error", "Streaming error")
                    raise RuntimeError(error_msg)
                # Ignore other event types (e.g., steps)

            return result_text

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            final_response = loop.run_until_complete(stream_response())
            loop.close()

            response_placeholder.markdown(final_response)

            # Add assistant message to history
            add_message("assistant", final_response)

        except Exception as e:
            st.error(f"Error: {str(e)}")
