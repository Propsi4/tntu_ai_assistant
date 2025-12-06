"""Sidebar component for the UI."""

import streamlit as st
from ui.config.settings import settings
from ui.utils.state import set_session_id
import uuid


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.title(f"{settings.PAGE_ICON} {settings.PAGE_TITLE}")

        st.divider()

        st.subheader("Session Management")

        # Session ID input/display
        current_session_id = st.session_state.get(
            "session_id", str(uuid.uuid4())
        )
        new_session_id = st.text_input("Session ID", value=current_session_id)

        if new_session_id != current_session_id:
            set_session_id(new_session_id)
            st.rerun()

        if st.button("New Session", width="content"):
            new_id = str(uuid.uuid4())
            set_session_id(new_id)
            st.rerun()

        st.divider()

        st.subheader("Model Settings")

        model = st.selectbox(
            "Model",
            ["gpt-5.1", "gpt-4o", "gpt-4o-mini"],
            index=0
        )
        st.session_state.model_name = model

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=settings.DEFAULT_LLM_TEMPERATURE,
            step=0.1
        )
        st.session_state.temperature = temperature

        st.divider()

        st.caption("Version: 0.1.0")
