"""Main page implementation."""

import streamlit as st
from ui.components.sidebar import render_sidebar
from ui.utils.state import init_session_state
from ui.config.settings import settings

st.set_page_config(
    page_title=settings.PAGE_TITLE,
    page_icon=settings.PAGE_ICON,
    layout=settings.LAYOUT
)

# Initialize state and client
init_session_state()

# Sidebar
render_sidebar()

# Main content
st.title(f"{settings.PAGE_ICON} Welcome to {settings.PAGE_TITLE}")

st.markdown(
    """
    This is the official interface for the **TNTU Assistant AI**.

    ### Features

    - **💬 Chat**: Interact with the AI agent to get information about
        the university.
    - **📚 Knowledge Base**: Upload documents (PDF, DOCX, PPTX) and index
        URLs to expand the agent's knowledge.
    - **📜 History**: View and manage your chat sessions.

    ### Getting Started

    Select a page from the sidebar to begin.
    """
)
