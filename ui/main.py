"""Main entry point for the UI."""

import streamlit as st
import sys
from streamlit.web import cli as stcli
from streamlit import Page


def main():
    """Run the Streamlit UI."""
    nav = st.navigation([
        Page(page="pages/home.py", title="Home", icon="🏠", default=True),
        Page(page="pages/chat.py", title="Chat", icon="✨"),
        Page(page="pages/knowledge_base.py", title="Knowledge Base", icon="📚"),
        Page(page="pages/history.py", title="History", icon="📜"),
    ])
    nav.run()


def run_ui():
    """Entry point for poetry script."""
    sys.argv = ["streamlit", "run", "ui/main.py"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
