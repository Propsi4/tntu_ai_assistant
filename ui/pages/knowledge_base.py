"""Knowledge Base management page."""

import asyncio
import pandas as pd
import streamlit as st

from ui.api.client import TNTUClient
from ui.components.rag import render_file_uploader, render_url_indexer
from ui.components.sidebar import render_sidebar
from ui.config.settings import settings
from ui.utils.state import init_session_state

st.set_page_config(
    page_title=f"Knowledge Base - {settings.PAGE_TITLE}",
    page_icon="📚",
    layout=settings.LAYOUT
)

init_session_state()
client = TNTUClient()
render_sidebar()

st.title("📚 Knowledge Base Management")

# Input section
col1, col2 = st.columns(2)

with col1:
    render_file_uploader(client)

with col2:
    render_url_indexer(client)

st.divider()

# Document List
st.subheader("Indexed Documents")

try:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(client.list_documents(limit=100))
    loop.close()
    docs = response.get("items", [])
    total = response.get("total", 0)

    st.caption(f"Total documents: {total}")

    if docs:
        # Prepare data for dataframe
        data = []
        for doc in docs:
            metadata = doc.get("metadata", {})
            data.append({
                "ID": doc["id"],
                "Content Preview": doc["content"][:100] + "...",
                "Source": metadata.get("source", "Unknown"),
                "Type": metadata.get("content_type", "Unknown"),
            })

        df = pd.DataFrame(data)
        st.dataframe(
            df,
            width="content",
            column_config={
                "ID": st.column_config.NumberColumn(width="small"),
                "Content Preview": st.column_config.TextColumn(width="large"),
            },
            hide_index=True
        )

        # Delete functionality
        st.subheader("Delete Document")
        doc_id = st.number_input(
            "Enter Document ID to delete", min_value=1, step=1
        )
        if st.button("Delete", type="primary"):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(client.delete_document(doc_id))
                loop.close()
                st.success(f"Document {doc_id} deleted")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting document: {str(e)}")
    else:
        st.info("No documents found in the knowledge base.")

except Exception as e:
    st.error(f"Failed to load documents: {str(e)}")
