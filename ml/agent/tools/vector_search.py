"""Vector search tool implementation for RAG."""

from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import text

from ml.config.settings import settings
from ml.agent.tools.decorator import dspy_tool
from ml.agent.conversation_history.schema import get_engine, get_db
from ml.agent.rag.models import Document as DBDocument
from ml.config.logger import get_logger
import json

logger = get_logger(__name__)


def _ensure_pgvector_extension():
    """Ensure pgvector extension exists."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()


def _ensure_table_exists():
    """Ensure documents table exists."""
    engine = get_engine()
    DBDocument.metadata.create_all(engine)


@dspy_tool
def vector_search(query: str) -> List[Document]:
    """
    Search for relevant university documents using vector similarity.

    This tool performs a semantic search over the university knowledge base using the provided query.
    It uses OpenAI embeddings to convert the query into a vector and finds the most similar
    documents in the database using cosine similarity (L2 distance).

    Parameters
    ----------
    query : str
        The search query string (e.g. "When is the admission deadline?", "History of Ivan Puluj").

    Returns
    -------
    List[Document]
        A list of relevant LangChain Documents containing content and metadata.
    """
    try:
        # Ensure DB setup
        _ensure_pgvector_extension()
        _ensure_table_exists()

        # Generate embedding
        embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        query_embedding = embeddings.embed_query(query)

        # Query database
        with get_db() as db:
            # Using L2 distance <-> operator which is supported by pgvector
            results = db.query(DBDocument).order_by(
                DBDocument.embedding.l2_distance(query_embedding)
            ).limit(5).all()

            documents = []
            for doc in results:
                documents.append(
                    Document(
                        page_content=doc.content,
                        metadata=doc.doc_metadata or {}
                    )
                )

            return json.dumps({"documents": documents}, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in vector_search: {str(e)}")
        return []
