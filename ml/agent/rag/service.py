"""Service layer for RAG document management."""

from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings

from ml.config.settings import settings
from ml.config.logger import get_logger
from ml.agent.rag.models import Document as DBDocument
from ml.agent.rag.schemas import DocumentResponse
from ml.agent.conversation_history.schema import get_db
from ml.agent.rag.parsers.factory import ParserFactory
from ml.agent.rag.chunker import Chunker

logger = get_logger(__name__)


class RagService:
    """Service for managing RAG documents."""

    def __init__(self):
        """Initialize the RAG service."""
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        self.chunker = Chunker()

    async def index_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> DocumentResponse:
        """
        Index a document by generating embedding and saving to database.

        Parameters
        ----------
        content : str
            The text content to index.
        metadata : Optional[Dict[str, Any]]
            Optional metadata associated with the document.

        Returns
        -------
        DocumentResponse
            The created document details.
        """
        try:
            # Generate embedding
            embedding_vector = await self.embeddings.aembed_query(content)

            # Save to database
            with get_db() as db:
                db_document = DBDocument(
                    content=content,
                    doc_metadata=metadata,
                    embedding=embedding_vector
                )
                db.add(db_document)
                db.commit()
                db.refresh(db_document)

                return DocumentResponse(
                    id=db_document.id,
                    content=db_document.content,
                    metadata=db_document.doc_metadata
                )

        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            raise

    async def process_and_index_document(self, content: bytes, filename: str, content_type: str) -> List[DocumentResponse]:
        """
        Parse, chunk, and index a document file.

        Parameters
        ----------
        content : bytes
            The raw file content.
        filename : str
            The name of the file.
        content_type : str
            The MIME type of the file.

        Returns
        -------
        List[DocumentResponse]
            List of created document chunks.
        """
        try:
            # 1. Get parser and parse content
            parser = ParserFactory.get_parser(content_type)
            text = await parser.parse(content)

            if not text.strip():
                logger.warning(f"No text extracted from {filename}")
                return []

            # 2. Chunk content
            chunks = self.chunker.chunk(text)

            # 3. Index each chunk
            indexed_docs = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": filename,
                    "content_type": content_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                doc = await self.index_document(chunk, metadata)
                indexed_docs.append(doc)

            logger.info(f"Successfully indexed {len(indexed_docs)} chunks from {filename}")
            return indexed_docs

        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            raise

    async def process_and_index_url(self, url: str) -> List[DocumentResponse]:
        """
        Parse, chunk, and index content from a URL.

        Parameters
        ----------
        url : str
            The URL to process.

        Returns
        -------
        List[DocumentResponse]
            List of created document chunks.
        """
        try:
            # 1. Get parser (UrlParser) and parse content
            parser = ParserFactory.get_parser("url")
            text = await parser.parse(url)

            if not text.strip():
                logger.warning(f"No text extracted from {url}")
                return []

            # 2. Chunk content
            chunks = self.chunker.chunk(text)

            # 3. Index each chunk
            indexed_docs = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": url,
                    "content_type": "url",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                doc = await self.index_document(chunk, metadata)
                indexed_docs.append(doc)

            logger.info(f"Successfully indexed {len(indexed_docs)} chunks from {url}")
            return indexed_docs

        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            raise

    async def list_documents(self, limit: int = 100, offset: int = 0) -> List[DocumentResponse]:
        """
        List available documents.

        Parameters
        ----------
        limit : int
            Maximum number of documents to return.
        offset : int
            Number of documents to skip.

        Returns
        -------
        List[DocumentResponse]
            List of documents.
        """
        try:
            with get_db() as db:
                docs = db.query(DBDocument).order_by(DBDocument.id.desc()).limit(limit).offset(offset).all()

                return [
                    DocumentResponse(
                        id=doc.id,
                        content=doc.content,
                        metadata=doc.doc_metadata
                    ) for doc in docs
                ]
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise

    async def count_documents(self) -> int:
        """
        Count total number of documents.

        Returns
        -------
        int
            Total count of documents.
        """
        try:
            with get_db() as db:
                return db.query(DBDocument).count()
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            raise

    async def deindex_document(self, document_id: int) -> bool:
        """
        Delete a document by ID.

        Parameters
        ----------
        document_id : int
            The ID of the document to delete.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        try:
            with get_db() as db:
                doc = db.query(DBDocument).filter(DBDocument.id == document_id).first()
                if not doc:
                    return False

                db.delete(doc)
                db.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise


# Global service instance
rag_service = RagService()
