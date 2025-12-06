"""FastAPI Router for RAG management endpoints."""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Body
from fastapi.responses import JSONResponse
from typing import List

from ml.agent.rag.schemas import IndexDocumentRequest, DocumentResponse, DocumentListResponse
from ml.agent.rag.service import rag_service
from ml.config.logger import get_logger

logger = get_logger(__name__)

# Create the router
router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post("/index", response_model=DocumentResponse)
async def index_document(request: IndexDocumentRequest) -> JSONResponse:
    """
    Index a new document for RAG.

    Parameters
    ----------
    request : IndexDocumentRequest
        The document content and metadata to index.

    Returns
    -------
    JSONResponse
        The indexed document details.
    """
    try:
        document = await rag_service.index_document(request.content, request.metadata)
        return JSONResponse(content=document.model_dump(), status_code=201)
    except Exception as e:
        logger.error(f"Error indexing document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")


@router.post("/upload", response_model=List[DocumentResponse])
async def upload_document(
    file: UploadFile = File(...),
) -> List[DocumentResponse]:
    """
    Upload and index a document (PDF, DOCX, PPTX, TXT).

    Parameters
    ----------
    file : UploadFile
        The file to upload and index.

    Returns
    -------
    List[DocumentResponse]
        List of created document chunks.
    """
    try:
        content = await file.read()
        documents = await rag_service.process_and_index_document(
            content=content,
            filename=file.filename,
            content_type=file.content_type
        )
        return documents
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")


@router.post("/index-url", response_model=List[DocumentResponse])
async def index_url(
    url: str = Body(..., embed=True),
) -> List[DocumentResponse]:
    """
    Scrape and index content from a URL.

    Parameters
    ----------
    url : str
        The URL to scrape and index.

    Returns
    -------
    List[DocumentResponse]
        List of created document chunks.
    """
    try:
        documents = await rag_service.process_and_index_url(url)
        return documents
    except Exception as e:
        logger.error(f"Error indexing URL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to index URL: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=10, ge=1, le=100, description="Number of documents to return"),
    offset: int = Query(default=0, ge=0, description="Number of documents to skip")
) -> DocumentListResponse:
    """
    List available documents in the knowledge base.

    Parameters
    ----------
    limit : int
        Maximum number of documents to return (default: 10).
    offset : int
        Number of documents to skip (default: 0).

    Returns
    -------
    DocumentListResponse
        List of documents and total count.
    """
    try:
        items = await rag_service.list_documents(limit=limit, offset=offset)
        total = await rag_service.count_documents()

        return DocumentListResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.delete("/document/{document_id}")
async def delete_document(document_id: int) -> JSONResponse:
    """
    Delete a document from the knowledge base.

    Parameters
    ----------
    document_id : int
        The ID of the document to delete.

    Returns
    -------
    JSONResponse
        Success status.
    """
    try:
        success = await rag_service.deindex_document(document_id)

        if success:
            return JSONResponse(content={"status": "success", "message": f"Document {document_id} deleted"}, status_code=200)
        else:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
