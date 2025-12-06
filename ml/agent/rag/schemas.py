"""Pydantic schemas for RAG API request and response models."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class IndexDocumentRequest(BaseModel):
    """Request model to index a new document."""

    content: str = Field(..., min_length=1, description="The text content of the document to index")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata (title, source, etc.) associated with the document"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "The Faculty of Computer Information Systems and Software Engineering (FIS) was founded in 1991.",
                "metadata": {"source": "wiki", "title": "Faculty History"},
            }
        }
    )


class DocumentResponse(BaseModel):
    """Response model representing a stored document."""

    id: int = Field(..., description="Unique identifier of the document")
    content: str = Field(..., description="The text content of the document")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata associated with the document")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "content": "The Faculty of Computer Information Systems and Software Engineering (FIS) was founded in 1991.",
                "metadata": {"source": "wiki", "title": "Faculty History"},
            }
        }
    )


class DocumentListResponse(BaseModel):
    """Response model for a list of documents."""

    items: List[DocumentResponse] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents retrieved")
    limit: int = Field(..., description="Limit used for pagination")
    offset: int = Field(..., description="Offset used for pagination")
