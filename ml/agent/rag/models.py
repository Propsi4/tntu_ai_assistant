"""SQLAlchemy ORM models for RAG storage."""

from sqlalchemy import Column, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from ml.agent.conversation_history.models import Base


class Document(Base):
    """SQLAlchemy model for documents table."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    doc_metadata = Column("metadata", JSONB, nullable=True)
    embedding = Column(Vector(1536))
