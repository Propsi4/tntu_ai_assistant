"""SQLAlchemy ORM models for chat history storage."""

from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, CheckConstraint, Index, func
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ChatSession(Base):
    """SQLAlchemy model for chat sessions table."""

    __tablename__ = "chat_sessions"

    session_id = Column(String(100), primary_key=True)
    title = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationship to messages
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """SQLAlchemy model for chat messages table."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationship to session
    session = relationship("ChatSession", back_populates="messages")

    # Add check constraint for role
    __table_args__ = (
        CheckConstraint("role IN ('user', 'tool', 'assistant')", name="check_role"),
        Index("idx_chat_messages_session_id", "session_id"),
        Index("idx_chat_messages_created_at", "created_at"),
    )
