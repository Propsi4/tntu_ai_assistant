"""ConversationHistoryManager for managing chat history operations.

This module provides a SQLAlchemy ORM-based chat history manager
that works with LangChain message types.
"""

from typing import List, Dict
from sqlalchemy import and_, select

from langchain_core.messages import BaseMessage, HumanMessage

from ml.config.logger import get_logger
from .schema import get_db, ensure_schema_exists
from .models import ChatSession, ChatMessage
from .utils import langchain_to_dict, dicts_to_langchain_messages

logger = get_logger(__name__)


class ConversationHistoryManager:
    """Manager for conversation history operations using SQLAlchemy ORM."""

    def __init__(self):
        """Initialize the conversation history manager."""
        self._schema_ensured = False

    async def _ensure_schema(self) -> None:
        """Ensure schema exists (lazy initialization)."""
        if not self._schema_ensured:
            await ensure_schema_exists()
            self._schema_ensured = True

    async def get_messages_for_session(self, session_id: str) -> List[BaseMessage]:
        """
        Retrieve all messages for a provided session as LangChain messages.

        Parameters
        ----------
        session_id : str
            The session ID to retrieve messages for.

        Returns
        -------
        List[BaseMessage]
            List of LangChain message objects (HumanMessage, AIMessage, ToolMessage).

        Examples
        --------
        >>> manager = ConversationHistoryManager()
        >>> messages = await manager.get_messages_for_session("session123")
        >>> # Returns: [HumanMessage(content="Hello"), AIMessage(content="Hi"), ...]
        """
        await self._ensure_schema()
        try:
            logger.info(f"Retrieving messages for session: {session_id}")

            with get_db() as db:
                messages_query = (
                    db.query(ChatMessage)
                    .filter(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at.asc())
                )
                messages = messages_query.all()

                # Convert to dictionaries then to LangChain messages
                message_dicts = [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ]
                langchain_messages = dicts_to_langchain_messages(message_dicts)

                logger.info(f"Retrieved {len(langchain_messages)} messages for session: {session_id}")
                return langchain_messages

        except Exception as e:
            logger.error(f"Error retrieving messages for session {session_id}: {e}")
            raise

    async def get_all_sessions(self) -> List[Dict[str, str]]:
        """
        Retrieve all chat sessions with basic metadata.

        Returns
        -------
        List[Dict[str, str]]
            List of dictionaries with session info.
        """
        await self._ensure_schema()
        try:
            with get_db() as db:
                sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
                result: List[Dict[str, str]] = []

                for session in sessions:
                    message_count = (
                        db.query(ChatMessage)
                        .filter(ChatMessage.session_id == session.session_id)
                        .count()
                    )
                    result.append(
                        {
                            "session_id": session.session_id,
                            "title": session.title or "Untitled",
                            "created_at": session.created_at.isoformat() if session.created_at else None,
                            "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                            "message_count": message_count,
                        }
                    )

                return result
        except Exception as e:
            logger.error(f"Error retrieving sessions: {e}")
            raise

    async def save_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        """
        Save a list of LangChain messages for a session.

        Parameters
        ----------
        session_id : str
            The session ID to save messages for.
        messages : List[BaseMessage]
            List of LangChain message objects to save.

        Examples
        --------
        >>> manager = ConversationHistoryManager()
        >>> await manager.save_messages(
        ...     "session123",
        ...     [HumanMessage(content="Hello"), AIMessage(content="Hi")]
        ... )
        """
        await self._ensure_schema()
        try:
            logger.info(f"Saving {len(messages)} messages for session: {session_id}")

            with get_db() as db:
                try:
                    # Ensure session exists (create if not exists)
                    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()

                    # Get first user message for title if session doesn't exist
                    first_user_msg = next(
                        (msg for msg in messages if isinstance(msg, HumanMessage)),
                        None
                    )
                    title = first_user_msg.content[:100] if first_user_msg else None

                    if not session:
                        session = ChatSession(
                            session_id=session_id,
                            title=title,
                        )
                        db.add(session)
                    else:
                        # Update timestamp is handled automatically by SQLAlchemy's onupdate=func.now()
                        # Update title if provided, otherwise touch it to trigger onupdate
                        if title:
                            session.title = title
                        elif session.title is None:
                            session.title = ""  # Set empty string if None to trigger update
                        # The onupdate=func.now() will automatically update updated_at when session changes

                    # Save all messages
                    for msg in messages:
                        msg_dict = langchain_to_dict(msg)
                        chat_msg = ChatMessage(
                            session_id=session_id,
                            role=msg_dict["role"],
                            content=msg_dict["content"],
                        )
                        db.add(chat_msg)

                    db.commit()
                    logger.info(f"Successfully saved {len(messages)} messages for session: {session_id}")
                except Exception:
                    db.rollback()
                    raise

        except Exception as e:
            logger.error(f"Error saving messages for session {session_id}: {e}")
            raise

    async def delete_chat_by_session_id(self, session_id: str) -> bool:
        """
        Delete all messages for a session.

        Parameters
        ----------
        session_id : str
            The session ID to delete.

        Returns
        -------
        bool
            True if deletion was successful, False otherwise.

        Examples
        --------
        >>> manager = ConversationHistoryManager()
        >>> success = await manager.delete_chat_by_session_id("session123")
        """
        await self._ensure_schema()
        try:
            logger.info(f"Deleting chat for session: {session_id}")

            with get_db() as db:
                try:
                    # Delete session (cascade will handle messages due to relationship)
                    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
                    if session:
                        db.delete(session)
                        db.commit()
                        logger.info(f"Successfully deleted chat for session: {session_id}")
                        return True
                    return False
                except Exception:
                    db.rollback()
                    raise

        except Exception as e:
            logger.error(f"Error deleting chat for session {session_id}: {e}")
            return False

    async def get_all_chat_titles(self) -> List[Dict[str, str]]:
        """
        Retrieve chat titles (first user message) for all sessions.

        Returns
        -------
        List[Dict[str, str]]
            List of dictionaries with 'session_id' and 'title' keys.
            Format: [{"session_id": "...", "title": "..."}, ...]

        Examples
        --------
        >>> manager = ConversationHistoryManager()
        >>> titles = await manager.get_all_chat_titles()
        >>> # Returns: [{"session_id": "session123", "title": "Hello"}, ...]
        """
        await self._ensure_schema()
        try:
            logger.info("Retrieving all chat titles")

            with get_db() as db:
                # Query all sessions with their first user message
                sessions = db.query(ChatSession).all()
                chat_titles = []

                for session in sessions:
                    # Get first user message for this session
                    first_user_msg = (
                        db.query(ChatMessage)
                        .filter(
                            and_(
                                ChatMessage.session_id == session.session_id,
                                ChatMessage.role == "user",
                            )
                        )
                        .order_by(ChatMessage.created_at.asc())
                        .first()
                    )

                    title = first_user_msg.content if first_user_msg else session.title
                    chat_titles.append(
                        {
                            "session_id": session.session_id,
                            "title": title or "Untitled",
                        }
                    )

                logger.info(f"Retrieved {len(chat_titles)} chat titles")
                return chat_titles

        except Exception as e:
            logger.error(f"Error retrieving chat titles: {e}")
            raise

    async def check_health(self) -> bool:
        """
        Check if the database connection is healthy.

        Returns
        -------
        bool
            True if database connection is healthy, False otherwise.

        Examples
        --------
        >>> manager = ConversationHistoryManager()
        >>> is_healthy = await manager.check_health()
        """
        await self._ensure_schema()
        try:
            with get_db() as db:
                # Simple query to test connection
                db.execute(select(1))
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
