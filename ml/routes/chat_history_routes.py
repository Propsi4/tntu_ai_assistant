"""FastAPI Router for chat history management endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from ml.agent.conversation_history.manager import ConversationHistoryManager
from ml.agent.conversation_history.utils import langchain_messages_to_dicts
from ml.config.logger import get_logger

logger = get_logger(__name__)

# Create the router
router = APIRouter(prefix="/api/v1/chat-history", tags=["chat-history"])

# Initialize the conversation history manager
history_manager = ConversationHistoryManager()


class ConversationSummary(BaseModel):
    """Summary info for a conversation session."""

    session_id: str
    title: str
    created_at: Optional[str]
    updated_at: Optional[str]
    message_count: int


@router.get("/messages/{session_id}")
async def get_messages_for_session(session_id: str) -> JSONResponse:
    """
    Get all messages for a specific session.

    Parameters
    ----------
    session_id : str
        The session identifier.

    Returns
    -------
    JSONResponse
        JSONResponse with messages for the session.
    """
    try:
        logger.info(f"Getting messages for session: {session_id}")

        # Get messages for the session (returns LangChain messages)
        messages_langchain = await history_manager.get_messages_for_session(session_id)
        # Convert LangChain messages to dicts for JSON serialization
        messages_dicts = langchain_messages_to_dicts(messages_langchain)

        response_data = {"session_id": session_id, "messages": messages_dicts, "total_messages": len(messages_dicts)}

        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        logger.error(f"Error getting messages for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve messages: {str(e)}")


@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations() -> JSONResponse:
    """
    List all conversation sessions with basic metadata.

    Returns
    -------
    JSONResponse
        List of conversations.
    """
    try:
        conversations = await history_manager.get_all_sessions()
        return JSONResponse(content=conversations, status_code=200)
    except Exception as e:
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")


@router.delete("/delete/{session_id}")
async def delete_session_history(session_id: str) -> JSONResponse:
    """
    Delete all messages for a specific session.

    Parameters
    ----------
    session_id : str
        The session identifier.

    Returns
    -------
    JSONResponse
        JSONResponse with success status.
    """
    try:
        logger.info(f"Deleting history for session: {session_id}")

        # Delete session history
        success = await history_manager.delete_chat_by_session_id(session_id)

        if success:
            return JSONResponse(content={"status": "success", "message": f"History cleared for session {session_id}"}, status_code=200)
        else:
            return JSONResponse(content={"status": "error", "message": f"Failed to clear history for session {session_id}"}, status_code=500)

    except Exception as e:
        logger.error(f"Error deleting history for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete history: {str(e)}")
