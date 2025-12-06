"""API Client for communicating with the backend."""

import json
from typing import Optional, Dict, Any, List, AsyncGenerator

import httpx
from ui.config.settings import settings


class TNTUClient:
    """Client for the TNTU Assistant API."""

    def __init__(self):
        """Initialize the client."""
        self.base_url = settings.API_BASE_URL
        self.timeout = 60.0  # Increased timeout for long LLM responses

    async def chat_stream(
        self,
        message: str,
        session_id: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat response from the agent.

        Parameters
        ----------
        message : str
            The user's message.
        session_id : str
            The session ID.
        model_name : Optional[str]
            The LLM model name.
        temperature : Optional[float]
            The LLM temperature.

        Yields
        ------
        str
            Chunks of the response.
        """
        url = f"{self.base_url}/agent/chat/stream"
        payload = {
            "message": message,
            "session_id": session_id,
            "llm_params": {
                "model_name": model_name,
                "temperature": temperature
            } if model_name or temperature else None
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if not payload_str:
                        continue
                    try:
                        event = json.loads(payload_str)
                        if isinstance(event, dict):
                            yield event
                    except json.JSONDecodeError:
                        continue

    async def get_history(self, session_id: str) -> Dict[str, Any]:
        """
        Get chat history for a session.

        Parameters
        ----------
        session_id : str
            The session ID.

        Returns
        -------
        Dict[str, Any]
            The chat history.
        """
        url = f"{self.base_url}/chat-history/messages/{session_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    async def delete_history(self, session_id: str) -> Dict[str, Any]:
        """
        Delete chat history for a session.

        Parameters
        ----------
        session_id : str
            The session ID.

        Returns
        -------
        Dict[str, Any]
            Response from the API.
        """
        url = f"{self.base_url}/chat-history/delete/{session_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url)
            response.raise_for_status()
            return response.json()

    async def list_documents(
        self, limit: int = 10, offset: int = 0
    ) -> Dict[str, Any]:
        """
        List RAG documents.

        Parameters
        ----------
        limit : int
            Number of documents to return.
        offset : int
            Number of documents to skip.

        Returns
        -------
        Dict[str, Any]
            List of documents and metadata.
        """
        url = f"{self.base_url}/rag/documents"
        params = {"limit": limit, "offset": offset}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: str
    ) -> List[Dict[str, Any]]:
        """
        Upload a document for indexing.

        Parameters
        ----------
        file_content : bytes
            The file content.
        filename : str
            The name of the file.
        content_type : str
            The MIME type of the file.

        Returns
        -------
        List[Dict[str, Any]]
            The indexed documents.
        """
        url = f"{self.base_url}/rag/upload"
        files = {"file": (filename, file_content, content_type)}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, files=files)
            response.raise_for_status()
            return response.json()

    async def index_url(self, url_to_index: str) -> List[Dict[str, Any]]:
        """
        Index content from a URL.

        Parameters
        ----------
        url_to_index : str
            The URL to index.

        Returns
        -------
        List[Dict[str, Any]]
            The indexed documents.
        """
        url = f"{self.base_url}/rag/index-url"
        # Pass as raw body string since the endpoint expects Body(embed=True)
        payload = {"url": url_to_index}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    async def delete_document(self, document_id: int) -> Dict[str, Any]:
        """
        Delete a document by ID.

        Parameters
        ----------
        document_id : int
            The document ID.

        Returns
        -------
        Dict[str, Any]
            Response from the API.
        """
        url = f"{self.base_url}/rag/document/{document_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url)
            response.raise_for_status()
            return response.json()

    async def check_health(self) -> bool:
        """
        Check if the API is healthy.

        Returns
        -------
        bool
            True if healthy, False otherwise.
        """
        # Need to handle global health check URL potentially differing from
        # base URL. Assuming base is /api/v1, root health is at /health
        # relative to host. Construct health URL based on assumption or
        # explicit path.

        # Let's try the agent health check which is definitely under /api/v1
        url = f"{self.base_url}/agent/health"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        except Exception:
            return False

    async def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all conversation sessions.

        Returns
        -------
        List[Dict[str, Any]]
            List of conversations with metadata.
        """
        url = f"{self.base_url}/chat-history/conversations"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
