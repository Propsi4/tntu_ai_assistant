# TNTU Assistant

A comprehensive AI-powered assistant for handling university-related queries (or drug assistant as per current logic), featuring a FastAPI backend and a Streamlit UI. This project leverages RAG (Retrieval-Augmented Generation) and agentic workflows to provide accurate responses.

## Project Overview

TNTU Assistant is designed to help users interact with a knowledge base through a chat interface. It consists of two main components:
- **ML Backend (FastAPI):** Handles the core logic, including agent orchestration, RAG pipelines, and database interactions.
- **UI Frontend (Streamlit):** Provides a user-friendly chat interface, history management, and document upload capabilities for the knowledge base.

## Key Features

- **Chat Interface:** Interactive chat with history persistence.
- **RAG (Retrieval-Augmented Generation):** Upload documents (PDF, DOCX, etc.) or index URLs to enhance the assistant's knowledge.
- **Agentic Workflow:** Uses tools to search the web or query the vector database for answers.
- **Session Management:** Create, switch between, and delete chat sessions.
- **Dockerized:** Fully containerized for easy deployment.

## Tools Available

The agent is equipped with the following tools:
- **Tavily Search:** Performs web searches for real-time information.
- **Vector Search:** Queries the local vector database for indexed knowledge.
- **Web Fetch:** Retrieves content from specific URLs.

## Prerequisites

Before running the application, ensure you have the following installed:
- **Docker** and **Docker Compose**
- **Python 3.11+**
- **PostgreSQL 12+** with the **pgvector** extension installed.
- **Poetry** (for dependency management)
- **OpenAI API Key**
- **Tavily API Key** (for web search)

> **Important:** This project requires a PostgreSQL database with the `pgvector` extension enabled for vector similarity search. You must set up your own PostgreSQL instance and install the extension manually.
>
> Follow the official installation guide here: [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
>
> Once installed, enable the extension in your database by running:
> ```sql
> CREATE EXTENSION vector;
> ```


## Environment Variables

The project uses environment variables to configure both the ML backend and the UI. You can set these in a `.env` file (see `example.env`) or pass them via Docker.

### ML Backend Configuration (`ml/config/settings.py`)

| Env Name | Description | Datatype | Optional |
| :--- | :--- | :--- | :--- |
| `TAVILY_API_KEY` | API key for Tavily search functionality. | `str` | **No** |
| `OPENAI_API_KEY` | API key for OpenAI language models. | `str` | **No** |
| `POSTGRES_HOST` | Database host address. | `str` | **No** |
| `POSTGRES_DATABASE` | Database name. | `str` | **No** |
| `POSTGRES_USER` | Database username. | `str` | **No** |
| `POSTGRES_PASSWORD` | Database password. | `str` | **No** |
| `POSTGRES_PORT` | Database port number. | `int` | Yes (default: `5432`) |
| `DEBUG` | Enable debug mode logging. | `bool` | Yes (default: `False`) |
| `OPENAI_EMBEDDING_MODEL` | Embedding model for vector search. | `str` | Yes (default: `text-embedding-3-small`) |
| `DEFAULT_LLM` | Default LLM model to use. | `str` | Yes (default: `gpt-5.1`) |
| `DEFAULT_LLM_TEMPERATURE` | Temperature for LLM responses (0.0 - 2.0). | `float` | Yes (default: `1.0`) |
| `AGENT_MAX_TOKENS` | Max tokens for agent responses. | `int` | Yes (default: `16000`) |
| `DOCUMENT_MAX_TOKENS` | Max tokens for document chunks. | `int` | Yes (default: `1000`) |
| `APP_NAME` | Name of the application. | `str` | Yes (default: `TNTU Assistant AI`) |
| `APP_VERSION` | Version of the application. | `str` | Yes (default: `1.0.0`) |
| `POSTGRES_SSL_MODE` | SSL mode for DB connection. | `str` | Yes (default: `prefer`) |
| `POSTGRES_SCHEMA` | Database schema name. | `str` | Yes (default: `public`) |

### UI Frontend Configuration (`ui/config/settings.py`)

| Env Name | Description | Datatype | Optional |
| :--- | :--- | :--- | :--- |
| `API_BASE_URL` | Base URL of the backend API. | `str` | Yes (default: `http://tntuassistant-ml:8000/api/v1`) |
| `PAGE_TITLE` | Title displayed in the browser tab. | `str` | Yes (default: `TNTU Assistant`) |
| `PAGE_ICON` | Icon displayed in the browser tab/sidebar. | `str` | Yes (default: `🎓`) |
| `LAYOUT` | Streamlit page layout (`wide` or `centered`). | `str` | Yes (default: `wide`) |
| `DEFAULT_LLM` | Default LLM model for the UI. | `str` | Yes (default: `gpt-5.1`) |
| `DEFAULT_LLM_TEMPERATURE` | Default LLM temperature for the UI. | `float` | Yes (default: `1.0`) |

## How to Start the App

1. **Clone the repository.**
2. **Set up environment variables:**
   - Copy `example.env` to `.env` and fill in your keys.
3. **Run with Docker Compose:**

```bash
docker compose up --build
```

This will start:
- **ML Backend** on port `8000`
- **Streamlit UI** on port `8500`
- **Postgres DB** on port `5432`

## Usage

### Web Interface
Access the Chat UI at:
[http://localhost:8500/](http://localhost:8500/)

### API Documentation
Explore the backend API via Swagger UI:
[http://localhost:8000/docs](http://localhost:8000/docs)

### API Endpoints

The API is organized into three main namespaces:
```
api/v1/
     |
     ├── agent/          (Chat & Health)
     ├── chat-history/   (Session Management)
     └── rag/            (Document Indexing)
```

#### Detailed Endpoint Reference

| Endpoint Title | Endpoint Path | Description | Request Type | Inputs Structure | Outputs Structure |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Agent Health** | `/api/v1/agent/health` | Check if the agent service is healthy. | `GET` | None | `{"status": "healthy", "agent_initialized": bool}` |
| **Agent Chat** | `/api/v1/agent/chat` | Send a message to the agent (non-streaming). | `POST` | `{"message": "str", "session_id": "str", "llm_params": {...}}` | `{"response": "str", "status": "success", "session_id": "str"}` |
| **Agent Stream** | `/api/v1/agent/chat/stream` | Send a message and get a streaming response (SSE). | `POST` | `{"message": "str", "session_id": "str", "llm_params": {...}}` | Stream of `{"type": "token/complete", "data": {...}}` |
| **Get Messages** | `/api/v1/chat-history/messages/{session_id}` | Retrieve full chat history for a session. | `GET` | Path param: `session_id` | `{"session_id": "str", "messages": [{"role": "user", "content": "..."}], "total_messages": int}` |
| **List Conversations** | `/api/v1/chat-history/conversations` | List all stored conversation sessions. | `GET` | None | List of `{"session_id": "str", "title": "str", "message_count": int}` |
| **Delete Session** | `/api/v1/chat-history/delete/{session_id}` | Delete a specific conversation history. | `DELETE` | Path param: `session_id` | `{"status": "success", "message": "History cleared..."}` |
| **Index Document** | `/api/v1/rag/index` | Index a text snippet manually. | `POST` | `{"content": "str", "metadata": {}}` | `{"id": int, "content": "str", "metadata": {}}` |
| **Upload Document** | `/api/v1/rag/upload` | Upload and index a file (PDF, DOCX, TXT). | `POST` | Multipart Form: `file` | List of `DocumentResponse` objects |
| **Index URL** | `/api/v1/rag/index-url` | Scrape and index content from a URL. | `POST` | `{"url": "https://..."}` | List of `DocumentResponse` objects |
| **List Documents** | `/api/v1/rag/documents` | List indexed documents with pagination. | `GET` | Query params: `limit`, `offset` | `{"items": [...], "total": int}` |
| **Delete Document** | `/api/v1/rag/document/{document_id}` | Remove a document from the index. | `DELETE` | Path param: `document_id` | `{"status": "success", "message": "Document deleted"}` |
