"""FastAPI Router for agent endpoints."""

import logging
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from langchain_core.messages import HumanMessage, AIMessage
import dspy

from ml.agent.agent import agent, configure_dspy_for_request
from ml.agent.schemas import AgentRequest, AgentResponse, StreamingResponse as StreamingResponseModel
from ml.agent.exceptions import InvalidModelException
from ml.agent.conversation_history.manager import ConversationHistoryManager
from ml.agent.conversation_history.utils import (
    langchain_messages_to_dicts,
    dict_to_dspy_format,
    dspy_prediction_to_response,
)
from ml.agent.utils import tool_messages_from_flat_json

logger = logging.getLogger(__name__)

# Create the router
router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

# Initialize the conversation history manager
history_manager = ConversationHistoryManager()


@router.get("/health")
async def agent_health() -> JSONResponse:
    """
    Health check endpoint for the agent.

    Returns
    -------
    JSONResponse
        JSONResponse with agent health status.
    """
    try:
        is_healthy = agent.is_healthy()

        response_data = {"status": "healthy" if is_healthy else "unhealthy", "agent_initialized": is_healthy}

        status_code = 200 if is_healthy else 503

        return JSONResponse(content=response_data, status_code=status_code)
    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


@router.post("/chat", response_model=AgentResponse)
async def agent_chat(request: AgentRequest) -> AgentResponse:
    """
    Non-streaming chat endpoint for agent interaction.

    Parameters
    ----------
    request : AgentRequest
        The chat request containing message and session_id.

    Returns
    -------
    AgentResponse
        AgentResponse with agent's reply.
    """
    try:
        # Validate that agent is healthy
        if not agent.is_healthy():
            raise HTTPException(status_code=503, detail="Agent service is not available")

        # Extract request parameters
        user_message = request.message
        session_id = request.session_id
        llm_params = request.llm_params

        # Load chat history from database (returns LangChain messages)
        chat_history_langchain = await history_manager.get_messages_for_session(session_id)
        # Convert LangChain messages to dicts, then to DSPy format
        chat_history_dicts = langchain_messages_to_dicts(chat_history_langchain)
        chat_history = dict_to_dspy_format(chat_history_dicts)

        # Configure DSPy with request-level model/temperature if provided
        model_name = llm_params.model_name if llm_params else None
        temperature = llm_params.temperature if llm_params else None

        # Create context manager for DSPy configuration
        dspy_context = configure_dspy_for_request(model_name=model_name, temperature=temperature)

        # Execute agent within the DSPy context
        with dspy_context:
            # Call agent's forward method
            prediction = agent(user_message=user_message, chat_history=chat_history)

        # Extract tool calls from DSPy's execution history (only new ones added during this call)
        # Prepare all messages to save: user message, tool calls, and assistant response
        messages_to_save = [HumanMessage(content=user_message)]
        messages_to_save.extend(tool_messages_from_flat_json(prediction.trajectory))  # Add tool calls
        messages_to_save.append(AIMessage(content=prediction.response))  # Add assistant response

        # Save messages to database using LangChain message types
        await history_manager.save_messages(
            session_id=session_id,
            messages=messages_to_save,
        )

        # Convert prediction to response schema
        response = dspy_prediction_to_response(prediction, session_id)

        return response

    except InvalidModelException as e:
        logger.warning(f"Invalid model exception: {e}")
        raise HTTPException(
            status_code=400,
            detail={"error": str(e), "invalid_model": e.model_name, "allowed_models": e.allowed_models}
        )

    except Exception as e:
        logger.error(f"Error in agent chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def agent_chat_stream(request: AgentRequest) -> StreamingResponse:
    """
    Streaming chat endpoint for agent interaction using FastAPI StreamingResponse.

    Parameters
    ----------
    request : AgentRequest
        The chat request containing message and session_id.

    Returns
    -------
    StreamingResponse
        StreamingResponse with Server-Sent Events (SSE) stream.
    """
    try:
        # Validate that agent is healthy
        if not agent.is_healthy():
            raise HTTPException(status_code=503, detail="Agent service is not available")

        # Extract request parameters
        user_message = request.message
        session_id = request.session_id
        llm_params = request.llm_params

        # Load chat history from database (returns LangChain messages)
        chat_history_langchain = await history_manager.get_messages_for_session(session_id)
        # Convert LangChain messages to dicts, then to DSPy format
        chat_history_dicts = langchain_messages_to_dicts(chat_history_langchain)
        chat_history = dict_to_dspy_format(chat_history_dicts)

        # Configure DSPy with request-level model/temperature if provided
        model_name = llm_params.model_name if llm_params else None
        temperature = llm_params.temperature if llm_params else None

        # Create context manager for DSPy configuration
        dspy_context = configure_dspy_for_request(model_name=model_name, temperature=temperature)

        async def generate_stream():
            """Generate streaming responses for the agent."""
            try:
                # Execute agent within the DSPy context
                with dspy_context:
                    # Create streamified agent
                    stream_listeners = [
                        dspy.streaming.StreamListener(signature_field_name="response")
                    ]
                    stream_agent = dspy.streamify(agent, stream_listeners=stream_listeners)

                    ai_response_chunks = []
                    final_prediction = None

                    # Call the streamified agent
                    output_stream = stream_agent(user_message=user_message, chat_history=chat_history)

                    async for chunk in output_stream:
                        if isinstance(chunk, dspy.streaming.StreamResponse):
                            # Stream token chunks
                            token = chunk.chunk
                            ai_response_chunks.append(token)

                            # Format as SSE
                            stream_response = StreamingResponseModel(
                                type="token",
                                status="success",
                                session_id=session_id,
                                data={"token": token}
                            )
                            json_data = json.dumps(stream_response.model_dump(mode='json'))
                            yield f"data: {json_data}\n\n"

                        elif isinstance(chunk, dspy.Prediction):
                            # Final prediction received
                            final_prediction = chunk

                            # Send completion event
                            complete_response = StreamingResponseModel(
                                type="complete",
                                status="success",
                                session_id=session_id,
                                data={"response": chunk.response}
                            )
                            json_data = json.dumps(complete_response.model_dump(mode='json'))
                            yield f"data: {json_data}\n\n"

                    if final_prediction:
                        ai_message = final_prediction.response

                        # Prepare all messages to save: user message, tool calls, and assistant response
                        messages_to_save = [HumanMessage(content=user_message)]
                        messages_to_save.extend(tool_messages_from_flat_json(final_prediction.trajectory))  # Add tool calls
                        messages_to_save.append(AIMessage(content=ai_message))  # Add assistant response

                        await history_manager.save_messages(
                            session_id=session_id,
                            messages=messages_to_save,
                        )

            except InvalidModelException as e:
                logger.error(f"Invalid model exception in streaming: {e}")
                error_response = StreamingResponseModel(
                    type="error",
                    status="error",
                    session_id=session_id,
                    data={
                        "error": f"Invalid model: {e.model_name}. Allowed models: {', '.join(e.allowed_models)}",
                        "model_name": e.model_name,
                        "allowed_models": e.allowed_models,
                    },
                )
                yield f"data: {json.dumps(error_response.model_dump(mode='json'))}\n\n"
            except Exception as e:
                logger.error(f"Error in streaming: {e}", exc_info=True)
                error_response = StreamingResponseModel(
                    type="error",
                    status="error",
                    session_id=session_id,
                    data={"error": f"An error occurred while streaming: {str(e)}"},
                )
                yield f"data: {json.dumps(error_response.model_dump(mode='json'))}\n\n"

        # Return FastAPI StreamingResponse
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in agent chat stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
