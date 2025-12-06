"""Pydantic schemas for API request and response models."""

from typing import Optional, Literal, Union, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
from ml.config.settings import settings


class ChatMessage(BaseModel):
    """Single chat message."""

    role: str = Field(..., description="Role of the message sender (user/assistant/system)")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp of the message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"role": "user", "content": "What is the weather in Kyiv?", "timestamp": "2025-10-22T10:30:00"}
        }
    )


class TokenUsage(BaseModel):
    """Token usage information for API calls."""

    input_tokens: Optional[int] = Field(default=None, description="Number of input tokens used (prompt)")
    output_tokens: Optional[int] = Field(default=None, description="Number of output tokens used (completion)")

    model_config = ConfigDict(json_schema_extra={"example": {"input_tokens": 250, "output_tokens": 150}})


class LLMParams(BaseModel):
    """LLM parameters for customizing model behavior."""

    model_name: Optional[str] = Field(
        default=settings.DEFAULT_LLM, description="Name of the LLM model to use (defaults to DEFAULT_LLM from settings)"
    )
    temperature: Optional[float] = Field(
        default=settings.DEFAULT_LLM_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation (defaults to DEFAULT_LLM_TEMPERATURE from settings)",
    )

    model_config = ConfigDict(json_schema_extra={"example": {"model_name": "gpt-5.1", "temperature": 1.0}})


class AgentRequest(BaseModel):
    """Request model for the AI agent."""

    message: str = Field(..., min_length=1, max_length=2000, description="User's question or message to the agent")

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Session identifier for conversation tracking. Used as thread_id for persistence.",
    )

    llm_params: Optional[LLMParams] = Field(
        default=None, description="Optional LLM parameters to customize model behavior"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Tell me about faculty of computer science",
                "session_id": "session456",
                "llm_params": {"model_name": "gpt-5.1", "temperature": 1.0},
            }
        }
    )


class AgentResponse(BaseModel):
    """Response model from the AI agent."""

    response: str = Field(..., description="Agent's response to the user query")
    status: Literal["success", "error"] = Field(default="success", description="Status of the request (success/error)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the response")
    session_id: Optional[str] = Field(default=None, description="Session identifier if provided in request")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "The faculty of computer science is a department of the university that teaches computer science and related subjects.",
                "status": "success",
                "timestamp": "2025-10-22T10:30:05",
                "session_id": "session456",
            }
        }
    )


class StreamingStep(BaseModel):
    """Intermediate step in agent processing."""

    step_type: str = Field(..., description="Type of step (tool_call, thinking, etc.)")
    content: Any = Field(..., description="Content of the step")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the step")


class StreamingToken(BaseModel):
    """Single token in streaming response."""

    token: str = Field(..., description="The token content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the token")


class StreamingResponse(BaseModel):
    """Unified streaming response with Literal type and unified data field."""

    type: Literal["step", "token", "complete", "error"] = Field(..., description="Type of streaming event")
    status: Literal["success", "error"] = Field(default="success", description="Status of the request (success/error)")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    data: Union[StreamingStep, StreamingToken, dict] = Field(
        ..., description="Event data (step, token, or dict for complete/error)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "step",
                "status": "success",
                "session_id": "session456",
                "data": {
                    "step_type": "initialization",
                    "content": "Initializing agent with model: gpt-4o",
                    "timestamp": "2025-10-22T10:30:01",
                },
            }
        }
    )
