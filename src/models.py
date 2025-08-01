from typing import Optional, List, Union, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = Field(default=False)


class GenerationResponse(BaseModel):
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = Field(default=False)


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
