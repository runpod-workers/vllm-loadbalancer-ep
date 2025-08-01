from typing import List
from transformers import AutoTokenizer
from .models import ChatMessage, ErrorResponse


def get_tokenizer(model_name: str):
    """Get tokenizer for the given model"""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def format_chat_prompt(messages: List[ChatMessage], model_name: str) -> str:
    """Format messages using the model's chat template"""
    tokenizer = get_tokenizer(model_name)

    # Use model's built-in chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        return tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True
        )

    # Fallback to common format
    formatted_prompt = ""
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"System: {message.content}\n\n"
        elif message.role == "user":
            formatted_prompt += f"Human: {message.content}\n\n"
        elif message.role == "assistant":
            formatted_prompt += f"Assistant: {message.content}\n\n"

    formatted_prompt += "Assistant: "
    return formatted_prompt


def create_error_response(error: str, detail: str, request_id: str = None) -> ErrorResponse:
    return ErrorResponse(error=error, detail=detail, request_id=request_id)