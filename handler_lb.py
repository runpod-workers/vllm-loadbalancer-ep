"""
Load balancer handler for vLLM.

Runs a FastAPI/uvicorn HTTP server instead of the RunPod serverless SDK.
RunPod's load balancer polls /ping to discover and route to healthy workers:
  - 204: initializing (do not route traffic here yet)
  - 200: ready (include in load balancer pool)

Start with: python3 /src/handler_lb.py
"""
import json
import logging
import multiprocessing
import os
import sys
import traceback
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

_is_ready = False
_chat_engine = None
_completion_engine = None
_responses_engine = None
_messages_engine = None
_serving_models = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _is_ready, _chat_engine, _completion_engine, _responses_engine, _messages_engine, _serving_models

    try:
        from engine import vLLMEngine
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIServingModels
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
        from vllm.entrypoints.anthropic.serving import AnthropicServingMessages


        log.info("Initializing vLLM engine...")
        vllm_engine = vLLMEngine()
        engine_args = vllm_engine.engine_args
        llm = vllm_engine.llm

        served_model_name = (
            os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE")
            or engine_args.served_model_name
            or engine_args.model
        )

        _serving_models = OpenAIServingModels(
            engine_client=llm,
            base_model_paths=[BaseModelPath(name=served_model_name, model_path=engine_args.model)],
            lora_modules=None,
        )
        await _serving_models.init_static_loras()

        chat_template = None
        if vllm_engine.tokenizer and hasattr(vllm_engine.tokenizer, "tokenizer"):
            chat_template = vllm_engine.tokenizer.tokenizer.chat_template

        _chat_engine = OpenAIServingChat(
            engine_client=llm,
            models=_serving_models,
            response_role=os.getenv("OPENAI_RESPONSE_ROLE", "assistant"),
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            trust_request_chat_template=os.getenv("TRUST_REQUEST_CHAT_TEMPLATE", "false").lower() == "true",
            return_tokens_as_token_ids=os.getenv("RETURN_TOKENS_AS_TOKEN_IDS", "false").lower() == "true",
            reasoning_parser=os.getenv("REASONING_PARSER", "") or "",
            enable_auto_tools=os.getenv("ENABLE_AUTO_TOOL_CHOICE", "false").lower() == "true",
            exclude_tools_when_tool_choice_none=os.getenv("EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE", "false").lower() == "true",
            tool_parser=os.getenv("TOOL_CALL_PARSER", "") or None,
            enable_prompt_tokens_details=os.getenv("ENABLE_PROMPT_TOKENS_DETAILS", "false").lower() == "true",
            enable_force_include_usage=os.getenv("ENABLE_FORCE_INCLUDE_USAGE", "false").lower() == "true",
            enable_log_outputs=os.getenv("ENABLE_LOG_OUTPUTS", "false").lower() == "true",
            log_error_stack=os.getenv("LOG_ERROR_STACK", "false").lower() == "true",
        )

        _completion_engine = OpenAIServingCompletion(
            engine_client=llm,
            models=_serving_models,
            request_logger=None,
            return_tokens_as_token_ids=os.getenv("RETURN_TOKENS_AS_TOKEN_IDS", "false").lower() == "true",
            enable_prompt_tokens_details=os.getenv("ENABLE_PROMPT_TOKENS_DETAILS", "false").lower() == "true",
            enable_force_include_usage=os.getenv("ENABLE_FORCE_INCLUDE_USAGE", "false").lower() == "true",
            log_error_stack=os.getenv("LOG_ERROR_STACK", "false").lower() == "true",
        )

        _responses_engine = OpenAIServingResponses(
            engine_client=llm,
            models=_serving_models,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            return_tokens_as_token_ids=os.getenv("RETURN_TOKENS_AS_TOKEN_IDS", "false").lower() == "true",
            reasoning_parser=os.getenv("REASONING_PARSER", "") or "",
            enable_auto_tools=os.getenv("ENABLE_AUTO_TOOL_CHOICE", "false").lower() == "true",
            tool_parser=os.getenv("TOOL_CALL_PARSER", "") or None,
            tool_server=None,
            enable_prompt_tokens_details=os.getenv("ENABLE_PROMPT_TOKENS_DETAILS", "false").lower() == "true",
            enable_force_include_usage=os.getenv("ENABLE_FORCE_INCLUDE_USAGE", "false").lower() == "true",
            enable_log_outputs=os.getenv("ENABLE_LOG_OUTPUTS", "false").lower() == "true",
            log_error_stack=os.getenv("LOG_ERROR_STACK", "false").lower() == "true",
        )

        _messages_engine = AnthropicServingMessages(
            engine_client=llm,
            models=_serving_models,
            response_role=os.getenv("OPENAI_RESPONSE_ROLE", "assistant"),
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            return_tokens_as_token_ids=os.getenv("RETURN_TOKENS_AS_TOKEN_IDS", "false").lower() == "true",
            reasoning_parser=os.getenv("REASONING_PARSER", "") or "",
            enable_auto_tools=os.getenv("ENABLE_AUTO_TOOL_CHOICE", "false").lower() == "true",
            tool_parser=os.getenv("TOOL_CALL_PARSER", "") or None,
            enable_prompt_tokens_details=os.getenv("ENABLE_PROMPT_TOKENS_DETAILS", "false").lower() == "true",
            enable_force_include_usage=os.getenv("ENABLE_FORCE_INCLUDE_USAGE", "false").lower() == "true",
        )

        _is_ready = True
        log.info("vLLM load balancer worker ready")

    except Exception as e:
        log.error(f"Startup failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    yield  # serve requests


app = FastAPI(title="vLLM Load Balancer Worker", lifespan=lifespan)


@app.get("/ping")
async def ping():
    """
    Health check required by RunPod load balancer.
    Returns 204 while engine is loading or not fully initialized, 200 once ready.
    """
    if not _is_ready:
        log.debug("Health check: Engine not ready")
        return Response(status_code=204)

    # Validate all engine states to prevent routing to partially initialized workers
    if not all(engines is not None for engines in [
        _chat_engine,
        _completion_engine,
        _responses_engine,
        _messages_engine
    ]):
        log.debug("Health check: Engine(s) not fully initialized")
        return Response(status_code=204)

    if not _serving_models:
        log.debug("Health check: Serving models not initialized")
        return Response(status_code=204)

    log.debug("Health check: Engine ready")
    return Response(status_code=200)


@app.get("/v1/models")
async def list_models():
    models = await _serving_models.show_available_models()
    return JSONResponse(models.model_dump())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse

    body = await request.json()

    try:
        req = ChatCompletionRequest(**body)
    except Exception as e:
        log.warning(f"Chat completions validation error: {e}")
        return JSONResponse(
            {"error": {"message": str(e), "type": "invalid_request_error"}},
            status_code=422,
        )

    response = await _chat_engine.create_chat_completion(req, raw_request=request)

    if isinstance(response, ErrorResponse):
        log.error(f"Chat completions engine error: {response.error.message} (code: {response.error.code})")
        return JSONResponse(response.model_dump(), status_code=response.error.code)

    if not body.get("stream"):
        return JSONResponse(response.model_dump())

    async def event_stream():
        async for chunk in response:
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse

    body = await request.json()

    try:
        req = CompletionRequest(**body)
    except Exception as e:
        log.warning(f"Completions validation error: {e}")
        return JSONResponse(
            {"error": {"message": str(e), "type": "invalid_request_error"}},
            status_code=422,
        )

    response = await _completion_engine.create_completion(req, raw_request=request)

    if isinstance(response, ErrorResponse):
        log.error(f"Completions engine error: {response.error.message} (code: {response.error.code})")
        return JSONResponse(response.model_dump(), status_code=response.error.code)

    if not body.get("stream"):
        return JSONResponse(response.model_dump())

    async def event_stream():
        async for chunk in response:
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/responses")
async def create_responses(request: Request):
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest, ResponsesResponse
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse

    body = await request.json()

    try:
        req = ResponsesRequest(**body)
    except Exception as e:
        log.warning(f"Responses validation error: {e}")
        return JSONResponse(
            {"error": {"message": str(e), "type": "invalid_request_error"}},
            status_code=422,
        )

    response = await _responses_engine.create_responses(req, raw_request=request)

    if isinstance(response, ErrorResponse):
        log.error(f"Responses engine error: {response.error.message} (code: {response.error.code})")
        return JSONResponse(response.model_dump(), status_code=response.error.code)

    if isinstance(response, ResponsesResponse):
        return JSONResponse(response.model_dump())

    async def event_stream():
        async for event in response:
            event_type = getattr(event, "type", "unknown")
            yield f"event: {event_type}\ndata: {event.model_dump_json(indent=None)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/v1/responses/{response_id}")
async def retrieve_responses(
    response_id: str,
    request: Request,
    starting_after: int | None = None,
    stream: bool | None = False,
):
    from vllm.entrypoints.openai.protocol import ResponsesResponse
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse

    try:
        response = await _responses_engine.retrieve_responses(
            response_id, starting_after=starting_after, stream=stream
        )
    except Exception as e:
        log.warning(f"Retrieve responses error: {e}")
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "message": str(e)}},
            status_code=422,
        )

    if isinstance(response, ErrorResponse):
        log.error(f"Retrieve responses engine error: {response.error.message} (code: {response.error.code})")
        return JSONResponse(response.model_dump(), status_code=response.error.code)

    if isinstance(response, ResponsesResponse):
        return JSONResponse(response.model_dump())

    async def event_stream():
        async for event in response:
            event_type = getattr(event, "type", "unknown")
            yield f"event: {event_type}\ndata: {event.model_dump_json(indent=None)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/responses/{response_id}/cancel")
async def cancel_responses(response_id: str, request: Request):
    from vllm.entrypoints.openai.protocol import ResponsesResponse
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse

    try:
        response = await _responses_engine.cancel_responses(response_id)
    except Exception as e:
        log.warning(f"Cancel responses error: {e}")
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "message": str(e)}},
            status_code=422,
        )

    if isinstance(response, ErrorResponse):
        log.error(f"Cancel responses engine error: {response.error.message} (code: {response.error.code})")
        return JSONResponse(response.model_dump(), status_code=response.error.code)

    return JSONResponse(response.model_dump())


@app.post("/v1/messages")
async def create_messages(request: Request):
    from vllm.entrypoints.anthropic.protocol import (
        AnthropicMessagesRequest,
        AnthropicMessagesResponse,
        AnthropicErrorResponse,
        AnthropicError,
    )
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse

    body = await request.json()

    try:
        req = AnthropicMessagesRequest(**body)
    except Exception as e:
        log.warning(f"Messages validation error: {e}")
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "message": str(e)}},
            status_code=422,
        )

    response = await _messages_engine.create_messages(req, raw_request=request)

    if isinstance(response, ErrorResponse):
        log.error(f"Messages engine error: {response.error.message} (code: {response.error.code})")
        return JSONResponse(
            AnthropicErrorResponse(
                error=AnthropicError(type=response.error.type, message=response.error.message)
            ).model_dump(),
            status_code=response.error.code,
        )

    if isinstance(response, AnthropicMessagesResponse):
        return JSONResponse(response.model_dump(exclude_none=True))

    return StreamingResponse(response, media_type="text/event-stream")


if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
