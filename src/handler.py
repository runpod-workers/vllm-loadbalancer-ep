from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Union, AsyncGenerator
import asyncio
import json
import os
import uvicorn
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

app = FastAPI(title="vLLM Load Balancing Server", version="1.0.0")

# Global variables
engine: Optional[AsyncLLMEngine] = None
engine_ready = False

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

async def create_engine():
    """Initialize the vLLM engine"""
    global engine, engine_ready
    
    try:
        # Get model name from environment variable
        model_name = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
        
        # Configure engine arguments
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=1,  # Adjust based on your GPU setup
            dtype="auto",
            trust_remote_code=True,
            max_model_len=None,  # Let vLLM decide based on model
            gpu_memory_utilization=0.9,
            enforce_eager=False,
        )
        
        # Create the engine
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        engine_ready = True
        print(f"vLLM engine initialized successfully with model: {model_name}")
        
    except Exception as e:
        print(f"Failed to initialize vLLM engine: {str(e)}")
        engine_ready = False
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the vLLM engine on startup"""
    await create_engine()

@app.get("/ping")
async def health_check():
    """Health check endpoint required by RunPod load balancer"""
    if not engine_ready:
        # Return 204 when initializing
        return {"status": "initializing"}, 204
    
    # Return 200 when healthy
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "vLLM Load Balancing Server",
        "status": "ready" if engine_ready else "initializing",
        "endpoints": {
            "health": "/ping",
            "generate": "/v1/completions",
            "chat": "/v1/chat/completions"
        }
    }

@app.post("/v1/completions", response_model=GenerationResponse)
async def generate_completion(request: GenerationRequest):
    """Generate text completion"""
    if not engine_ready or engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    try:
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
        )
        
        # Generate request ID
        request_id = random_uuid()
        
        if request.stream:
            return StreamingResponse(
                stream_completion(request.prompt, sampling_params, request_id),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming generation
            results = engine.generate(request.prompt, sampling_params, request_id)
            final_output = None
            async for output in results:
                final_output = output
            
            if final_output is None:
                raise HTTPException(status_code=500, detail="No output generated")
            
            generated_text = final_output.outputs[0].text
            finish_reason = final_output.outputs[0].finish_reason
            
            # Calculate token counts (approximate)
            prompt_tokens = len(request.prompt.split())
            completion_tokens = len(generated_text.split())
            
            return GenerationResponse(
                text=generated_text,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

async def stream_completion(prompt: str, sampling_params: SamplingParams, request_id: str) -> AsyncGenerator[str, None]:
    """Stream completion generator"""
    try:
        results = engine.generate(prompt, sampling_params, request_id)
        async for output in results:
            for output_item in output.outputs:
                yield f"data: {json.dumps({'text': output_item.text, 'finish_reason': output_item.finish_reason})}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """OpenAI-compatible chat completions endpoint"""
    if not engine_ready or engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    try:
        # Extract messages and convert to prompt
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Simple conversion of messages to prompt (you may want to improve this)
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt += f"{role}: {content}\n"
        prompt += "assistant: "
        
        # Create sampling parameters from request
        sampling_params = SamplingParams(
            max_tokens=request.get("max_tokens", 512),
            temperature=request.get("temperature", 0.7),
            top_p=request.get("top_p", 0.9),
            stop=request.get("stop"),
        )
        
        # Generate
        request_id = random_uuid()
        results = engine.generate(prompt, sampling_params, request_id)
        final_output = None
        async for output in results:
            final_output = output
        
        if final_output is None:
            raise HTTPException(status_code=500, detail="No output generated")
        
        generated_text = final_output.outputs[0].text
        
        # Return OpenAI-compatible response
        return {
            "id": request_id,
            "object": "chat.completion",
            "model": os.getenv("MODEL_NAME", "unknown"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": final_output.outputs[0].finish_reason
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(prompt.split()) + len(generated_text.split())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

if __name__ == "__main__":
    # Get ports from environment variables
    port = int(os.getenv("PORT", 8000))
    print(f"Starting vLLM server on port {port}")
    
    # If health port is different, you'd need to run a separate health server
    # For simplicity, we're using the same port here
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )