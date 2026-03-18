# OpenAI-Compatible vLLM Load Balancer
![vLLM worker banner](https://image.runpod.ai/preview/vllm/vllm-banner.png)


A FastAPI-based load balancer for serving vLLM models on Runpod. Built on top of [worker-vllm](https://github.com/runpod-workers/worker-vllm) as the base inference engine, extending it with Runpod's load balancer protocol and additional API endpoints.

## Architecture

```
worker-vllm (submodule)
  └── src/engine.py        → vLLMEngine: model loading, engine args, tokenizer
  └── src/download_model.py → optional model pre-baking

handler_lb.py              → FastAPI server with Runpod LB protocol
  ├── /ping                → health check (204 = init, 200 = ready)
  ├── /v1/models           → list available models
  ├── /v1/chat/completions → OpenAI chat completions (streaming + non-streaming)
  ├── /v1/completions      → OpenAI text completions (streaming + non-streaming)
  ├── /v1/responses        → OpenAI Responses API (streaming + non-streaming)
  └── /v1/messages         → Anthropic Messages API (streaming + non-streaming)
```

Runpod's load balancer polls `/ping` to manage worker routing:
- `204` — worker is initializing (not routed)
- `200` — worker is ready (included in pool)

## Prerequisites

- A Runpod account ([runpod.io](https://runpod.io))
- Runpod API key (available in your RunP[d dashboard)

## Docker Image

Use the pre-built Docker image: `runpod/vllm-loadbalancer:latest`

## Environment Variables

### Core (from worker-vllm)

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `MODEL_NAME` | **Yes** | HuggingFace model identifier | None |
| `HF_TOKEN` | No | HuggingFace token for gated models | None |
| `TENSOR_PARALLEL_SIZE` | No | Number of GPUs for tensor parallelism | `1` |
| `DTYPE` | No | Model precision | `auto` |
| `TRUST_REMOTE_CODE` | No | Allow remote code execution | `true` |
| `MAX_MODEL_LEN` | No | Maximum sequence length | auto |
| `GPU_MEMORY_UTILIZATION` | No | GPU memory usage ratio | `0.9` |
| `ENFORCE_EAGER` | No | Disable CUDA graphs | `false` |
| `QUANTIZATION` | No | Quantization method (e.g. `awq`, `gptq`) | None |

### Serving overrides

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | Override the served model name | model path |
| `OPENAI_RESPONSE_ROLE` | Role for assistant responses | `assistant` |
| `TRUST_REQUEST_CHAT_TEMPLATE` | Allow client-supplied chat templates | `false` |
| `REASONING_PARSER` | Reasoning parser (e.g. `deepseek_r1`) | None |
| `TOOL_CALL_PARSER` | Tool call parser | None |
| `ENABLE_AUTO_TOOL_CHOICE` | Enable automatic tool selection | `false` |
| `RETURN_TOKENS_AS_TOKEN_IDS` | Return token IDs instead of strings | `false` |
| `ENABLE_PROMPT_TOKENS_DETAILS` | Include prompt token details in usage | `false` |
| `ENABLE_FORCE_INCLUDE_USAGE` | Always include usage in response | `false` |
| `PORT` | HTTP server port | `80` |

## Deployment on Runpod

1. Create a new serverless endpoint
2. Use Docker image: `runpod/vllm-loadbalancer:latest`
3. Set `MODEL_NAME` (e.g. `meta-llama/Llama-3.1-8B-Instruct`)
4. Configure additional environment variables as needed

## API Usage

### OpenAI-compatible (chat completions)

```bash
curl -X POST "https://<endpoint-id>.api.runpod.ai/v1/chat/completions" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 100
  }'
```

### OpenAI-compatible (streaming)

```bash
curl -X POST "https://<endpoint-id>.api.runpod.ai/v1/chat/completions" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Anthropic Messages API

```bash
curl -X POST "https://<endpoint-id>.api.runpod.ai/v1/messages" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 100
  }'
```

### Health Check

```bash
curl -X GET "https://<endpoint-id>.api.runpod.ai/ping" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

## Using with Claude Code / Anthropic SDK

This endpoint exposes a `/v1/messages` route compatible with the Anthropic Messages API. Point the Anthropic SDK or Claude Code at your RunPod endpoint to use the deployed model.

```bash
export ANTHROPIC_BASE_URL=https://<endpoint-id>.api.runpod.ai/
export ANTHROPIC_API_KEY=$RUNPOD_API_KEY
```

Then use Claude Code normally — requests will be routed to your vLLM-backed endpoint instead of Anthropic's API.

Example with a specific endpoint:

```bash
export ANTHROPIC_BASE_URL=https://c0d2nwfzao5dej.api.runpod.ai/
export ANTHROPIC_API_KEY=$RUNPOD_API_KEY
claude --model <MODEL_NAME>
# example: claude --model zai-org/GLM-4.7-Flash
```


## Building from Source

```bash
git clone --recurse-submodules https://github.com/runpod-workers/vllm-loadbalancer-ep
docker build -t vllm-loadbalancer .
```

To bake a model into the image:

```bash
docker build \
  --build-arg MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  --secret id=HF_TOKEN \
  -t vllm-loadbalancer-llama .
```
