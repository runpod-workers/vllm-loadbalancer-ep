# vLLM Load Balancer

A FastAPI-based load balancer for serving vLLM models with RunPod integration. Provides OpenAI-compatible APIs with streaming and non-streaming text generation.

## Prerequisites

Before you begin, make sure you have:

- A RunPod account (sign up at [runpod.io](https://runpod.io))
- RunPod API key (available in your RunPod dashboard)
- Basic understanding of REST APIs and HTTP requests
- `curl` or a similar tool for testing API endpoints

## Docker Image

Use the pre-built Docker image: `runpod/vllm-loadbalancer:dev`

## Environment Variables

Configure these environment variables in your RunPod endpoint:

| Variable | Required | Description | Default | Example |
|----------|----------|-------------|---------|---------|
| `MODEL_NAME` | **Yes** | HuggingFace model identifier | None | `microsoft/DialoGPT-medium` |
| `TENSOR_PARALLEL_SIZE` | No | Number of GPUs for model parallelism | `1` | `2` |
| `DTYPE` | No | Model precision type | `auto` | `float16` |
| `TRUST_REMOTE_CODE` | No | Allow remote code execution | `true` | `false` |
| `MAX_MODEL_LEN` | No | Maximum sequence length | None (auto) | `2048` |
| `GPU_MEMORY_UTILIZATION` | No | GPU memory usage ratio | `0.9` | `0.8` |
| `ENFORCE_EAGER` | No | Disable CUDA graphs | `false` | `true` |

## Deployment on RunPod

1. Create a new serverless endpoint
2. Use Docker image: `runpod/vllm-loadbalancer:dev`
3. Set required environment variable: `MODEL_NAME` (e.g., "microsoft/DialoGPT-medium")
4. Optional: Configure additional environment variables as needed

## API Usage with curl

### Text Completion (Non-streaming)

```bash
curl -X POST "https://your-endpoint-id.api.runpod.ai/v1/completions" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a story about a brave knight",
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

### Text Completion (Streaming)

```bash
curl -X POST "https://your-endpoint-id.api.runpod.ai/v1/completions" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me about artificial intelligence",
    "max_tokens": 200,
    "temperature": 0.8,
    "stream": true
  }'
```

### Chat Completions

```bash
curl -X POST "https://your-endpoint-id.api.runpod.ai/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Health Check

```bash
curl -X GET "https://your-endpoint-id.api.runpod.ai/ping" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

## Local Testing

Run the test script:
```bash
export ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"
python example.py
```