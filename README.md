# vLLM Load Balancer

A FastAPI-based load balancer for serving vLLM models with RunPod integration. Provides OpenAI-compatible APIs with streaming and non-streaming text generation.

## Docker Image

Use the pre-built Docker image: `mwiki/lbvll:v1`

## Deployment on RunPod

1. Create a new serverless endpoint
2. Use Docker image: `mwiki/lbvll:v4`
3. Set environment variable: `MODEL_NAME` (e.g., "microsoft/DialoGPT-medium")

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