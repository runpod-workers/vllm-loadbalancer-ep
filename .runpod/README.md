![vLLM worker banner](https://image.runpod.ai/preview/vllm/vllm-banner.png)

Run LLMs using [vLLM](https://docs.vllm.ai) with OpenAI-compatible and Anthropic-compatible APIs on Runpod's Load Balancer for high-throughput, multi-worker scalability.

Built on [worker-vllm](https://github.com/runpod-workers/worker-vllm) as the base inference engine.

---

## Endpoint Configuration

All behaviour is controlled through environment variables:

| Environment Variable                | Description                                       | Default             | Options                                                            |
| ----------------------------------- | ------------------------------------------------- | ------------------- | ------------------------------------------------------------------ |
| `MODEL_NAME`                        | Path of the model weights                         | "facebook/opt-125m" | Local folder or Hugging Face repo ID                               |
| `HF_TOKEN`                          | HuggingFace access token for gated/private models |                     | Your HuggingFace access token                                      |
| `MAX_MODEL_LEN`                     | Model's maximum context length                    |                     | Integer (e.g., 4096)                                               |
| `QUANTIZATION`                      | Quantization method                               |                     | "awq", "gptq", "squeezellm", "bitsandbytes"                        |
| `TENSOR_PARALLEL_SIZE`              | Number of GPUs                                    | 1                   | Integer                                                            |
| `GPU_MEMORY_UTILIZATION`            | Fraction of GPU memory to use                     | 0.95                | Float between 0.0 and 1.0                                          |
| `MAX_NUM_SEQS`                      | Maximum number of sequences per iteration         | 256                 | Integer                                                            |
| `ENABLE_AUTO_TOOL_CHOICE`           | Enable automatic tool selection                   | false               | boolean (true or false)                                            |
| `TOOL_CALL_PARSER`                  | Parser for tool calls                             |                     | "mistral", "hermes", "llama3_json", "granite", "deepseek_v3", etc. |
| `REASONING_PARSER`                  | Parser for reasoning-capable models               |                     | "deepseek_r1", "qwen3", "granite", "hunyuan_a13b"                  |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | Override served model name in API                 |                     | String                                                             |
| `MAX_CONCURRENCY`                   | Maximum concurrent requests                       | 10                 | Integer                                                            |

**Pass any vLLM engine arg** not listed above by setting an env var with the **UPPERCASED** field name (e.g. `MAX_MODEL_LEN=4096`, `ENABLE_CHUNKED_PREFILL=true`). The worker auto-discovers all `AsyncEngineArgs` fields from env. See the [vLLM engine args docs](https://docs.vllm.ai/en/latest/configuration/engine_args) for all available options.

For complete configuration options, see the [full configuration documentation](https://github.com/runpod-workers/worker-vllm/blob/main/docs/configuration.md).

## API Endpoints

This worker exposes direct HTTP endpoints (no Runpod serverless wrapper). Use your endpoint URL directly:

```
https://<ENDPOINT_ID>.api.runpod.ai/
```

| Path | Method | Description |
|------|--------|-------------|
| `/ping` | GET | Health check (204 = init, 200 = ready) |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | OpenAI chat completions |
| `/v1/completions` | POST | OpenAI text completions |
| `/v1/responses` | POST | OpenAI Responses API |
| `/v1/messages` | POST | Anthropic Messages API |

### OpenAI-Compatible API

#### Chat Completions

```bash
curl -X POST "https://<ENDPOINT_ID>.api.runpod.ai/v1/chat/completions" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "What is the capital of France?" }
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

#### Chat Completions (Streaming)

```bash
curl -X POST "https://<ENDPOINT_ID>.api.runpod.ai/v1/chat/completions" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      { "role": "user", "content": "Write a short story about a robot." }
    ],
    "max_tokens": 500,
    "temperature": 0.8,
    "stream": true
  }'
```

#### Text Completions

```bash
curl -X POST "https://<ENDPOINT_ID>.api.runpod.ai/v1/completions" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

---

### Anthropic Messages API

Compatible with the Anthropic SDK and Claude Code. Point `ANTHROPIC_BASE_URL` at your endpoint:

```bash
export ANTHROPIC_BASE_URL=https://<ENDPOINT_ID>.api.runpod.ai/
export ANTHROPIC_API_KEY=$RUNPOD_API_KEY
```

#### Messages (Non-Streaming)

```bash
curl -X POST "https://<ENDPOINT_ID>.api.runpod.ai/v1/messages" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      { "role": "user", "content": "What is the capital of France?" }
    ],
    "max_tokens": 100
  }'
```

#### Messages (Streaming)

```bash
curl -X POST "https://<ENDPOINT_ID>.api.runpod.ai/v1/messages" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      { "role": "user", "content": "Write a short story about a robot." }
    ],
    "max_tokens": 500,
    "stream": true
  }'
```

---

## Usage

Below are minimal `python` snippets to get started quickly.

> Replace `<ENDPOINT_ID>` with your endpoint ID and `<API_KEY>` with a [RunPod API key](https://docs.runpod.io/get-started/api-keys).

### OpenAI SDK

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("RUNPOD_API_KEY"),
    base_url=f"https://<ENDPOINT_ID>.api.runpod.ai/v1",
)
```

`Chat Completions (Non-Streaming)`

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}],
    temperature=0,
    max_tokens=100,
)
print(response.choices[0].message.content)
```

`Chat Completions (Streaming)`

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}],
    temperature=0,
    max_tokens=100,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Anthropic SDK

```python
import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv("RUNPOD_API_KEY"),
    base_url=f"https://<ENDPOINT_ID>.api.runpod.ai/",
)

response = client.messages.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}],
    max_tokens=100,
)
print(response.content[0].text)
```

### Claude Code

```bash
export ANTHROPIC_BASE_URL=https://<ENDPOINT_ID>.api.runpod.ai/
export ANTHROPIC_API_KEY=$RUNPOD_API_KEY
claude --model <MODEL_NAME>
```

## Compatibility

For supported models, see the [vLLM supported models documentation](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Documentation

- **[🚀 Deployment Guide](https://docs.runpod.io/serverless/vllm/get-started)** - Step-by-step setup
- **[📖 Configuration Reference](https://github.com/runpod-workers/worker-vllm/blob/main/docs/configuration.md)** - All environment variables
- **[🔧 Development Guide](https://github.com/runpod-workers/worker-vllm/blob/main/docs/conventions.md)** - Architecture and patterns
