# Anchor

**PaliGemma2 multi-LoRA serving with OpenAI-compatible API.**

Load multiple LoRA adapters once. Switch between them at inference time — 216ms, no reload.

```
                    ┌─────────────────────────────────┐
  Request           │           Anchor                │
  model="short" ───▶│                                 │
                    │  PaliGemma2 base  (VRAM)        │
                    │  ├── adapter: missing_hole  ◀─  │──▶ "YES / NO"
                    │  ├── adapter: open_circuit  ◀─  │
                    │  ├── adapter: short  ◀──────── ─│  pointer swap
                    │  ├── adapter: mouse_bite    ◀─  │     216ms
                    │  └── adapter: spur          ◀─  │
                    └─────────────────────────────────┘
```

```bash
# Call the open_circuit adapter
curl https://your-anchor-endpoint/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "open_circuit",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "Does this PCB have an open circuit defect? Answer YES or NO."}
      ]
    }],
    "max_tokens": 3
  }'
```

## Quick Demo

```bash
# 1. Clone and build
git clone https://github.com/recursia-lab/anchor
docker build -t anchor .

# 2. Run (mount your model and adapters)
docker run --gpus all \
  -v /path/to/paligemma2:/model \
  -v /path/to/lora:/lora \
  -p 8080:8080 anchor

# 3. Query any adapter by name
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"open_circuit","messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,<b64>"}},
    {"type":"text","text":"Defect present? YES or NO."}
  ]}],"max_tokens":3}'
# → {"choices":[{"message":{"content":"YES"}}],"usage":{"latency_ms":216}}
```

## Why Anchor

Most serving frameworks load LoRA adapters **per request** — fetching from disk or
swapping from CPU at inference time. For production workloads where multiple
fine-tuned adapters are in active use, this adds hundreds of milliseconds per request.

Anchor takes a different approach: **all adapters live in GPU memory simultaneously**.
Switching is a pointer swap — 216ms, no disk I/O, no model reload.

| Framework | PaliGemma2 LoRA | Multi-adapter | Dynamic switch |
|---|---|---|---|
| **Anchor** | ✅ | ✅ all in VRAM | ✅ 216ms |
| vLLM | ✅ (since v0.7.0) | ✅ | per-request load |
| SGLang | 🚧 [PR #24034](https://github.com/sgl-project/sglang/pull/24034) | — | — |
| Ollama | ❌ | — | — |
| TGI / LoRAX | ❌ | — | — |

**When to use Anchor:** production scenarios with 2–10 adapters that all need
low-latency access. When one adapter is enough, vLLM works fine.

## Architecture

```
/model          ← PaliGemma2 base (bfloat16, device_map=auto)
/lora/
  adapter_1/    ← PEFT LoRA adapter (loaded via load_adapter)
  adapter_2/
  adapter_3/

Request: model="adapter_1"  →  set_adapter("adapter_1")  →  generate()  →  216ms
Request: model="adapter_2"  →  set_adapter("adapter_2")  →  generate()  →  216ms
Request: model="base"       →  disable_adapters()         →  generate()
```

All adapters stay in VRAM. Switching is just a pointer swap — no disk I/O, no model reload.

## Quick Start

### Local (GPU required)

```bash
# 1. Clone
git clone https://github.com/recursia-lab/anchor
cd anchor

# 2. Install
pip install -r requirements.txt

# 3. Place model and adapters
#    /model   → PaliGemma2 weights (from HuggingFace or your fine-tune)
#    /lora/   → one subfolder per adapter

MODEL_PATH=/path/to/model LORA_PATH=/path/to/lora python server.py
```

### Docker

```bash
docker build -t anchor .
docker run --gpus all \
  -v /path/to/model:/model \
  -v /path/to/lora:/lora \
  -p 8080:8080 \
  anchor
```

### Google Cloud Run (GPU)

```bash
# Edit cloudbuild.yaml substitutions, then:
gcloud builds submit --config cloudbuild.yaml

gcloud beta run deploy anchor \
  --image YOUR_IMAGE \
  --region us-east4 \
  --gpu=1 --gpu-type=nvidia-l4 \
  --cpu=8 --memory=32Gi \
  --no-cpu-throttling \
  --no-gpu-zonal-redundancy \
  --min-instances=0 \
  --startup-probe="tcpSocket.port=8080,initialDelaySeconds=240,timeoutSeconds=240,periodSeconds=240,failureThreshold=1"
```

## API

### `GET /health`

```json
{"status": "ok", "adapters": ["open_circuit", "short", "mouse_bite"]}
```

### `GET /v1/models`

Lists all loaded adapters in OpenAI format.

### `POST /v1/chat/completions`

OpenAI-compatible. Use `model` field to select adapter.

**Request:**
```json
{
  "model": "open_circuit",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<b64>"}},
      {"type": "text", "text": "<your prompt>"}
    ]
  }],
  "max_tokens": 10
}
```

**Response:**
```json
{
  "model": "open_circuit",
  "choices": [{"message": {"role": "assistant", "content": "YES"}}],
  "usage": {"prompt_tokens": 271, "completion_tokens": 1, "latency_ms": 216}
}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/model` | Path to PaliGemma2 base model |
| `LORA_PATH` | `/lora` | Directory of LoRA adapter subfolders |
| `PORT` | `8080` | HTTP port |

## Performance (Google Cloud Run, NVIDIA L4)

| Metric | Value |
|---|---|
| Cold start (model load) | ~3 min |
| Adapter switch latency | 216ms |
| Concurrent adapters in VRAM | 6 (tested) |
| GPU memory (6 PCB adapters) | ~12GB / 24GB L4 |

## Ecosystem

- **Adapters:** [recursia-lab/paligemma2-adapters](https://github.com/recursia-lab/paligemma2-adapters) — community LoRA adapter index
- **SGLang:** [PR #24034](https://github.com/sgl-project/sglang/pull/24034) — native PaliGemma2 LoRA support (pending merge)
- **vLLM:** supported since v0.7.0

## Roadmap

- [x] PEFT multi-LoRA server (this repo)
- [x] Google Cloud Run deployment
- [x] SGLang PR
- [ ] Ollama Modelfile
- [ ] AWQ quantization (2-5x speedup)
- [ ] Continuous batching
- [ ] LangChain integration

## About

Built by [Recursia Lab](https://github.com/recursia-lab) for industrial visual inspection.

PaliGemma2 is a vision-language model by Google DeepMind.
