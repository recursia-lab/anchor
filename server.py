"""
Anchor — PaliGemma2 Multi-LoRA Serving Server
OpenAI-compatible API for PaliGemma2 with dynamic LoRA adapter switching.

Usage:
  model="open_circuit"  → routes to open_circuit LoRA adapter
  model="base"          → uses base model without LoRA
"""

import os
import time
import base64
import io
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("anchor")

MODEL_PATH = os.environ.get("MODEL_PATH", "/model")
LORA_PATH  = os.environ.get("LORA_PATH",  "/lora")
PORT       = int(os.environ.get("PORT", 8080))

model = None
processor = None
loaded_adapters: list[str] = []


def _load_image(image_url: str) -> Image.Image:
    if image_url.startswith("data:image"):
        b64 = image_url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    raise ValueError(f"Unsupported image_url scheme: {image_url[:30]}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, loaded_adapters

    log.info("Loading PaliGemma2 base model from %s ...", MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base.eval()

    lora_root = Path(LORA_PATH)
    adapters_found = sorted([d.name for d in lora_root.iterdir() if d.is_dir()]) if lora_root.exists() else []

    if adapters_found:
        log.info("Loading %d LoRA adapters: %s", len(adapters_found), adapters_found)
        global model
        model = base
        for adapter_name in adapters_found:
            adapter_dir = lora_root / adapter_name
            model.load_adapter(str(adapter_dir), adapter_name=adapter_name)
            loaded_adapters.append(adapter_name)
            log.info("  ✓ %s", adapter_name)
        model.disable_adapters()
    else:
        log.info("No LoRA adapters found, serving base model only")
        model = base

    log.info("Anchor ready. Adapters: %s", loaded_adapters or ["base"])
    yield

    log.info("Shutting down Anchor")


app = FastAPI(title="Anchor", description="PaliGemma2 Multi-LoRA Serving", lifespan=lifespan)


# ── API schemas ───────────────────────────────────────────────────────────────

class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    content: str | list[ContentItem]

class ChatRequest(BaseModel):
    model: str = "base"
    messages: list[Message]
    max_tokens: int = 10
    temperature: float = 0.0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "adapters": loaded_adapters}


@app.get("/v1/models")
def list_models():
    model_ids = loaded_adapters if loaded_adapters else ["base"]
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": 0,
                "owned_by": "recursia",
                "root": mid,
                "parent": None,
            }
            for mid in model_ids
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    adapter = req.model

    image: Optional[Image.Image] = None
    text_parts: list[str] = []

    for msg in req.messages:
        if msg.role != "user":
            continue
        content = msg.content
        if isinstance(content, str):
            text_parts.append(content)
        else:
            for item in content:
                if item.type == "image_url" and item.image_url:
                    try:
                        image = _load_image(item.image_url.url)
                    except Exception as e:
                        raise HTTPException(400, f"Image load error: {e}")
                elif item.type == "text" and item.text:
                    text_parts.append(item.text)

    prompt = " ".join(text_parts)

    # PaliGemma requires <image> token prefix when image is present
    if image is not None and not prompt.startswith("<image>"):
        prompt = "<image>" + prompt

    # Switch adapter
    try:
        if adapter in loaded_adapters:
            model.set_adapter(adapter)
        else:
            model.disable_adapters()
    except Exception as e:
        raise HTTPException(400, f"Adapter error: {e}")

    # Inference
    t0 = time.time()
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    answer = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    latency_ms = round((time.time() - t0) * 1000)

    prompt_tokens = input_len
    completion_tokens = len(output_ids[0]) - input_len

    log.info("adapter=%s latency=%dms answer=%r", adapter, latency_ms, answer)

    return {
        "id": f"chatcmpl-anchor-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": adapter,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency_ms,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
