"""
LangChain integration for Anchor.

Use Anchor as a LangChain BaseTool or directly in a chain.

Requirements:
    pip install langchain langchain-core

Usage:
    from langchain_integration import AnchorVisionTool

    tool = AnchorVisionTool(
        endpoint="https://your-anchor.run.app",
        adapter="open_circuit",
        prompt="Is there an open circuit defect? Answer YES or NO.",
    )

    result = tool.invoke({"image_path": "path/to/image.jpg"})
    print(result)  # "YES" or "NO"
"""

import base64
import json
import urllib.request
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class AnchorInput(BaseModel):
    image_path: str = Field(description="Path to the image file to inspect")


class AnchorVisionTool(BaseTool):
    """LangChain tool that queries an Anchor multi-LoRA endpoint."""

    name: str = "anchor_vision"
    description: str = "Inspect an image using a fine-tuned PaliGemma2 LoRA adapter"
    args_schema: Type[BaseModel] = AnchorInput

    endpoint: str
    adapter: str = "base"
    prompt: str
    max_tokens: int = 10

    def _run(self, image_path: str, **kwargs: Any) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        payload = {
            "model": self.adapter,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
        }

        req = urllib.request.Request(
            f"{self.endpoint.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        return result["choices"][0]["message"]["content"].strip()


# ── Example: use in a simple chain ───────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python langchain_integration.py <endpoint> <image_path>")
        sys.exit(1)

    endpoint, image_path = sys.argv[1], sys.argv[2]

    tool = AnchorVisionTool(
        endpoint=endpoint,
        adapter="open_circuit",
        prompt="Is there a defect in this PCB image? Answer YES or NO.",
    )

    result = tool.invoke({"image_path": image_path})
    print(f"Inspection result: {result}")
