"""LangChain tool wrapper for Anchor vision inference."""

from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .client import AnchorClient


class AnchorInput(BaseModel):
    image_path: str = Field(description="Path to the image file to inspect")


class AnchorVisionTool(BaseTool):
    """LangChain tool that queries an Anchor multi-LoRA endpoint.

    Example::

        from anchor_vision import AnchorVisionTool

        tool = AnchorVisionTool(
            endpoint="https://your-anchor.run.app",
            adapter="open_circuit",
            prompt="Is there an open circuit defect? Answer YES or NO.",
        )
        result = tool.invoke({"image_path": "image.jpg"})
        # → "YES"
    """

    name: str = "anchor_vision"
    description: str = "Inspect an image using a fine-tuned PaliGemma2 LoRA adapter"
    args_schema: Type[BaseModel] = AnchorInput

    endpoint: str
    adapter: str = "base"
    prompt: str = "Describe what you see."
    max_tokens: int = 10
    timeout: int = 30

    def _run(self, image_path: str, **kwargs: Any) -> str:
        client = AnchorClient(self.endpoint, timeout=self.timeout)
        result = client.inspect(image_path, prompt=self.prompt, adapter=self.adapter, max_tokens=self.max_tokens)
        return result.answer
