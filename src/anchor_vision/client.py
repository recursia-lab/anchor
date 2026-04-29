"""Anchor API client — stdlib only, no external dependencies."""

import base64
import json
import urllib.request
from pathlib import Path
from typing import Optional, Union


class AnchorClient:
    """Synchronous client for the Anchor multi-LoRA vision API.

    Example::

        client = AnchorClient("https://your-anchor.run.app")
        result = client.inspect("image.jpg", adapter="open_circuit")
        print(result.answer)   # "YES"
        print(result.latency_ms)  # 216
    """

    def __init__(self, endpoint: str, timeout: int = 30):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def inspect(
        self,
        image: Union[str, Path, bytes],
        prompt: str = "Defect present? Answer YES or NO.",
        adapter: str = "base",
        max_tokens: int = 10,
    ) -> "InspectionResult":
        """Run inference on a single image.

        Args:
            image: File path, Path object, or raw bytes.
            prompt: Text prompt sent to the model.
            adapter: Name of the LoRA adapter to use.
            max_tokens: Maximum tokens to generate.

        Returns:
            InspectionResult with .answer and .latency_ms.
        """
        b64 = self._encode_image(image)
        payload = {
            "model": adapter,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }
        raw = self._post("/v1/chat/completions", payload)
        return InspectionResult(
            answer=raw["choices"][0]["message"]["content"].strip(),
            adapter=adapter,
            latency_ms=raw.get("usage", {}).get("latency_ms"),
            prompt_tokens=raw.get("usage", {}).get("prompt_tokens"),
            completion_tokens=raw.get("usage", {}).get("completion_tokens"),
        )

    def health(self) -> dict:
        """Return server health and loaded adapter list."""
        req = urllib.request.Request(f"{self.endpoint}/health")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def list_adapters(self) -> list:
        """Return list of loaded adapter names."""
        return self.health().get("adapters", [])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_image(self, image: Union[str, Path, bytes]) -> str:
        if isinstance(image, bytes):
            return base64.b64encode(image).decode()
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())


class InspectionResult:
    """Result from a single Anchor inference call."""

    def __init__(
        self,
        answer: str,
        adapter: str,
        latency_ms: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ):
        self.answer = answer
        self.adapter = adapter
        self.latency_ms = latency_ms
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def __repr__(self):
        return f"InspectionResult(answer={self.answer!r}, adapter={self.adapter!r}, latency_ms={self.latency_ms})"

    def __str__(self):
        return self.answer
