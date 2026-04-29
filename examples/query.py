"""
Query Anchor: send an image + question to a specific LoRA adapter.

Usage:
    python query.py --url https://your-anchor.run.app \
                    --adapter open_circuit \
                    --image path/to/image.jpg \
                    --prompt "Is there an open circuit defect? Answer YES or NO."
"""

import argparse
import base64
import json
import sys
import urllib.request


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def query(url: str, adapter: str, image_path: str, prompt: str, max_tokens: int = 10) -> str:
    payload = {
        "model": adapter,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())

    answer = result["choices"][0]["message"]["content"]
    latency = result["usage"].get("latency_ms", "?")
    print(f"Answer: {answer}  ({latency}ms)")
    return answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Anchor endpoint URL")
    p.add_argument("--adapter", default="base", help="LoRA adapter name (or 'base')")
    p.add_argument("--image", required=True, help="Path to image file")
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument("--max-tokens", type=int, default=10)
    args = p.parse_args()

    query(args.url, args.adapter, args.image, args.prompt, args.max_tokens)


if __name__ == "__main__":
    main()
