"""
Batch query: run multiple images against one adapter, print results.

Usage:
    python batch_query.py --url https://your-anchor.run.app \
                          --adapter open_circuit \
                          --images dir/with/images/ \
                          --prompt "Defect present? YES or NO."
"""

import argparse
import base64
import json
import os
import time
import urllib.request
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def query_one(url: str, adapter: str, image_path: str, prompt: str) -> dict:
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
        "max_tokens": 5,
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--adapter", default="base")
    p.add_argument("--images", required=True, help="Directory of images or single image path")
    p.add_argument("--prompt", required=True)
    args = p.parse_args()

    path = Path(args.images)
    if path.is_dir():
        images = sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    else:
        images = [path]

    print(f"Querying {len(images)} images with adapter '{args.adapter}'")
    print(f"{'Image':<40} {'Answer':<10} {'ms':>6}")
    print("-" * 60)

    for img in images:
        t0 = time.time()
        try:
            result = query_one(args.url, args.adapter, str(img), args.prompt)
            answer = result["choices"][0]["message"]["content"].strip()
            latency = result["usage"].get("latency_ms", round((time.time() - t0) * 1000))
            print(f"{img.name:<40} {answer:<10} {latency:>6}ms")
        except Exception as e:
            print(f"{img.name:<40} ERROR: {e}")


if __name__ == "__main__":
    main()
