"""
Local stand-in for the vLLM OpenAI-compatible server.

Purpose: validate load_test_infertutor.py and score_infertutor.py end to end
on a laptop with no GPU and no Modal account. It streams fake tokens with a
configurable time-to-first-token and inter-token gap so the harness produces
realistic-looking JSON.

Usage:
    python tools/mock_vllm_server.py --port 8000 --ttft-ms 120 --itl-ms 15
    python starter_code/load_test_infertutor.py --url http://127.0.0.1:8000 \
        --mode mixed --users 20 --duration 20 --ramp-up 5 --label mock

Only aiohttp is used; it is already a dependency of the modal package.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from aiohttp import web

WORDS = ["kv", "cache", "decode", "prefill", "batch", "latency", "vllm", "gpu", "token", "queue"]


def build_chunk(index: int, model: str, content: str) -> bytes:
    payload = {
        "id": f"chatcmpl-mock-{index}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


async def health(_: web.Request) -> web.Response:
    return web.Response(text="")


async def chat_completions(request: web.Request) -> web.StreamResponse:
    app = request.app
    body = await request.json()
    max_tokens = int(body.get("max_tokens", 96))
    model = body.get("model", "mock")
    has_image = any(
        isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
        for m in body.get("messages", [])
    )

    # Simulate the vision encoder costing extra prefill time.
    ttft = app["ttft_s"] * (2.0 if has_image else 1.0)
    await asyncio.sleep(ttft)

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    for i in range(max_tokens):
        await resp.write(build_chunk(i, model, WORDS[i % len(WORDS)] + " "))
        await asyncio.sleep(app["itl_s"])
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ttft-ms", type=float, default=120.0)
    parser.add_argument("--itl-ms", type=float, default=15.0)
    args = parser.parse_args()

    app = web.Application()
    app["ttft_s"] = args.ttft_ms / 1000
    app["itl_s"] = args.itl_ms / 1000
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat_completions)
    print(f"Mock vLLM listening on http://{args.host}:{args.port} (ttft={args.ttft_ms}ms itl={args.itl_ms}ms)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
