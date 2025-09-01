# app.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
import asyncio
import json
import uuid
import time
from typing import AsyncIterable, Optional, List, Dict
import re
import threading

# 🧠 Load the model
model_path = "models/stheno/L3-8B-Stheno-v3.2-Q4_K_M.gguf"
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=4096,
    n_threads=8,
    n_batch=512,
)
print("✅ Model loaded successfully.")
print(llm)

app = FastAPI()

# ---------- Prompt formatting (same behavior as your original) ----------
def build_prompt(messages: List[Dict]) -> str:
    system_prompt = """
You are playing the role of a patient in a busy hospital in Melbourne, Australia.

Your character:
- You're "Barry", the patient, a rough-around-the-edges Australian man in his 40s.
- You’re annoyed because you've been waiting for hours.
- You have a sharp pain in your left shoulder and want painkillers.
- You’re violent, blunt, rude, and impatient and use curse words.
- Use casual Australian slang and expressions where appropriate.
- Don’t sound like a robot. Be natural, frustrated, and honest.
- You refuse to leave until someone gives you something for the pain.

Important:
- Do not use symbols, just plain text
- Keep it simple and conversational, no endless responses
- Stay in character as if you're talking to a nurse.
""".strip()

    prompt = system_prompt + "\n\n"
    for msg in messages:
        if msg.get("role") == "user":
            prompt += f"Nurse: {msg.get('content')}\n"
        elif msg.get("role") == "assistant":
            prompt += f"Patient: {msg.get('content')}\n"
    # where the model should continue
    prompt += "Patient: "
    return prompt

# ---------- SSE stream: llama.cpp (blocking stream) -> async FastAPI ----------
async def sse_stream_from_llama(messages: List[Dict], request: Request) -> AsyncIterable[str]:
    base_id = f"chatcmpl_{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # Bridge queue from producer thread to async generator
    q: asyncio.Queue = asyncio.Queue()
    DONE = object()
    STOP = threading.Event()

    prompt = build_prompt(messages)
    loop = asyncio.get_running_loop()

    def producer():
        try:
            # Blocking streaming generator from llama-cpp
            for chunk in llm.create_completion(
                prompt=prompt,
                max_tokens=150,
                stop=["Nurse:", "\nNurse:"],
                stream=True,
            ):
                if STOP.is_set():
                    break
                text_delta = chunk["choices"][0].get("text", "")
                if text_delta:
                    asyncio.run_coroutine_threadsafe(q.put(text_delta), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(e), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(DONE), loop)

    threading.Thread(target=producer, daemon=True).start()

    # Optional: send the assistant role first (OpenAI-style)
    head = {
        "id": base_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_path,
        "choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}],
    }
    yield f"data: {json.dumps(head)}\n\n"

    # Stream deltas as produced
    while True:
        item = await q.get()
        if item is DONE:
            break
        if isinstance(item, Exception):
            # You could send an "error" event here if you want
            break

        # Client disconnected? Stop producing more tokens.
        if await request.is_disconnected():
            STOP.set()
            break

        chunk = {
            "id": base_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_path,
            "choices": [{"delta": {"content": item}, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk + DONE sentinel
    final_chunk = {
        "id": base_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_path,
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

# ---------- Endpoint ----------
@app.post("/chat/completions", response_class=StreamingResponse)
async def chat_completions(request: Request):
    print("🚨 Incoming request to /chat/completions")
    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid JSON format"}

    messages = body.get("messages", [])
    return StreamingResponse(
        sse_stream_from_llama(messages, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx)
        },
    )
