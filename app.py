# app.py

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import asyncio
import json
import uuid
import time
from typing import AsyncIterable, Optional
import re

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
# Initialize FastAPI app
app = FastAPI()

# 🧾 Prompt formatting and LLM call
def generate_response(messages: list[dict]) -> str:
    def clean_labels(text: str) -> str:
        return re.sub(r"(?m)^\s*(\w+\s?){1,2}:\s*", "", text)

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
    """

    prompt = system_prompt + "\n"
    for msg in messages:
        if msg.get("role") == "user":
            prompt += f"Nurse: {msg.get('content')}\n"
        elif msg.get("role") == "assistant":
            prompt += f"Patient: {msg.get('content')}\n"

    output = llm(prompt, max_tokens=150, stop=["Nurse:", "\nNurse:"])
    result = output["choices"][0]["text"].strip()

    return clean_labels(result)

# Endpoint
@app.post("/chat/completions", response_class=StreamingResponse)
async def chat_completions(request: Request):
    print("🚨 Incoming request to /chat/completions")
    try:
        body = await request.json()
    except Exception as e:
        return {"error": "Invalid JSON format"}

    messages = body.get("messages", [])
    return StreamingResponse(
        get_response(messages),
        media_type="text/event-stream"
    )

# Streaming generator
async def get_response(messages: list[dict], custom_session_id: Optional[str] = None) -> AsyncIterable[str]:
    base_id = f"chatcmpl_{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    full_response = generate_response(messages)

    for word in full_response.split():
        chunk = {
            "id": base_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "dolphin-2.8-mistral-7b-v02.GGUF",
            "choices": [
                {
                    "delta": {"content": word + " "},
                    "index": 0,
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.03)

    final_chunk = {
        "id": base_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "dolphin-2.8-mistral-7b-v02.GGUF",
        "choices": [
            {
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
