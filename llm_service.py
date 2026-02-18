# llm_service.py
# start with:
# uvicorn llm_service:app --host 0.0.0.0 --port 8001 --log-level info

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# =========================================================
# 🔹 GENERATION PRESETS
# =========================================================

GENERATION_PRESETS = {
    "sao_stheno": {
        "temperature": 1.12,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "max_new_tokens": 250,
    }
}

MODEL_NAME = "Sao10K/L3-8B-Stheno-v3.2"
PRESET = "sao_stheno"

# =========================================================
# 🔹 LOAD MODEL
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_8bit=True,
)

logger.info("✅ Model loaded successfully.")

# =========================================================
# 🔹 GENERATION ENDPOINT
# =========================================================

@app.post("/generate")
async def generate(request: Request):

    body = await request.json()
    messages = body["messages"]
    max_tokens = body.get("max_tokens", 400)

    preset = GENERATION_PRESETS[PRESET].copy()
    preset["max_new_tokens"] = max_tokens

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    except:
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        prompt = f"{system_msg}\n\n{user_msg}\n\n"

    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_args = {
        "inputs": model_inputs.input_ids,
        "attention_mask": model_inputs.attention_mask,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id,
        **preset,
    }

    def run_generation():
        with torch.no_grad():
            model.generate(**generation_args)

    thread = Thread(target=run_generation)
    thread.start()

    async def stream():
        buffer = ""
        last_len = 0
        for text in streamer:
            buffer += text
            delta = buffer[last_len:]
            last_len = len(buffer)
            if delta.strip():
                yield delta

    return StreamingResponse(stream(), media_type="text/plain")
