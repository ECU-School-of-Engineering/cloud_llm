# app.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
import asyncio
import json
import uuid
import time
from typing import AsyncIterable, List, Dict
import threading
import sqlite3
import uuid
# =========================================================
# Conversation Manager: handles history persistence
# =========================================================

# =========================================================
# Global session management
# =========================================================
DEFAULT_SESSION_ID = str(uuid.uuid4())  # created once at startup
print(f"✨ Default session initialized: {DEFAULT_SESSION_ID}")


class ConversationManager:
    def __init__(self, db_path="conversations.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            session_id TEXT,
            turn_number INTEGER,
            role TEXT,
            content TEXT,
            PRIMARY KEY (session_id, turn_number)
        )
        """)
        conn.commit()
        conn.close()


    def add_message(self, session_id: str, role: str, content: str):
        """Insert a single message with next incremental turn_number"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get last turn_number for this session
        cursor.execute(
            "SELECT COALESCE(MAX(turn_number), 0) FROM conversations WHERE session_id = ?",
            (session_id,)
        )
        last_turn = cursor.fetchone()[0]
        next_turn = last_turn + 1

        # Insert new message
        cursor.execute(
            "INSERT INTO conversations (session_id, turn_number, role, content) VALUES (?, ?, ?, ?)",
            (session_id, next_turn, role, content)
        )

        conn.commit()
        conn.close()



    def get_history(self, session_id: str) -> List[Dict]:
        """Fetch all messages for a given session"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT turn_number, role, content FROM conversations WHERE session_id = ? ORDER BY turn_number ASC", (session_id,))
        rows = c.fetchall()
        conn.close()
        return [{"turn_number": t, "role": r, "content": c} for t, r, c in rows]

    def clear_history(self, session_id: str):
        """Delete all messages for a given session"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

    def get_history(self, session_id: str) -> List[Dict]:
        """Fetch messages ordered by turn_number"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY turn_number ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"role": role, "content": content} for role, content in rows]


# =========================================================
# Prompt Manager: builds the final prompt for the model
# =========================================================
class PromptManager:
    def __init__(self, base_context: str, character_behavior: str):
        self.base_context = base_context.strip()
        self.character_behavior = character_behavior.strip()

    def build_prompt(self, history: List[Dict]) -> str:
        """Combine context, character, and conversation history into a prompt"""
        prompt = f"{self.base_context}\n\n{self.character_behavior}\n\n"
        for msg in history:
            if msg["role"] == "user":
                prompt += f"Nurse: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Patient: {msg['content']}\n"
        prompt += "Patient: "
        print(f"📝 Prompt sent to model:\n{prompt}\n")
        return prompt

# =========================================================
# Model initialization
# =========================================================
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

# =========================================================
# Managers
# =========================================================
conv_manager = ConversationManager()
prompt_manager = PromptManager(
    base_context="You are in a busy hospital in Melbourne, Australia.",
    character_behavior="""
- You're "Barry", a rough Australian man in his 40s.
- You're annoyed because you've been waiting for hours.
- You have a sharp pain in your left shoulder and want painkillers.
- You're violent, blunt, rude, and impatient and use curse words.
- Use casual Australian slang and expressions where appropriate.
- Stay in character as if talking to a nurse.
- Do not use symbols, keep it simple, conversational, and short.
""",
)

# =========================================================
# FastAPI app
# =========================================================
app = FastAPI()

# =========================================================
# SSE streaming bridge
# =========================================================
async def sse_stream_from_llama(session_id: str, request: Request) -> AsyncIterable[str]:
    base_id = f"chatcmpl_{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    q: asyncio.Queue = asyncio.Queue()
    DONE = object()

    # Get conversation history and build prompt
    history = conv_manager.get_history(session_id)
    print(f"\tHistory: {history}")
    prompt = prompt_manager.build_prompt(history)

    loop = asyncio.get_running_loop()
    full_text = []

    def producer():
        """Run llama.cpp in a background thread and push chunks into a queue"""
        try:
            for chunk in llm.create_completion(
                prompt=prompt,
                max_tokens=150,
                stop=["Nurse:", "\nNurse:"],
                stream=True,
            ):
                text_delta = chunk["choices"][0].get("text", "")
                if text_delta:
                    asyncio.run_coroutine_threadsafe(q.put(text_delta), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(e), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(DONE), loop)

    threading.Thread(target=producer, daemon=True).start()

    # Initial assistant role message (OpenAI-style)
    head = {
        "id": base_id,
        "object": "chat.completion.chunk",
        "system_fingerprint": session_id,
        "created": created,
        "model": model_path,
        "choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}],
    }
    yield f"data: {json.dumps(head)}\n\n"

    # Stream deltas to client
    while True:
        item = await q.get()
        if item is DONE:
            break
        if isinstance(item, Exception):
            break
        if await request.is_disconnected():
            break

        full_text.append(item)

        chunk = {
            "id": base_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_path,
            "choices": [{"delta": {"content": item}, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Save assistant's full reply into history
    assistant_reply = "".join(full_text)
    if assistant_reply!="":
        conv_manager.add_message(session_id, "assistant", assistant_reply)
    print(f"💾 Saved assistant reply for [{session_id}]: {assistant_reply[:50]}...")

    # Final stop chunk
    final_chunk = {
        "id": base_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_path,
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

# =========================================================
# Endpoints
# =========================================================
@app.post("/chat/completions", response_class=StreamingResponse)
async def chat_completions(request: Request):
    """Main endpoint: receive user messages, stream assistant reply"""
    raw = await request.body()
    print("📩 Raw:", raw.decode("utf-8"))
    
    body = json.loads(raw)
    # session_id = body.get("session_id", "default")
    session_id = body.get("session_id", DEFAULT_SESSION_ID)
    if not session_id:
        session_id = str(uuid.uuid4())  # auto-generate one
        print(f"✨ Created new session: {session_id}")
    
    user_messages = body.get("messages", [])

    # Save user messages into history
    for m in user_messages:
        conv_manager.add_message(session_id, m["role"], m["content"])

    return StreamingResponse(
        sse_stream_from_llama(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/chat/new_session")
async def new_session():
    """Endpoint to create a new session (conversation)"""
    session_id = str(uuid.uuid4())  # generate unique session id
    # Optionally: you could insert a "system" message here if you want
    return {"status": "ok", "session_id": session_id}

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Return the full conversation history for a given session_id"""
    history = conv_manager.get_history(session_id)
    return {
        "session_id": session_id,
        "history": history
    }

@app.get("/escalation/{escalation}")
async def get_escalation(escalation: str):
    """Return the full conversation history for a given session_id"""
    return {
        "escalation": escalation
    }