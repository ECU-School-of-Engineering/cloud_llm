# app.py
# start with: uvicorn app:app --host 0.0.0.0 --port 8080 --log-level debug


from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import uuid
import time
from typing import AsyncIterable, List, Dict
import threading
import sqlite3
import uuid
import logging
import yaml
from typing import List, Dict
from contextlib import asynccontextmanager
# Configure logging 
logging.basicConfig(
    level=logging.DEBUG,  # default log level
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

## Content classes ###
#----------------------------------------------#

class Role:
    def __init__(self, id: str, name: str, base_context: str, character_behavior: str):
        self.id = id
        self.name = name
        self.base_context = base_context
        self.character_behavior = character_behavior


class BehaviourLevel:
    def __init__(self, level: int, description: str):
        self.level = level
        self.description = description


class BehaviourSet:
    def __init__(self, id: str, levels: List[BehaviourLevel]):
        self.id = id
        self.levels = levels

    def get_level(self, level: int) -> BehaviourLevel:
        for l in self.levels:
            if l.level == level:
                return l
        # fallback: return the highest defined level
        return max(self.levels, key=lambda l: l.level)



class Milestone:
    def __init__(self, order: int, description: str):
        self.order = order
        self.description = description

    def __str__(self):
        return self.description


class Recipe:
    def __init__(self, id: str, role: Role, behaviours: BehaviourSet, milestones: List[Milestone]):
        self.id = id
        self.role = role
        self.behaviours = behaviours
        self.milestones = milestones

## Confing Loader classes ###
#----------------------------------------------#

class ConfigLoader:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Grab defaults section if it exists
        self.defaults = self.config.get("defaults", {})

    def get_default_recipe_id(self) -> str:
        return self.defaults.get("recipe_id")

    def get_default_behaviour_level(self) -> int:
        return int(self.defaults.get("behaviour_level", 1))

    def get_default_turns_per_step(self) -> int:
        return int(self.defaults.get("turns_per_step", 3))

    def get_recipe(self, recipe_id: str = None) -> Recipe:
        # Fall back to default recipe if not provided
        recipe_id = recipe_id or self.get_default_recipe_id()
        recipe_data = next(r for r in self.config["recipes"] if r["id"] == recipe_id)

        # Resolve role
        role_data = next(r for r in self.config["roles"] if r["id"] == recipe_data["role"])
        role = Role(
            id=role_data["id"],
            name=role_data["name"],
            base_context=role_data["base_context"],
            character_behavior=role_data["character_behavior"]
        )

        # Resolve behaviour set
        behaviours_data = next(b for b in self.config["behaviours"] if b["id"] == recipe_data["behaviours"])
        behaviour_levels = [BehaviourLevel(l["level"], l["description"]) for l in behaviours_data["levels"]]
        behaviour_set = BehaviourSet(id=behaviours_data["id"], levels=behaviour_levels)

        # Resolve milestones
        milestones_data = next(m for m in self.config["milestones"] if m["id"] == recipe_data["milestones"])
        milestones = [Milestone(s["order"], s["milestone"]) for s in milestones_data["steps"]]

        return Recipe(
            id=recipe_data["id"],
            role=role,
            behaviours=behaviour_set,
            milestones=milestones
        )



##===========================
# Milestone Tracker ####
class MilestoneTracker:
    def __init__(self, milestones: List[Milestone], turns_per_step: int = 3):
        self.milestones = sorted(milestones, key=lambda m: m.order)
        self.index = 0
        self.turn_counter = 0
        self.turns_per_step = turns_per_step

    def current(self) -> Milestone:
        return self.milestones[self.index]

    def record_turn(self):
        self.turn_counter += 1

    def should_advance(self) -> bool:
        return self.turn_counter >= self.turns_per_step and self.index < len(self.milestones) - 1

    def advance(self):
        if self.should_advance():
            self.index += 1
            self.turn_counter = 0


# =========================================================
# Conversation Manager: handles history persistence
# =========================================================

class ConversationManager:
    def __init__(self, db_path="conversations.db"):
        self.db_path = db_path
        self._init_db()
        self.trackers = {}

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

        cursor.execute(
            "SELECT COALESCE(MAX(turn_number), 0) FROM conversations WHERE session_id = ?",
            (session_id,)
        )
        last_turn = cursor.fetchone()[0]
        next_turn = last_turn + 1

        cursor.execute(
            "INSERT INTO conversations (session_id, turn_number, role, content) VALUES (?, ?, ?, ?)",
            (session_id, next_turn, role, content)
        )

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

    def get_tracker(self, session_id: str) -> "MilestoneTracker":
        return self.trackers.get(session_id)

    def set_tracker(self, session_id: str, tracker: "MilestoneTracker"):
        self.trackers[session_id] = tracker
    
    def get_or_create_tracker(self, session_id: str, recipe: "Recipe") -> "MilestoneTracker":
        if session_id not in self.trackers:
            self.trackers[session_id] = MilestoneTracker(recipe.milestones)
        return self.trackers[session_id]

## CREATE SESSION ####
def create_session(session_id: str= str(uuid.uuid4()), recipe_id: str = None) -> dict:

    # Use loader defaults if recipe_id not provided
    recipe = loader.get_recipe(recipe_id)
    behaviour_level = recipe.behaviours.get_level(loader.get_default_behaviour_level())
    tracker = MilestoneTracker(recipe.milestones, turns_per_step=loader.get_default_turns_per_step())

    conv_manager.set_tracker(session_id, tracker)
    session_recipes[session_id] = recipe

    # 🔹 Log the session info and recipe content
    logger.info(f"✨ Created session: {session_id}")
    logger.info(f"📖 Recipe ID: {recipe.id}")
    logger.info(f"👤 Role: {recipe.role.name} ({recipe.role.id})")
    logger.info(f"🎭 Character Behavior: {recipe.role.character_behavior.strip()}")
    logger.info("🚩 Milestones: " + ", ".join(m.description for m in recipe.milestones))
    logger.info("⚙️ Behaviours: " + ", ".join(f"{l.level}:{l.description}" for l in recipe.behaviours.levels))

    return {
        "status": "ok",
        "session_id": session_id,
        "recipe_id": recipe.id,
        "behaviour_level": behaviour_level.level,
    }



# =========================================================
# Prompt Manager: builds the final prompt for the model
# =========================================================
class PromptManager:
    def __init__(self, base_context: str, character_behavior: str):
        self.base_context = base_context.strip()
        self.character_behavior = character_behavior.strip()

    def build_prompt(self, history: List[Dict], milestone: "Milestone" = None) -> str:
        """Combine context, character, milestone, and conversation history into a prompt"""
        prompt = f"{self.base_context}\n"
        prompt += f"\nYour general behaviour is: {self.character_behavior}\n"

        if milestone:
            prompt += f"In the roleplay you are currently {milestone}\n"

        prompt += f"So far the conversation has been: {milestone}\n"
        for msg in history:
            if msg["role"] == "user":
                prompt += f"Nurse: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Patient: {msg['content']}\n"

        prompt += "Patient: "
        logger.info(f"📝 Prompt sent to model:\n{prompt}\n")
        return prompt



# =========================================================
# LLM Backends
# =========================================================
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def stream(self, prompt: str, **kwargs):
        """Yield text chunks as they are generated"""
        pass


# Backend 1: llama.cpp
from llama_cpp import Llama


class LlamaBackend(LLMBackend):
    def __init__(self, model_path: str, **kwargs):
        self.llm = Llama(model_path=model_path, **kwargs)

    def stream(self, prompt: str, max_tokens=150, stop=None, **kwargs):
        for chunk in self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            stop=stop,
            stream=True,
        ):
            text_delta = chunk["choices"][0].get("text", "")
            if text_delta:
                yield text_delta


# Backend 2: HF Transformers + TextIteratorStreamer
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class HFStreamerBackend(LLMBackend):
    def __init__(self, model_name: str, device="cuda", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map="auto" if device.startswith("cuda") else None,
        )
        self.device = device

    def stream(self, prompt: str, max_tokens=150, stop=None, **kwargs):
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_args = {
            "inputs": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask, 
            "max_new_tokens": max_tokens,
            "streamer": streamer,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id
        }

        thread = Thread(target=self.model.generate, kwargs=generation_args)
        thread.start()

        buffer = ""
        last_len = 0
        for text in streamer:
            buffer += text

            # compute delta
            delta = buffer[last_len:]
            last_len = len(buffer)

            if delta.strip() == "":
                continue  # skip empty fragments

            logger.debug(f"🟢 Sending delta: [{delta}]")
            yield delta




# =========================================================
# Model initialization
# =========================================================
# 🔽 CHOOSE BACKEND HERE
# backend = LlamaBackend(
#     model_path="models/stheno/L3-8B-Stheno-v3.2-Q4_K_M.gguf",
#     n_gpu_layers=-1,
#     n_ctx=4096,
#     n_threads=8,
#     n_batch=512,
# )
backend = HFStreamerBackend("Sao10K/L3-8B-Stheno-v3.2", device="cuda")
logger.info("✅ Model backend loaded successfully.")

# Config
loader = ConfigLoader("scenarios.yml")
session_recipes = {}  


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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🔹 Startup: create session(s)
    startup_session = create_session()
    logger.info(f"✨ Startup session created: {startup_session['session_id']} with recipe {startup_session['recipe_id']}")

    # Yield control to the app
    yield

    # 🔹 Shutdown: cleanup if needed
    logger.info("👋 Shutting down...")

app = FastAPI(lifespan=lifespan)

# =========================================================
# SSE streaming bridge
# =========================================================
async def sse_stream(session_id: str, request: Request, backend: LLMBackend) -> AsyncIterable[str]:
    base_id = f"chatcmpl_{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    q: asyncio.Queue = asyncio.Queue()
    DONE = object()

    history = conv_manager.get_history(session_id)
    logger.info(f"\tHistory: {history}")

    # 🔑 Fetch recipe from memory
    recipe = session_recipes[session_id]

    tracker = conv_manager.get_or_create_tracker(session_id, recipe)
    tracker.record_turn()

    if tracker.should_advance():
        tracker.advance()

    prompt = prompt_manager.build_prompt(
        milestone=tracker.current(),
        history=history
    )

    loop = asyncio.get_running_loop()
    full_text = []

    def producer():
        try:
            for text_delta in backend.stream(
                prompt,
                max_tokens=150,
                stop=["Nurse:", "\nNurse:"],
            ):
                asyncio.run_coroutine_threadsafe(q.put(text_delta), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(e), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(DONE), loop)

    threading.Thread(target=producer, daemon=True).start()

    head = {
        "id": base_id,
        "object": "chat.completion.chunk",
        "system_fingerprint": session_id,
        "created": created,
        "model": str(backend),
        "choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}],
    }
    yield f"data: {json.dumps(head)}\n\n"

    while True:
        item = await q.get()
        if item is DONE:
            logger.debug("🔴 DONE received from producer")  # DEBUG
            break
        if isinstance(item, Exception):
            logger.debug(f"❌ Exception in producer: {item}")  # DEBUG
            break
        if await request.is_disconnected():
            logger.debug("⚠️ Client disconnected")  # DEBUG
            break

        logger.debug(f"🟡 SSE yielding: [{item}]")  # DEBUG
        full_text.append(item)
        chunk = {
            "id": base_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": str(backend),
            "choices": [{"delta": {"content": item}, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        await asyncio.sleep(0)  # let event loop flush



    assistant_reply = "".join(full_text)
    if assistant_reply:
        conv_manager.add_message(session_id, "assistant", assistant_reply)
    logger.info(f"💾 Saved assistant reply for [{session_id}]: {assistant_reply[:50]}...")

    # ✅ Only send a stop signal, not the whole text again
    final_chunk = {
        "id": base_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": str(backend),
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# =========================================================
# Endpoints
# =========================================================
@app.post("/chat/completions", response_class=StreamingResponse)
async def chat_completions(request: Request):
    raw = await request.body()
    logger.debug(f"📩 Raw: {raw.decode('utf-8')}")

    body = json.loads(raw)
    session_id = body.get("session_id")
    logger.info(f"📩 Session ID received: {session_id}")
    # If no session_id was passed, create a proper new session
    # if not session_id or session_id not in session_recipes:
    #     session_data = create_session()
    #     session_id = session_data["session_id"]
    #     logger.info(f"✨ Created new session via /chat/completions: {session_id}")
    if session_id not in session_recipes:
        session_info = create_session(session_id)
        session_recipes[session_id] = session_info["recipe"]
        conv_manager.set_tracker(
            session_id, MilestoneTracker(session_info["recipe"].milestones)
        )
    user_messages = body.get("messages", [])
    for m in user_messages:
        conv_manager.add_message(session_id, m["role"], m["content"])

    return StreamingResponse(
        sse_stream(session_id, request, backend),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



@app.post("/chat/new_session")
async def new_session(recipe_id: str = None):
    return create_session(recipe_id)




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

