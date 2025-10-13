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
import logging
import yaml
from typing import List, Dict
from contextlib import asynccontextmanager
import re

# Configure logging 
logging.basicConfig(
    level=logging.INFO,  # default log level
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

def strip_emotion_tags(text: str) -> str:
    """Remove anything inside curly braces { ... } from a string."""
    return re.sub(r"\{[^}]*\}", "", text).strip()


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
    def __init__(self, id: str, role: Role, behaviours: BehaviourSet, milestones: List[Milestone], starting_escalation: int = 1):
        self.id = id
        self.role = role
        self.behaviours = behaviours
        self.milestones = milestones
        self.starting_escalation = starting_escalation

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

    def get_default_turns_per_milestone(self) -> int:
        return int(self.defaults.get("turns_per_milestone", 3))

    def get_recipe(self, recipe_id: str = None) -> Recipe:
        # Fall back to default recipe if not provided
        recipe_id = recipe_id or self.get_default_recipe_id()
        recipe_data = next(r for r in self.config["recipes"] if r["id"] == recipe_id)
        starting_escalation = int(recipe_data.get("starting_escalation", self.get_default_behaviour_level()))

        # Resolve role
        role_data = next(r for r in self.config["roles"] if r["id"] == recipe_data["role"])
        role = Role(
            id=role_data["id"],
            name=role_data["name"],
            base_context=role_data["base_context"],
            character_behavior=role_data["character_behavior"]
        )

        # Resolve behaviour set
        behaviours_data = next(b for b in self.config["behaviour_levels"] if b["id"] == recipe_data["behaviour_levels"])
        behaviour_levels = [BehaviourLevel(l["level"], l["description"]) for l in behaviours_data["levels"]]
        behaviour_set = BehaviourSet(id=behaviours_data["id"], levels=behaviour_levels)

        # Resolve milestones
        milestones_data = next(m for m in self.config["milestones"] if m["id"] == recipe_data["milestones"])
        milestones = [Milestone(s["order"], s["milestone"]) for s in milestones_data["steps"]]

        return Recipe(
            id=recipe_data["id"],
            role=role,
            behaviours=behaviour_set,
            milestones=milestones,
            starting_escalation=starting_escalation
        )



##===========================
# Milestone Tracker ####
class MilestoneTracker:
    def __init__(self, milestones: List[Milestone], turns_per_milestone: int = 3):
        self.milestones = sorted(milestones, key=lambda m: m.order)
        self.index = 0
        self.turn_counter = 0
        self.turns_per_milestone = turns_per_milestone

    def current(self) -> Milestone:
        return self.milestones[self.index]

    def record_turn(self):
        self.turn_counter += 1

    def should_advance(self) -> bool:
        return self.turn_counter >= self.turns_per_milestone and self.index < len(self.milestones) - 1

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
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Conversation history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            session_id TEXT,
            turn_number INTEGER,
            role TEXT,
            content TEXT,
            models_json TEXT,
            milestone TEXT,
            behaviour TEXT,
            escalation INTEGER,
            PRIMARY KEY (session_id, turn_number)
        )
        """)

        # Sessions tracking table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            recipe_id TEXT,
            created_at TEXT
        )
        """)

        conn.commit()
        conn.close()

    def log_session(self, session_id: str, recipe_id: str):
        """Insert or update session info in the sessions table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (session_id, recipe_id, created_at)
            VALUES (?, ?, ?)
        """, (session_id, recipe_id, timestamp))
        conn.commit()
        conn.close()
        logger.info(f"🧾 Logged session {session_id} ({recipe_id}) at {timestamp}")

    def add_message(self, session_id: str, role: str, content: str,
                models_json: str = None, milestone: str = None, behaviour: str = None, escalation: int = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COALESCE(MAX(turn_number), 0) FROM conversations WHERE session_id = ?",
            (session_id,)
        )
        last_turn = cursor.fetchone()[0]
        next_turn = last_turn + 1

        cursor.execute(
            """INSERT INTO conversations
            (session_id, turn_number, role, content, models_json, milestone, behaviour, escalation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, next_turn, role, content, models_json, milestone, behaviour, escalation)
)

        conn.commit()
        conn.close()



    def get_history(self, session_id: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content, milestone, behaviour
            FROM conversations
            WHERE session_id = ?
            ORDER BY turn_number ASC
        """, (session_id,))
        rows = cursor.fetchall()
        conn.close()
        return [
            {"role": role, "content": content, "milestone": milestone, "behaviour": behaviour}
            for role, content, milestone, behaviour in rows
        ]


    def get_tracker(self, session_id: str) -> "MilestoneTracker":
        return self.trackers.get(session_id)

    def set_tracker(self, session_id: str, tracker: "MilestoneTracker"):
        self.trackers[session_id] = tracker
    
    def get_or_create_tracker(self, session_id: str, recipe: "Recipe") -> "MilestoneTracker":
        if session_id not in self.trackers:
            self.trackers[session_id] = MilestoneTracker(recipe.milestones)
        return self.trackers[session_id]

## CREATE SESSION ####
def create_session(session_id: str= None, recipe_id: str = None) -> dict:
    global last_session_id
    if session_id is None:
        session_id = str(uuid.uuid4())
    # Use loader defaults if recipe_id not provided
    recipe = loader.get_recipe(recipe_id)
    behaviour_level = recipe.behaviours.get_level(loader.get_default_behaviour_level())
    session_escalations[session_id] = recipe.starting_escalation
    tracker = MilestoneTracker(recipe.milestones, turns_per_milestone=loader.get_default_turns_per_milestone())

    conv_manager.set_tracker(session_id, tracker)
    session_recipes[session_id] = recipe

    conv_manager.log_session(session_id, recipe.id)

    last_session_id = session_id
    # 🔹 Log the session info and recipe content
    logger.info(f"✨ Created session: {session_id}")
    logger.info(f"📖 Recipe ID: {recipe.id}")
    logger.info(f"👤 Role: {recipe.role.name} ({recipe.role.id})")
    logger.info(f"🎭 Character Behavior: {recipe.role.character_behavior.strip()}")
    logger.info("🚩 Milestones: " + ", ".join(m.description for m in recipe.milestones))
    logger.info("⚙️ Behaviours: " + ", ".join(f"{l.level}:{l.description}" for l in recipe.behaviours.levels))
    current_level = recipe.starting_escalation
    current_behaviour = recipe.behaviours.get_level(current_level).description
    return {
        "status": "ok",
        "session_id": session_id,
        "recipe_id": recipe.id,
        "behaviour_level": behaviour_level.level,
        "current_behaviour": current_behaviour,
    }



# =========================================================
# Prompt Manager: builds the final prompt for the model
# =========================================================
class PromptManager:
    def __init__(self, base_context: str, character_behavior: str):
        self.base_context = base_context.strip()
        self.character_behavior = character_behavior.strip()

    def build_messages(self, history: List[Dict], milestone: "Milestone" = None, behaviour: "BehaviourLevel" = None) -> List[Dict]:
        """Return structured chat messages instead of a single prompt"""

        # 🔹 System prompt: Barry’s persona + general behaviour + your behaviour_level
        system_prompt = f"{self.base_context}\n\nYour general behaviour is: {self.character_behavior}"
        if behaviour is not None:
            system_prompt += f"\n\nYour current behaviour for this chat is: {behaviour.description}"

        # 🔹 Build user-facing conversation (Nurse + Barry dialogue)
        convo = "Here is the conversation so far:\n"
        last_nurse_msg = next((msg for msg in reversed(history) if msg["role"] == "user"), None)

        for msg in history:
            if msg is last_nurse_msg:
                if milestone:
                    convo += f"\nIn the story arc of the roleplay you are currently {milestone}"
                convo += "\n***Now the Nurse asks:***\n"
            if msg["role"] == "user":
                convo += f"Nurse: {msg['content']}\n"
            elif msg["role"] == "assistant":
                convo += f"Barry: {msg['content']}\n"

        # convo += (
        #     "\nYour task: Write only Barry’s next line of dialogue in quotes "
        #     "Do not add anything else. Do not explain. Do not write stage directions or cues. Do not write assistant\n"
        # )
        convo += (
        "\nYour task: Write only Barry’s next line of dialogue. "
        "Start directly with Barry’s words. "
        "Do not include 'Barry:', 'Patient:', 'Assistant:', or any role labels. "
        "Do not add narration, explanations, or stage directions.\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": convo},
        ]

        logger.info(f"📝 Messages built for model:\n{messages}\n")
        return messages




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

    def stream(self, prompt: str, max_tokens=50, stop=None, **kwargs):
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

    def stream(self, messages_or_prompt, max_tokens=150, stop=None, **kwargs):
        """
        Stream text chunks.
        Accepts either:
        - messages: a list of {"role": "system"|"user"|"assistant", "content": "..."}
        - or a raw prompt string.
        """
        # 🔹 Convert messages → prompt string if needed
        if isinstance(messages_or_prompt, list):
            prompt = self.tokenizer.apply_chat_template(messages_or_prompt, tokenize=False)
        else:
            prompt = messages_or_prompt  # fallback for raw string

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
            delta = buffer[last_len:]
            last_len = len(buffer)
            if delta.strip() == "":
                continue
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
session_escalations: Dict[str, int] = {}

# =========================================================
# ConversationManager Manager
# =========================================================
conv_manager = ConversationManager()


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
last_session_id = None

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

    # 🔑 Load
    recipe = session_recipes[session_id]
    prompt_manager = PromptManager(
    base_context=recipe.role.base_context,
    character_behavior=recipe.role.character_behavior,
    )
    

    tracker = conv_manager.get_or_create_tracker(session_id, recipe)
    tracker.record_turn()

    if tracker.should_advance():
        tracker.advance()
    level = session_escalations.get(session_id, loader.get_default_behaviour_level())
    behaviour_level = recipe.behaviours.get_level(level)
    current_behaviour = behaviour_level.description
    messages = prompt_manager.build_messages(history, milestone=tracker.current(),behaviour=behaviour_level)

    loop = asyncio.get_running_loop()
    full_text = []

    def producer():
        try:
            for text_delta in backend.stream(
                messages,   # <--- just pass structured messages now
                max_tokens=150,
                stop=["Nurse:", "\nNurse:", "assistant", "Assistant:", "Patient:", "\nBarry:"],
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



    assistant_reply = "".join(full_text).strip()
    if assistant_reply:
        current_milestone = tracker.current().description
        conv_manager.add_message(session_id, "assistant", assistant_reply,milestone=current_milestone, behaviour=current_behaviour, escalation=level)
    logger.info(f"💾 Saved assistant reply for [{session_id}]: {assistant_reply}")

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
    session_id = body.get("session_id") or last_session_id
    logger.info(f"📩 Session ID received: {session_id}")

    if not session_id:
        # no session ever created yet → make one
        session = create_session()
        session_id = session["session_id"]

    # If no session_id was passed, create a proper new session
    # if not session_id or session_id not in session_recipes:
    #     session_data = create_session()
    #     session_id = session_data["session_id"]
    #     logger.info(f"✨ Created new session via /chat/completions: {session_id}")
    if session_id not in session_recipes:
        create_session(session_id)   # this populates session_recipes
        conv_manager.set_tracker(
            session_id, MilestoneTracker(recipe.milestones)
        )
    recipe = session_recipes[session_id]
    user_messages = body.get("messages", [])
    # for m in user_messages:
    #     clean_content = strip_emotion_tags(m["content"])
    #     conv_manager.add_message(session_id, m["role"], clean_content)
    if user_messages:
        latest = user_messages[-1]
        if latest["role"] == "user":
            clean_content = strip_emotion_tags(latest["content"])
            models_json = json.dumps(latest.get("models", {}))  
            current_milestone = conv_manager.get_tracker(session_id).current().description
            level = session_escalations.get(session_id, loader.get_default_behaviour_level())
            current_behaviour = recipe.behaviours.get_level(level).description
            conv_manager.add_message(
                session_id, "user", clean_content,
                models_json=models_json,
                milestone=current_milestone,
                behaviour=current_behaviour,
                escalation=level
            )


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

@app.get("/chat/new_session")
async def new_session_get(recipe_id: str = None):
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
async def get_escalation(escalation: int, session_id: str = None):
    """
    Get (and optionally set) the current escalation level.
    If session_id is provided, store it for that session.
    """
    recipe_id = loader.get_default_recipe_id()
    recipe = loader.get_recipe(recipe_id)

    level = int(escalation)
    behaviour = recipe.behaviours.get_level(level)

    if session_id:
        session_escalations[session_id] = level
        logger.info(f"🔥 Escalation for session {session_id} set to {level}")

    return {
        "session_id": session_id,
        "level": behaviour.level,
        "behaviour": behaviour.description
    }


@app.get("/chat/sessions")
async def list_sessions():
    conn = sqlite3.connect(conv_manager.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, recipe_id, created_at FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"session_id": r[0], "recipe_id": r[1], "created_at": r[2]} for r in rows]

