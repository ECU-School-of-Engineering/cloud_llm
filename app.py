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
from fastapi.middleware.cors import CORSMiddleware
# Configure logging 
logging.basicConfig(
    level=logging.DEBUG,  # default log level
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
        self.service_config = self.config.get("service_config", {}) 

    def get_service_config(self) -> dict:
        """Return runtime/service configuration section with defaults."""
        return {
            "flush_chars": int(self.service_config.get("flush_chars", 40)),
            "flush_delay": float(self.service_config.get("flush_delay", 0.05)),
            "max_tokens": int(self.service_config.get("max_tokens", 400)),
            "stream_log_level": self.service_config.get("stream_log_level", "info"),
        }

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

    def advance(self, behaviour_level: int = None):
        """Advance to the next milestone, with optional behavior-based overrides."""
        logger.info(f"🛑 START advance | current.order={self.current().order}, behaviour={behaviour_level}, index={self.index}")

        if not self.should_advance():
            return

        # Default: go to next milestone
        next_index = self.index + 1

        # 🔹 Rule 1: Jump from order 3 to order 5 if behaviour==1
        if self.current().order == 3 and behaviour_level == 1:
            logger.info("🚀 Jump rule triggered (3 → 5, good ending)")
            try:
                next_index = next(
                    i for i, m in enumerate(self.milestones)
                    if m.order == 5
                )
            except StopIteration:
                logger.warning("⚠️ Could not find milestone with order=5; staying put.")
                next_index = self.index

        # 🔹 Rule 2: Stay at milestone 4 or 5
        elif self.current().order in (4, 5):
            logger.info("🛑 Staying on current milestone (4 or 5, no progression rule)")
            next_index = self.index

        # Apply update
        if 0 <= next_index < len(self.milestones):
            self.index = next_index
        else:
            logger.warning(f"⚠️ Invalid next_index {next_index}, staying at {self.index}")

        self.turn_counter = 0
        logger.info(f"✅ END advance | new.order={self.current().order}, index={self.index}, behaviour={behaviour_level}")



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
            recipe_json TEXT,
            created_at TEXT
        )
        """)
        
        # Feedback table (stores any arbitrary JSON feedback)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            payload_json TEXT,
            created_at TEXT
        )
        """)

        conn.commit()
        conn.close()

    def log_session(self, session_id: str, recipe: "Recipe"):
            """Insert or update session info in the sessions table, including full recipe"""
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Serialize recipe details
            recipe_data = {
                "id": recipe.id,
                "role": {
                    "id": recipe.role.id,
                    "name": recipe.role.name,
                    "base_context": recipe.role.base_context,
                    "character_behavior": recipe.role.character_behavior,
                },
                "behaviour_levels": [
                    {"level": l.level, "description": l.description}
                    for l in recipe.behaviours.levels
                ],
                "milestones": [
                    {"order": m.order, "description": m.description}
                    for m in recipe.milestones
                ],
                "starting_escalation": recipe.starting_escalation,
            }

            cursor.execute("""
                INSERT OR REPLACE INTO sessions (session_id, recipe_id, recipe_json, created_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, recipe.id, json.dumps(recipe_data), timestamp))

            conn.commit()
            conn.close()
            logger.info(f"🧾 Logged session {session_id} ({recipe.id}) with recipe details at {timestamp}")

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

    def get_history_rolecontent(self, session_id: str) -> List[Dict]:
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
            {"role": role, "content": content}
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
    behaviour_level = recipe.behaviours.get_level(recipe.starting_escalation)
    session_escalations[session_id] = recipe.starting_escalation
    tracker = MilestoneTracker(recipe.milestones, turns_per_milestone=loader.get_default_turns_per_milestone())

    conv_manager.set_tracker(session_id, tracker)
    session_recipes[session_id] = recipe

    conv_manager.log_session(session_id, recipe)

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
# Prompt Manager: builds structured JSON prompts for the model
# =========================================================
class PromptManager:
    def __init__(self, base_context: str, character_behavior: str):
        self.base_context = base_context.strip()
        self.character_behavior = character_behavior.strip()

    def build_messages(
        self,
        history: List[Dict],
        milestone: "Milestone" = None,
        behaviour: "BehaviourLevel" = None,
    ) -> List[Dict]:
        """
        Build a structured JSON-style prompt for consistent roleplay.
        """

        # ---- SYSTEM PROMPT ----
        system_prompt = (
            "You are a roleplaying AI actor.\n"
            "You must output ONLY valid JSON with this structure:\n"
            "{\n"
            '  "reply": "Barry’s in-character spoken response",\n'
            '  "emotion": "current emotion",\n'
            '  "action": "short description of what Barry does",\n'
            '  "escalation": "yes or no, assessing if what the nurse had just said helps to calm down Barry",\n'
            '  "summary": "sum up the conversation so far"\n'
            "}\n\n"
            "No explanations, no narration outside JSON."
        )

        # ---- STRUCTURED CONTEXT ----
        structured_context = {
            "situation": self.base_context,
            "character_description": {
                "behavior_baseline": self.character_behavior,
                "current_behavior": behaviour.description if behaviour else None,
            },
            "story_arc": {
                "current_milestone": str(milestone) if milestone else None,
            },
            "interaction_history": history[-6:] if history else [],
        }

        # ---- USER PROMPT ----
        user_prompt = json.dumps(structured_context, indent=2)
        user_prompt += "\n\nNow, as Barry, produce the next line of dialogue using your current_behavior and your current milestone in the story_arc in JSON format as specified.\nassistant:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"🧠 Structured messages built for model:\n{json.dumps(messages, indent=2)}")
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

    def stream(self, prompt: str, max_tokens=200, stop=None, **kwargs):
        for chunk in self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            stop=stop,
            stream=True,
        ):
            text_delta = chunk["choices"][0].get("text", "")
            if text_delta:
                yield text_delta


# =========================================================
# Backend 2: HF Transformers + TextIteratorStreamer
# =========================================================
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

torch.cuda.empty_cache()
# 🔹 Generation presets (optional tuning per model)
GENERATION_PRESETS = {
    "sao_stheno": {
        "temperature": 1.12,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "max_new_tokens": 250,
    },
    "openchat": {
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "max_new_tokens": 200,
    },
    "vicuna": {
        "temperature": 0.75,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "max_new_tokens": 220,
    },
}


class HFStreamerBackend(LLMBackend):
    def __init__(
        self,
        model_name: str,
        device="cuda",
        preset_name="openchat",
        use_chat_template=True,       
        **kwargs,
    ):
        self.use_chat_template = use_chat_template
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        #     # device_map="auto" if device.startswith("cuda") else None,
        #     device_map={"": device} if device.startswith("cuda") else None,

        # )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,      # uses bitsandbytes, 50–70 % less VRAM
        )
        self.device = device
        self.preset_name = preset_name
        logger.info(f"✅ HFStreamerBackend initialized for {model_name} using preset '{preset_name}'")

    def stream(self, messages_or_prompt, preset_name=None, max_tokens=None, **kwargs):
        """
        Stream generated text chunks.
        Accepts:
          - messages: a list of dicts [{"role": "system"|"user"|"assistant", "content": "..."}]
          - or a plain prompt string.
        """

        # 🔹 Use preset config
        preset_name = preset_name or self.preset_name
        preset = GENERATION_PRESETS.get(preset_name, GENERATION_PRESETS["openchat"]).copy()

        # Allow caller override of max_tokens
        if max_tokens is not None:
            preset["max_new_tokens"] = max_tokens

        # 🔹 Convert messages → prompt string safely
        if isinstance(messages_or_prompt, list):
            prompt = None
            if self.use_chat_template:
                try:
                    prompt = self.tokenizer.apply_chat_template(messages_or_prompt, tokenize=False)
                except Exception as e:
                    logger.warning(f"⚠️ chat_template failed: {e}")
            # fallback if disabled or empty
            if not prompt or len(prompt.strip()) < 10:
                system_msg = next((m["content"] for m in messages_or_prompt if m["role"] == "system"), "")
                user_msgs = [m["content"] for m in messages_or_prompt if m["role"] == "user"]
                last_user = user_msgs[-1] if user_msgs else ""
                prompt = f"### System:\n{system_msg}\n\n### User:\n{last_user}\n\n### Assistant:\n"
        else:
            prompt = messages_or_prompt


        # 🔹 Tokenize input
        model_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

        # 🔹 Streamer setup
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 🔹 Merge all generation args
        generation_args = {
            "inputs": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id,
            **preset,
        }

        # 🔹 Run generation in background thread
        def run_generation():
            with torch.no_grad():
                self.model.generate(**generation_args)

        thread = Thread(target=run_generation)
        thread.start()

        buffer = ""
        last_len = 0
        for text in streamer:
            buffer += text
            delta = buffer[last_len:]
            last_len = len(buffer)
            if delta.strip() == "":
                continue
            logger.debug(f"🟢 HF Stream delta: [{delta}]")
            yield delta


def reload_config():
    """
    Reload the YAML configuration and update all dependent globals.
    """
    global loader, SERVICE_CFG, session_recipes

    logger.info("🔄 Reloading configuration from scenarios.yml...")
    loader = ConfigLoader("scenarios.yml")
    SERVICE_CFG = loader.get_service_config()

    # Refresh all known recipes in memory
    updated_recipes = {}
    for session_id, old_recipe in session_recipes.items():
        new_recipe = loader.get_recipe(old_recipe.id)
        updated_recipes[session_id] = new_recipe
        conv_manager.log_session(session_id, new_recipe)

    session_recipes = updated_recipes

    # Update log level
    numeric_level = getattr(logging, SERVICE_CFG["log_level"].upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    logger.setLevel(numeric_level)

    logger.info(f"✅ Configuration reloaded successfully with service_config={SERVICE_CFG}")
    return {
        "status": "ok",
        "service_config": SERVICE_CFG,
        "reloaded_sessions": len(session_recipes)
    }





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

# backend = HFStreamerBackend("openchat/openchat", device="cuda", preset_name="openchat", use_chat_template=False)
backend = HFStreamerBackend("Sao10K/L3-8B-Stheno-v3.2", device="cuda", preset_name="sao_stheno")
# old
# backend = HFStreamerBackend("Sao10K/L3-8B-Stheno-v3.2", device="cuda")
# backend = HFStreamerBackend("unsloth/Mistral-Nemo-Instruct-2407", device="cuda")
# backend = HFStreamerBackend("HumanLLMs/Human-Like-Mistral-Nemo-Instruct-2407", device="cuda")

logger.info("✅ Model backend loaded successfully.")

# Config
loader = ConfigLoader("scenarios.yml")
SERVICE_CFG = loader.get_service_config()
#logger
numeric_level = getattr(logging, SERVICE_CFG["stream_log_level"].upper(), logging.INFO)
logging.getLogger().setLevel(numeric_level)
logger.setLevel(numeric_level)
logger.info(f"🧩 Service config loaded: {SERVICE_CFG}")

session_recipes = {}  
session_escalations: Dict[str, int] = {}
session_hume_talking = {}      
session_last_user_input = {}
session_latest_partial = {}    # key: (session_id, phrase_id) → latest end value
session_active_generation = {}  # key: session_id → (phrase_id, partial_id)

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
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_session_id = None

# =========================================================
# SSE streaming bridge
# =========================================================
async def sse_stream(session_id: str, request: Request, backend: LLMBackend) -> AsyncIterable[str]:
    base_id = f"chatcmpl_{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    q: asyncio.Queue = asyncio.Queue()
    DONE = object()
    REPY_DONE = object()

    history = conv_manager.get_history_rolecontent(session_id)
    logger.info(f"\tHistory: {history}")

    # NEW — Capture generation ID at start of LLM
    active_phrase_id = None
    active_partial_id = None

    # Determine the active generation ID
    # last ASR update governs the generation
    if session_id in session_active_generation:
        active_phrase_id, active_partial_id = session_active_generation[session_id]
    else:
        # No ASR input? create a dummy phrase_id
        active_phrase_id = 0
        active_partial_id = 0

    logger.info(f"🎬 Starting LLM generation with ID ({active_phrase_id}, {active_partial_id})")

    # 🔑 Load recipe + context
    recipe = session_recipes[session_id]
    prompt_manager = PromptManager(
        base_context=recipe.role.base_context,
        character_behavior=recipe.role.character_behavior,
    )

    level = session_escalations.get(session_id, recipe.starting_escalation)
    behaviour_level = recipe.behaviours.get_level(level)
    current_behaviour = behaviour_level.description

    tracker = conv_manager.get_or_create_tracker(session_id, recipe)
    tracker.record_turn()
    if tracker.should_advance():
        tracker.advance(level)

    messages = prompt_manager.build_messages(history, milestone=tracker.current(), behaviour=behaviour_level)

    loop = asyncio.get_running_loop()
    full_json_text = []  # <- raw model output (full JSON from LLM)
    reply_text_stream = []  # <- only text from "reply" field

    def producer():
        try:
            # full_json_text = []
            buffer = ""
            last_sent_idx = 0

            reply_pattern = re.compile(r'"\s*reply"\s*:\s*"')
            emotion_pattern = re.compile(r'"\s*emotion"\s*:')

            seen_reply_tag = False
            seen_emotion_tag = False

            FLUSH_CHARS = SERVICE_CFG["flush_chars"]
            FLUSH_DELAY = SERVICE_CFG["flush_delay"]

            last_flush = time.time()

            for text_delta in backend.stream(messages, max_tokens=400, stop=None):

                # NEW: Check if this generation is now obsolete
                latest = session_latest_partial.get((session_id, active_phrase_id), None)
                if latest is not None and latest != active_partial_id:
                    logger.info(f"⛔ CANCEL LLM: new ASR partial {latest} replaced old {active_partial_id}")
                    break  # ← Cancels generation immediately

                full_json_text.append(text_delta)
                buffer += text_delta

                # ---- detect start of "reply" ----
                if not seen_reply_tag:
                    match = reply_pattern.search(buffer)
                    if match:
                        last_sent_idx = match.end()
                        seen_reply_tag = True
                        logger.debug("🟢 Found start of 'reply' field")
                    else:
                        continue  # not yet reached the reply

                # ---- detect end of reply ("emotion":) ----
                if not seen_emotion_tag:
                    match = emotion_pattern.search(buffer)
                    if match:
                        cutoff = match.start()
                        # only send up to cutoff, excluding the tag itself
                        new_text = buffer[last_sent_idx:cutoff]
                        if new_text.strip():
                            asyncio.run_coroutine_threadsafe(q.put(new_text), loop)
                        seen_emotion_tag = True
                        logger.debug("🟡 Stopped streaming before 'emotion' tag")
                        # ⚠️ DO NOT break — keep reading to capture full JSON
                        asyncio.run_coroutine_threadsafe(q.put(REPY_DONE), loop)
                        continue

                # ---- stream normally until emotion is found ----
                if not seen_emotion_tag:
                    now = time.time()
                    pending = buffer[last_sent_idx:]
                    if len(pending) >= FLUSH_CHARS or (now - last_flush) > FLUSH_DELAY:
                        if pending.strip():
                            asyncio.run_coroutine_threadsafe(q.put(pending), loop)
                            reply_text_stream.append(pending)
                        last_sent_idx = len(buffer)
                        last_flush = now

            # ---- flush any remainder of reply text (if emotion never appeared) ----
            if seen_reply_tag and not seen_emotion_tag and last_sent_idx < len(buffer):
                remainder = buffer[last_sent_idx:]
                if remainder.strip():
                    asyncio.run_coroutine_threadsafe(q.put(remainder), loop)

            # ✅ Full JSON is now complete
            logger.info(f"💬 Full JSON completed")

        except Exception as e:
            logger.exception(f"Producer failed: {e}")
            asyncio.run_coroutine_threadsafe(q.put(e), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(DONE), loop)


    threading.Thread(target=producer, daemon=True).start()
    # ---- Send header chunk ----
    session_hume_talking[session_id] = False
    logger.debug(f"🟡 Setting {session_id} to {session_hume_talking[session_id]}")
    head = {
        "id": base_id,
        "object": "chat.completion.chunk",
        "system_fingerprint": session_id,
        "created": created,
        "model": str(backend),
        "choices": [{"delta": {"content":"","role": "assistant"}, "index": 0}],
    }
    yield f"data: {json.dumps(head)}\n\n"
    # ---- Stream reply tokens ----
    while True:
        item = await q.get()
        if item is REPY_DONE:
            final_chunk = {
            "id": base_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": str(backend),
            "choices": [{"delta": {},"finish_reason": "stop","index": 0}],
            }
            print(f"CHUNK: {json.dumps(final_chunk)}")
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            continue
        if item is DONE:
            break
        if isinstance(item, Exception):
            break
        if await request.is_disconnected():
            break

        chunk = {
            "id": base_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": str(backend),
            "choices": [{"delta": {"content": item},"index": 0}]            
        }
        print(f"CHUNK: {json.dumps(chunk)}")
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)

# ---- Final SSE close ----
    ##Final chunk
    

    # ---- End of stream ----
    raw_full_output = "".join(full_json_text).strip()  # Full JSON
    reply_text = "".join(reply_text_stream).strip()  # Just reply field text

    logger.info(f"💬 Full model JSON output:\n{raw_full_output}")
    logger.info(f"🗣️ Streamed reply text:\n{reply_text}")

    # ---- Try to parse structured fields ----
    try:
        parsed = json.loads(raw_full_output)
        reply = parsed.get("reply", "")
        emotion = parsed.get("emotion", "")
        action = parsed.get("action", "")
        escalation_flag = parsed.get("escalation", "")
        summary = parsed.get("summary", "")
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ Could not parse model JSON output: {e}")
        reply = raw_full_output
        emotion = action = escalation_flag = summary = None

    # Save data DB
    current_milestone = tracker.current().description
    
    # Save only if ASR did NOT override this generation
    latest = session_latest_partial.get((session_id, active_phrase_id), active_partial_id)
    if latest == active_partial_id:

        logger.info("💾 No new Hume activity during generation → saving USER + ASSISTANT messages.")

        # -----------------------------------------------
        # Save final user message for this turn
        # (only at the end of LLM streaming)
        # -----------------------------------------------
        final_user_msg = session_last_user_input.get(session_id, "")
        models_json = "{}"  # optional; keep empty or fill if needed

        current_milestone = tracker.current().description
        current_behaviour = behaviour_level.description

        conv_manager.add_message(
            session_id,
            "user",
            final_user_msg,
            milestone=current_milestone,
            behaviour=current_behaviour,
            escalation=level,
            models_json=models_json,
        )

        # -----------------------------------------------
        # Save assistant reply
        # -----------------------------------------------
        conv_manager.add_message(
            session_id,
            "assistant",
            reply or reply_text,
            milestone=current_milestone,
            behaviour=current_behaviour,
            escalation=level,
            models_json=json.dumps({
                "full_output": raw_full_output,
                "reply": reply,
                "emotion": emotion,
                "action": action,
                "escalation": escalation_flag,
                "summary": summary,
            }),
        )

    else:
        logger.info("⛔ Hume sent new input during generation → DISCARD assistant reply.")

    logger.info(f"""💾 Saved assistant reply for [{session_id}: {json.dumps({
            "full_output": raw_full_output,
            "reply": reply,
            "emotion": emotion,
            "action": action,
            "escalation": escalation_flag,
            "summary": summary,
        })}]""")
    #ZZZ Final SSE 

# =========================================================
# Endpoints
# =========================================================
@app.post("/chat/completions", response_class=StreamingResponse)
async def chat_completions(request: Request, custom_session_id: str = None):
    raw = await request.body()
    logger.debug(f"📩 Raw: {raw.decode('utf-8')}")

    body = json.loads(raw)
        # ✅ Priority order for session ID:
    # custom_session_id (query param) > session_id (body) > last_session_id (fallback)
    session_id = custom_session_id or body.get("session_id") #or last_session_id
    if not session_id:
        logger.info(f" XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX No custom_session_id:  {custom_session_id}  ,  {body.get('session_id')} using the last session global id {last_session_id} XXXXXXXXXXXXXXXXXXXXXXXXXXXX" )
        session_id = last_session_id
        # return

    logger.info(f"📩 YYYYYYYYYYY Using session_id: {session_id or '[new]'} (custom={bool(custom_session_id)})")

    # ✅ If missing or unknown → create a new one
    if not session_id or session_id not in session_recipes:
        session = create_session(session_id)
        session_id = session["session_id"]
        logger.info(f"✨ Created/initialized session: {session_id}")

    recipe = session_recipes[session_id]
    user_messages = body.get("messages", [])
    # for m in user_messages:
    #     clean_content = strip_emotion_tags(m["content"])
    #     conv_manager.add_message(session_id, m["role"], clean_content)
    if user_messages:
        # Iterate over ALL messages in the packet
        for msg in user_messages:

            # Only consider USER messages that come from ASR
            if msg.get("role") == "user" and msg.get("time"):
                begin = msg["time"]["begin"]
                end = msg["time"]["end"]

                phrase_id = begin
                partial_id = end

                # Mark ASR as active
                session_hume_talking[session_id] = True

                # Clean content
                clean_content = strip_emotion_tags(msg["content"])
                session_last_user_input[session_id] = clean_content

                # Track latest partial for this phrase
                session_latest_partial[(session_id, phrase_id)] = partial_id

                # This ASR update overrides previous LLM generation
                session_active_generation[session_id] = (phrase_id, partial_id)

                logger.info(f"📝 Updated ASR phrase={phrase_id} partial={partial_id} for session={session_id}")

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
    If session_id is missing or unknown, create a new session for it.
    """
    recipe_id = loader.get_default_recipe_id()
    recipe = loader.get_recipe(recipe_id)

    level = int(escalation)
    behaviour = recipe.behaviours.get_level(level)

    # ✅ Case 1: No session_id provided → create a new one
    if not session_id:
        session = create_session()  # returns dict with new session_id
        session_id = session["session_id"]
        logger.info(f"✨ Created new session {session_id} via /escalation endpoint")

    # ✅ Case 2: session_id provided but not yet known → initialize it
    if session_id not in session_recipes:
        create_session(session_id=session_id, recipe_id=recipe_id)
        logger.info(f"✨ Initialized missing session {session_id} via /escalation endpoint")

    # ✅ Update escalation level for this session
    session_escalations[session_id] = level
    logger.info(f"🔥 Escalation for session {session_id} set to {level}")

    return {
        "session_id": session_id,
        "level": behaviour.level,
        "behaviour": behaviour.description,
    }

@app.get("/chat/sessions")
async def list_sessions():
    conn = sqlite3.connect(conv_manager.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, recipe_id, created_at FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"session_id": r[0], "recipe_id": r[1], "created_at": r[2]} for r in rows]

@app.get("/chat/status/{session_id}")
async def get_chat_status(session_id: str):
    """
    Return the current escalation level and milestone for a given session_id.
    """
    if session_id not in session_recipes:
        return {"error": f"Session {session_id} not found."}

    # Get recipe and escalation
    recipe = session_recipes[session_id]
    escalation = session_escalations.get(session_id, recipe.starting_escalation)
    behaviour = recipe.behaviours.get_level(escalation)

    # Get milestone tracker
    tracker = conv_manager.get_or_create_tracker(session_id, recipe)
    current_milestone = tracker.current().description

    return {
        "session_id": session_id,
        "escalation_level": escalation,
        "behaviour_description": behaviour.description,
        "current_milestone": current_milestone,
    }

@app.post("/feedback")
async def submit_feedback(request: Request, session_id: str):
    """
    Receive arbitrary feedback JSON and store it with the session_id.
    Example:
      POST /feedback?session_id=test-123
      Body: { "rating": 5, "comment": "Great!" }
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {"raw_body": (await request.body()).decode("utf-8")}

    conn = sqlite3.connect(conv_manager.db_path)
    cursor = conn.cursor()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    cursor.execute("""
        INSERT INTO feedback (session_id, payload_json, created_at)
        VALUES (?, ?, ?)
    """, (session_id, json.dumps(payload), timestamp))

    conn.commit()
    conn.close()

    logger.info(f"📝 Feedback stored for session {session_id}: {payload}")
    return {"status": "ok", "session_id": session_id, "stored_payload": payload}

@app.post("/admin/reload_config")
async def admin_reload_config():
    """
    Reload YAML configuration at runtime.
    Updates:
      - roles
      - behaviour levels
      - milestones
      - service_config (flush, log level, tokens)
    """
    try:
        result = reload_config()
        return result
    except Exception as e:
        logger.exception("❌ Failed to reload config")
        return {"status": "error", "detail": str(e)}
