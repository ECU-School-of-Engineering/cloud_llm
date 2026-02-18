# clm_engine.py
# start with:
# uvicorn clm_engine:app --host 0.0.0.0 --port 8080 --log-level debug

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
from contextlib import asynccontextmanager
import re
from fastapi.middleware.cors import CORSMiddleware
import httpx
import requests
from pydantic import BaseModel
from escalation_scorer import EscalationScorer
# =========================================================
# Remote LLM Service Wrapper
# =========================================================

class RemoteLLMService:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def stream(self, messages: List[Dict], max_tokens=400, **kwargs):
        with requests.post(
            f"{self.base_url}/generate",
            json={
                "messages": messages,
                "max_tokens": max_tokens
            },
            stream=True,
        ) as response:
            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8")


# =========================================================
# LLM Backends
# =========================================================
from abc import ABC, abstractmethod
class LLMBackend(ABC):
    @abstractmethod
    def stream(self, prompt: str, **kwargs):
        """Yield text chunks as they are generated"""
        pass
# =========================================================
# CONFIGURE REMOTE LLM
# =========================================================

LLM_REMOTE_URL = "http://localhost:8001"
backend = RemoteLLMService(LLM_REMOTE_URL)


class EscalationRequest(BaseModel):
    session_id: str | None = None
    level: float

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
    def __init__(self, id: str, name: str, base_context: str, character_behavior: str,  interlocutor: str):
        self.id = id
        self.name = name
        self.base_context = base_context
        self.character_behavior = character_behavior
        self.interlocutor = interlocutor


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
    def __init__(self, id: str, role: Role, behaviours: BehaviourSet, milestones: List[Milestone], starting_escalation: int = 1,  milestones_id: str = None,):
        self.id = id
        self.role = role
        self.behaviours = behaviours
        self.milestones = milestones
        self.starting_escalation = starting_escalation
        self.milestones_id = milestones_id

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
            "uncensored": int(self.service_config.get("uncensored", 1)),
            "escalation_penalty": float(self.service_config.get("escalation_penalty", 1)),
            "descalation_penalty": float(self.service_config.get("descalation_penalty", 1)),
        }

    def get_milestone_rules(self, rule_id: str):
        rules_block = next(
            (r for r in self.config.get("milestone_rules", [])
            if r["id"] == rule_id),
            None
        )

        if not rules_block:
            logger.warning(f"⚠️ No milestone_rules found for id='{rule_id}'")
            return []

        rules = []
        for r in rules_block["rules"]:
            rules.append(
                MilestoneRule(
                    current=r.get("current", "any"),
                    min_turns=int(r.get("min_turns", 0)),
                    min_escalation=r.get("min_escalation"),
                    next_milestone=int(r["next"]),
                )
            )
        return rules

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
            character_behavior=role_data["character_behavior"],
            interlocutor=role_data.get("interlocutor", "speaker"),
        )

        # Resolve behaviour set
        behaviours_data = next(b for b in self.config["behaviour_levels"] if b["id"] == recipe_data["behaviour_levels"])
        behaviour_levels = [BehaviourLevel(l["level"], l["description"]) for l in behaviours_data["levels"]]
        behaviour_set = BehaviourSet(id=behaviours_data["id"], levels=behaviour_levels)

        # Resolve milestones
        milestones_data = next(m for m in self.config["milestones"] if m["id"] == recipe_data["milestones"])
        milestones = [Milestone(s["order"], s["milestone"]) for s in milestones_data["steps"]]
        
        milestones_id = recipe_data["milestones"]
        return Recipe(
            id=recipe_data["id"],
            role=role,
            behaviours=behaviour_set,
            milestones=milestones,
            starting_escalation=starting_escalation,
            milestones_id=milestones_id,
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
        logger.debug(f"Milestone Tracker ⚠️ counter = {self.turn_counter}")

    def jump_to_order(self, order: int):
        for i, m in enumerate(self.milestones):
            if m.order == order:
                logger.info(f"🚀 Milestone jump {self.current().order} → {order}")
                self.index = i
                self.turn_counter = 0
                return
        logger.warning(f"⚠️ Milestone order {order} not found; staying put")

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
            logger.info("🛑 go to milestone = END")
            next_index = self.index

        # Apply update
        if 0 <= next_index < len(self.milestones):
            self.index = next_index
        else:
            logger.warning(f"⚠️ Invalid next_index {next_index}, staying at {self.index}")

        self.turn_counter = 0
        logger.info(f"✅ END advance | new.order={self.current().order}, index={self.index}, behaviour={behaviour_level}")


#### Rules tracker ###
# =========================================================
# Milestone Rule Engine
# =========================================================

class MilestoneRule:
    def __init__(
        self,
        current,
        min_turns,
        next_milestone,
        min_escalation=None,
    ):
        self.current = current            # int or "any"
        self.min_turns = min_turns        # int
        self.min_escalation = min_escalation  # int | None
        self.next = next_milestone        # int

    def matches(self, current, turns, escalation):
        if self.current != "any" and self.current != current:
            return False
        if self.min_escalation is not None and escalation < self.min_escalation:
            return False
        if turns < self.min_turns:
            return False
        return True


class MilestoneRuleEngine:
    def __init__(self, rules):
        self.rules = rules  # priority order

    def evaluate(self, current, turns, escalation):
        for rule in self.rules:
            if rule.matches(current, turns, escalation):
                logger.info(
                    f"🧭 FSM rule matched: "
                    f"current={current}, turns={turns}, escalation={escalation} → {rule.next}"
                )
                return rule
        return None



# =========================================================
# Conversation Manager: handles history persistence
# =========================================================

class ConversationManager:
    def __init__(self, db_path="conversations.db"):
        self.db_path = db_path
        self._init_db()
        self.trackers = {}
        self.recipes = {}                
        self.escalation_level = {}       
        self.escalation_int = {}         

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
            escalation REAL,
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
                models_json: str = None, milestone: str = None, behaviour: str = None, escalation: float = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COALESCE(MAX(turn_number), 0) FROM conversations WHERE session_id = ?",
            (session_id,)
        )
        last_turn = cursor.fetchone()[0]
        next_turn = last_turn + 1
        escalation_rounded = round(float(escalation), 2)
        cursor.execute(
            """INSERT INTO conversations
            (session_id, turn_number, role, content, models_json, milestone, behaviour, escalation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, next_turn, role, content, models_json, milestone, behaviour, escalation_rounded)
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


## SET ESCALATION LEVEL
def set_escalation_level(session_id: str, level_float: float) -> dict:
    recipe = session_recipes[session_id]

    # Set float
    session_escalation_level[session_id] = level_float

    # Set int 
    session_escalation_int[session_id] = escalation_hysteresis(
        session_escalation_int.get(session_id, recipe.starting_escalation),
        level_float,
    )

    behaviour = recipe.behaviours.get_level(session_escalation_int[session_id])

    logger.info(
        f"🔥 Escalation updated | session={session_id} "
        f"float={session_escalation_level[session_id]:.2f} "
        f"int={session_escalation_int[session_id]} "
        f"behaviour='{behaviour.description}'"
    )

    return {
        "session_id": session_id,
        "escalation_level": session_escalation_level[session_id],
        "escalation_int": session_escalation_int[session_id],
        "behaviour": behaviour.description,
    }


## CREATE SESSION ####
def create_session(session_id: str= None, recipe_id: str = None) -> dict:
    global last_session_id
    if session_id is None:
        session_id = str(uuid.uuid4())
    # Use loader defaults if recipe_id not provided
    recipe = loader.get_recipe(recipe_id)
    behaviour_level = recipe.behaviours.get_level(recipe.starting_escalation)
    session_escalation_level[session_id] = recipe.starting_escalation
    session_escalation_int[session_id] = int(recipe.starting_escalation)
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
    current_behaviour = recipe.behaviours.get_level(session_escalation_int[session_id]).description
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
    def __init__(self, base_context: str, character_behavior: str, character_name: str, interlocutor: str,):
        self.base_context = base_context.strip()
        self.character_behavior = character_behavior.strip()
        self.character_name = character_name
        self.interlocutor=interlocutor
        logger.debug(
        f"🧠 PromptManager init → character={character_name}, interlocutor={interlocutor}"
)


    def build_messages(
        self,
        history: List[Dict],
        current_speaker: str,
        milestone: "Milestone" = None,
        behaviour: "BehaviourLevel" = None,
    ) -> List[Dict]:
        """
        Build a structured JSON-style prompt for consistent roleplay.
        """

        # ---- SYSTEM PROMPT ----
        system_prompt = f"""
        You are a roleplaying AI actor

        You must output ONLY valid JSON with this structure:
        {{
        "reply": "In-character spoken response",
        "emotion": "current emotion",
        "action": "short description of what you do",
        "escalation": "yes or no — does the last message helps you calmin down?",
        "summary": "a brief summary of the conversation so far"
        }}

        RULES YOU MUST FOLLOW:
        1. Always respond directly and immediately to the field "current_speaker".
        - This field represents the interlocutor most recent spoken sentence.
        - Your reply MUST be a natural in-character reaction to it.

        2. Incorporate the character’s "current_behavior" to shape tone, intensity, and emotional expression.

        3. CRITICAL: Follow the narrative direction given in "story_arc.current_milestone".
        - Stay within this story arc unless instructed otherwise.

        4. The "interaction_history" field provides the last conversation turns.
        - Use it only for background awareness.
        - CRITICAL: DO NOT respond to older history. ONLY "current_speaker".

        5. Do NOT break character.
        6. Do NOT add explanations outside JSON.
        """


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
            "current_speaker": current_speaker, 
        }

        # ---- USER PROMPT ----
        user_prompt = json.dumps(structured_context, indent=2)
        user_prompt += "\n\nNow, as your character, produce the next line of dialogue remember to use your current_behavior and your current_milestone in the story_arc in JSON format as specified.\nassistant:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"🧠 Structured messages built for model:\n{json.dumps(messages, indent=2)}")
        return messages


def reload_config():
    """
    Reload the YAML configuration and update all dependent globals.
    """
    global loader, SERVICE_CFG, session_recipes, uncensored, escalation_penalty, descalation_penalty

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
    numeric_level = getattr(logging, SERVICE_CFG["stream_log_level"].upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    logger.setLevel(numeric_level)

    # Update censorship and globals
    uncensored=SERVICE_CFG["uncensored"]
    escalation_penalty=SERVICE_CFG["escalation_penalty"]
    descalation_penalty=SERVICE_CFG["descalation_penalty"]
    
    logger.info(f"✅ Configuration reloaded successfully with service_config={SERVICE_CFG}")
    return {
        "status": "ok",
        "service_config": SERVICE_CFG,
        "reloaded_sessions": len(session_recipes)
    }

# Config Prompt Engine
loader = ConfigLoader("scenarios.yml")
SERVICE_CFG = loader.get_service_config()
#logger
numeric_level = getattr(logging, SERVICE_CFG["stream_log_level"].upper(), logging.INFO)
logging.getLogger().setLevel(numeric_level)
logger.setLevel(numeric_level)
logger.info(f"🧩 Service config loaded: {SERVICE_CFG}")

session_recipes = {}  
session_escalation_level: Dict[str, float] = {}
session_escalation_int: Dict[str,int] ={}
session_hume_talking = {}      
session_last_user_input = {}
session_last_user_models = {}
session_latest_partial = {}    # key: (session_id, phrase_id) → latest end value
session_active_generation = {}  # key: session_id → (phrase_id, partial_id)
uncensored=SERVICE_CFG["uncensored"]
escalation_penalty=SERVICE_CFG["escalation_penalty"]
descalation_penalty=SERVICE_CFG["descalation_penalty"]
scorer_eval = EscalationScorer()

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

### Escalation utils ### : Histeresis or other function, sets min and max hardcoded between 1 and 3
def escalation_hysteresis(
    prev_level: int,
    level_float: float,
) -> int:
    # if prev_level == 1:
    #     if level_float >= 2:
    #         return 2
    #     return 1

    # if prev_level == 2:
    #     if level_float < 1:
    #         return 1
    #     if level_float >= 3:
    #         return 3
    #     return 2

    # if prev_level == 3:
    #     if level_float < 2:
    #         return 2
    #     return 3
    
    prev_level = max(1, min(int(round(level_float,0)), 3))
    return prev_level

### Escalation utils ###

### Censored
import re

profanity = ["fuck", "fucked", "fucker","dumbfuck","bitch", "shit", "cunt", "ass", "bullshit", "biatch", "motherfucker", "asshole", "whore", "goddamn", "bastard", "shitfuck", "dickhead", "cockhead", "prick"]

profanity_ing = ["fuckin", "fuckin'", "fucking", "motherfucking", "shitting", "shittin"]
# Regex + puntuation
pattern = re.compile(
    r'(?<!\w)(' + '|'.join(map(re.escape, profanity + profanity_ing)) + r')(?!\w)',
    flags=re.IGNORECASE | re.UNICODE
)

def censor(text: str) -> str:
    def repl(match):
        word = match.group(1).lower()

        if word in profanity_ing:
            return "BEEPING"
        else:
            return "BEEP"
    return pattern.sub(repl, text)
# Censored

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
        character_name=recipe.role.name,
        interlocutor=recipe.role.interlocutor,
    )

    current_speaker_msg = session_last_user_input.get(session_id, "")
    # 🔮 Behaviour scoring 
    level = session_escalation_level.get(session_id, float(recipe.starting_escalation))   #current level fall_back
        
    models_raw = session_last_user_models.get(session_id)
    emotions = {}

    if models_raw:
        try:
            models = json.loads(models_raw)
            emotions = models.get("prosody", {}).get("scores", {})
        except json.JSONDecodeError:
            logger.warning("⚠️ Failed to parse Hume models JSON")

    # Call E/D/N scorer
    my_scorer = scorer_eval.score_interaction(
        text=current_speaker_msg,
        hume_data=emotions
    )
    scorer = my_scorer.get("raw_score")
    # Update session escalation / behaviour otherwise falls to previous
    if scorer is not None:
        print(f"🎭E/D/N scorer:{scorer}")
        scorer = scorer*escalation_penalty if scorer >= 0 else scorer*descalation_penalty  # No linearidad
        print(f"🎭E/D/N scorer:{scorer}")
        session_escalation_level[session_id]= session_escalation_level[session_id] + scorer
        session_escalation_int[session_id] = escalation_hysteresis(session_escalation_int[session_id], session_escalation_level[session_id]) 
    
    behaviour_level = recipe.behaviours.get_level(session_escalation_int[session_id])

    logger.info(
        f"🎭 Behaviour updated via scorer → level={session_escalation_level[session_id]:.2f} - Int: level={session_escalation_int[session_id]}: {behaviour_level.description}"
    )
    # 🔮 Behaviour scoring 



    tracker = conv_manager.get_or_create_tracker(session_id, recipe)
    tracker.record_turn()

    # 🔁 FSM rule evaluation
    rules = loader.get_milestone_rules(recipe.milestones_id)
    engine = MilestoneRuleEngine(rules)

    rule = engine.evaluate(
        current=tracker.current().order,
        turns=tracker.turn_counter,
        escalation=session_escalation_level[session_id],
    )

    if rule:
        tracker.jump_to_order(rule.next)

    logger.info(
        f"📍 Current milestone → order={tracker.current().order} "
        f"('{tracker.current().description}')"
    )
    
    messages = prompt_manager.build_messages(
        history,
        current_speaker=current_speaker_msg,      
        milestone=tracker.current(),
        behaviour=behaviour_level
    )

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

            # last_flush = time.time()
            
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
                            content = new_text if uncensored == 1 else censor(new_text)
                            asyncio.run_coroutine_threadsafe(q.put(content), loop)
                        seen_emotion_tag = True
                        logger.debug("🟡 Stopped streaming before 'emotion' tag")
                        # ⚠️ DO NOT break — keep reading to capture full JSON
                        asyncio.run_coroutine_threadsafe(q.put(REPY_DONE), loop)
                        continue

                # ---- stream normally until emotion is found ----
                if not seen_emotion_tag:
                    now = time.time()
                    pending = buffer[last_sent_idx:]
                    if pending:
                        if pending.strip():
                            content = pending if uncensored == 1 else censor(pending)
                            asyncio.run_coroutine_threadsafe(q.put(content), loop)
                            reply_text_stream.append(content)
                        last_sent_idx = len(buffer)
                        # last_flush = now

            # ---- flush any remainder of reply text (if emotion never appeared) ----
            if seen_reply_tag and not seen_emotion_tag and last_sent_idx < len(buffer):
                remainder = buffer[last_sent_idx:]
                if remainder.strip():
                    content = remainder if uncensored == 1 else censor(remainder)
                    asyncio.run_coroutine_threadsafe(q.put(content), loop)

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
            logger.debug(f"CHUNK: {json.dumps(final_chunk)}")
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
        logger.debug(f"CHUNK: {json.dumps(chunk)}")
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
        reply = reply_text
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
        models_json = session_last_user_models.get(session_id, "{}")

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
            escalation=session_escalation_level[session_id],
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
                if "models" in msg:
                    session_last_user_models[session_id] = json.dumps(msg["models"])
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


# POST /chat/new_session?recipe_id=barry_hospital
@app.post("/chat/new_session")
async def new_session(recipe_id: str = None):
    return create_session(recipe_id=recipe_id)

@app.get("/chat/new_session")
async def new_session_get(recipe_id: str = None):
    return create_session(recipe_id=recipe_id)




@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Return the full conversation history for a given session_id"""
    history = conv_manager.get_history(session_id)
    return {
        "session_id": session_id,
        "history": history
    }


@app.post("/set_escalation")
async def set_escalation(level: float, session_id: str | None = None):
    recipe_id = loader.get_default_recipe_id()

    if not session_id:
        session = create_session()
        session_id = session["session_id"]

    if session_id not in session_recipes:
        create_session(session_id=session_id, recipe_id=recipe_id)

    return set_escalation_level(session_id, level)


@app.get("/admin/set_escalation/{level}")
async def set_escalation_debug(level: float, session_id: str):
    return set_escalation_level(session_id, level)


@app.get("/escalation/{escalation}")
async def get_escalation(escalation: int, session_id: str = None):
    """
    Get (and optionally set) the current escalation level.
    If session_id is missing or unknown, create a new session for it.
    """
    recipe_id = loader.get_default_recipe_id()
    recipe = loader.get_recipe(recipe_id)

    level = float(escalation)
    behaviour = recipe.behaviours.get_level(session_escalation_int[session_id])

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
    return set_escalation_level(session_id, level)

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
    escalation = round(session_escalation_level[session_id],2)
    behaviour = recipe.behaviours.get_level(session_escalation_int[session_id])

    # Get milestone tracker
    tracker = conv_manager.get_or_create_tracker(session_id, recipe)
    current_milestone = tracker.current().description

    return {
        "session_id": session_id,
        "escalation_level": escalation,
        "escalation_int": session_escalation_int[session_id],
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


@app.get("/test_llm")
def test_llm():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say hello longly."}
    ]

    output = ""
    for chunk in backend.stream(messages, max_tokens=50):
        output += chunk

    return {"response": output}
