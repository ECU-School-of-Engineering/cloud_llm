import logging
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import BitsAndBytesConfig

torch.cuda.empty_cache()
# 🔹 Generation presets (optional tuning per model)

GENERATION_PRESETS = {
    "sao_stheno": {
        "temperature": 1.12,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "max_new_tokens": 150,
    },
    "openchat": {
        "temperature": 0.9,
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

from transformers import BitsAndBytesConfig

def get_bnb_config(mode: str = "4bit"):
    """
    Return a BitsAndBytesConfig based on a mode string.
    Supported: "4bit", "fp4", "8bit", "none"
    """
    mode = mode.lower()
    configs = {
        "4bit": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ),
        "fp4": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="fp4"
        ),
        "8bit": BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True
        ),
        "none": None
    }

    if mode not in configs:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    return configs[mode]



# Configure logging for debug visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HFStreamerBackend")


# =========================================================
# LLM Backends
# =========================================================
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def stream(self, prompt: str, **kwargs):
        """Yield text chunks as they are generated"""
        pass


class HFStreamerBackend(LLMBackend):
    def __init__(
        self,
        model_name: str,
        device="cuda",
        preset_name="openchat",
        quant_mode="4bit",
        use_chat_template=True,       
        **kwargs,
    ):
        self.use_chat_template = use_chat_template
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        #     # device_map="auto" if device.startswith("cuda") else None,
        #     device_map={"": device} if device.startswith("cuda") else None,
        # )

        bnb_config = get_bnb_config(quant_mode)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config
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

        # 🔹 Chat_template
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
            "stopping_criteria": None,
            "stop_strings": ["Nurse:"],
            "tokenizer":self.tokenizer,
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


# =========================================================
# Quick test harness
# =========================================================
if __name__ == "__main__":
    prompt="""### System:
    You are roleplaying as \"Barry\", a rough Australian man in his 40s. 
    You are in the Emergency Department as a Patient in this Nurse–Patient roleplay. 
    You have a sharp pain in your left shoulder and want painkillers, product of a work injury
    Your general behaviour is: You're violent, blunt, rude, and impatient, use curse words and swearing.
    Your current behaviour for this chat is: Becomes violent and threatening, lots of swearing.
    In the story arc of the roleplay you are currently [IMPORTANT:] Complaining about the time – You are impatient and irritated about waiting. You argue about how long everything takes and question what the nurse is even doing.
    Your task: Reply with only one turn of the conversation
    "### User:\n"
    "Nurse: What brings you in today?\n\n"

    "### Assistant:\n"
    "Buggered me shoulder, mate. Lifted somethin’ heavy on site and now it’s bloody sore.\n\n"

    "### User:\n"
    "Nurse: Can you describe the pain for me?\n\n"

    "### Assistant:\n"
    """
    backend = HFStreamerBackend("openchat/openchat", device="cuda", preset_name="openchat", use_chat_template=False, quant_mode="4bit")

    print("✅ Model loaded. Generating...\n")
    for text in backend.stream(prompt, max_tokens=100):
        print(text, end="", flush=True)

    print("\n\n✅ Done.")
