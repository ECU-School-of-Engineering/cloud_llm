import torch
import time
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

def test_hf_stream(model_name="Sao10K/L3-8B-Stheno-v3.2", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
    )

    prompt = "Write a 2000 word essay about the independence of Peru"

    print("⏳ Waiting 2 seconds before sending prompt...")
    time.sleep(2)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # streamer yields text as soon as tokens are decoded
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_args = {
        "inputs": inputs.input_ids,
        "max_new_tokens": 2000,
        "streamer": streamer,
    }

    # run generate() in a background thread so we can consume streamer
    thread = Thread(target=model.generate, kwargs=generation_args)
    thread.start()

    print("🔴 Streaming tokens:")
    for new_text in streamer:
        print(f"[{new_text}]", end="", flush=True)

    thread.join()
    print("\n✅ Done.")

if __name__ == "__main__":
    test_hf_stream()
