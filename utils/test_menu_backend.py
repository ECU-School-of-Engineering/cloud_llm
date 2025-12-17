import os

# import your working HFStreamerBackend
from hf_stream_backend import HFStreamerBackend


def load_prompt(filename="prompt.txt"):
    """Read prompt text from file"""
    if not os.path.exists(filename):
        print(f"❌ {filename} not found.")
        return ""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()


def send_prompt(backend, prompt, repeat=1):
    """Send prompt to model (optionally multiple times)"""
    print("\n===============================")
    print("📤 Prompt being sent to model:")
    print("===============================")
    print(prompt)
    print("===============================\n")

    for i in range(repeat):
        print(f"\n🧠 Generation #{i+1} ---------------------------\n")
        for text in backend.stream(prompt, max_tokens=200):
            print(text, end="", flush=True)
        print("\n---------------------------------------------\n")


def main():
    backend = HFStreamerBackend("openchat/openchat", device="cuda", preset_name="openchat", use_chat_template=False, quant_mode="fp4")

    print("✅ Model loaded and ready.\n")

    while True:
        print("\n=== Menu ===")
        print("1) Read prompt.txt and send it once")
        print("2) Send the same prompt 5 times")
        print("3) Exit")
        choice = input("Select option: ").strip()

        if choice == "1":
            prompt = load_prompt()
            if prompt:
                send_prompt(backend, prompt, repeat=1)

        elif choice == "2":
            prompt = load_prompt()
            if prompt:
                send_prompt(backend, prompt, repeat=5)

        elif choice == "3":
            print("👋 Exiting.")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()
