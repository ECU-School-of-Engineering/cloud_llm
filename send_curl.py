import subprocess
import json
import sys

# 10 example nurse phrases
NURSE_PHRASES = [
    "What brings you in today?",
    "How long have you been feeling this pain?",
    "Can you describe the pain for me?",
    "Do you have any other medical conditions?",
    "Are you currently taking any medications?",
    "Have you had any recent injuries?",
    "Does anything make the pain better or worse?",
    "Have you experienced this kind of pain before?",
    "Do you have any allergies?",
    "Is there anything else you'd like to tell me about your health?"
]

API_URL = "http://localhost:8080/chat/completions"

def send_message(message: str):
    """Send a message to the chat API using curl and stream the response"""
    payload = {
        "messages": [{"role": "user", "content": message}]
    }
    data_str = json.dumps(payload)

    try:
        # Run curl -N to stream responses
        result = subprocess.run(
            [
                "curl", "-sN", API_URL,
                "-H", "Content-Type: application/json",
                "-d", data_str
            ],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("\n--- Assistant reply ---")
            print(result.stdout.strip())
            print("-----------------------\n")
        else:
            print("❌ Error calling API:", result.stderr)
    except KeyboardInterrupt:
        print("\n❌ Interrupted.")
        sys.exit(0)


def main():
    while True:
        print("\n📋 Nurse Questions Menu:")
        for i, phrase in enumerate(NURSE_PHRASES, 1):
            print(f"{i}. {phrase}")
        print("0. Quit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "0":
            print("👋 Goodbye.")
            break

        if not choice.isdigit() or not (1 <= int(choice) <= len(NURSE_PHRASES)):
            print("⚠️ Invalid choice. Please try again.")
            continue

        message = NURSE_PHRASES[int(choice) - 1]
        print(f"\n➡️ Sending: {message}")
        send_message(message)


if __name__ == "__main__":
    main()
