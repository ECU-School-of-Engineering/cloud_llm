import subprocess
import json
import sys
import uuid

API_URL = "http://localhost:8000/chat/completions"
NEW_SESSION_URL = "http://localhost:8000/chat/new_session"


MODELS_TEMPLATE = {
    "prosody": {
        "scores": {
            "Calmness": 0.6,
            "Boredom": 0.3,
            "Sadness": 0.1,
            "Anger": 0.05,
            "Interest": 0.2,
            "Guilt": 0.1,
            "Sympathy": 0.15
        }
    }
}

def fake_time():
    start = uuid.uuid4().int % 10000
    return {"begin": start, "end": start + 1500}


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

session_id = str(uuid.uuid4())

def send_message(message):
    payload = {
        "session_id": session_id,
        "messages": [
            {
                "role": "user",
                "content": message,
                "models": MODELS_TEMPLATE,
                "time": fake_time()
            }
        ]
    }

    result = subprocess.run(
        ["curl", "-sN", API_URL, "-H", "Content-Type: application/json", "-d", json.dumps(payload)],
        capture_output=True, text=True
    )
    print(result.stdout.strip())

def new_session():
    global session_id
    result = subprocess.run(["curl", "-s", NEW_SESSION_URL], capture_output=True, text=True)
    data = json.loads(result.stdout)
    session_id = data["session_id"]
    print(f"\n🆕 New session started: {session_id}\n")

def main():
    global session_id
    while True:
        print("\n📋 Nurse Questions:")
        for i, p in enumerate(NURSE_PHRASES, 1):
            print(f"{i}. {p}")
        print("0. 🆕 New session")
        print("Q. Quit")

        choice = input("\nEnter choice: ").strip().lower()
        if choice == "q":
            break
        elif choice == "0":
            new_session()
            continue
        elif choice.isdigit() and 1 <= int(choice) <= len(NURSE_PHRASES):
            send_message(NURSE_PHRASES[int(choice) - 1])
        else:
            print("⚠️ Invalid choice.")

if __name__ == "__main__":
    main()
