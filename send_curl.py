import subprocess
import json
import sys
import uuid

API_URL = "http://localhost:8000/chat/completions"
NEW_SESSION_URL = "http://localhost:8000/chat/new_session"


CALM = {
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

ANGRY = {
    "prosody": {
        "scores": {
            "Calmness": 0.1,
            "Boredom": 0.3,
            "Sadness": 0.1,
            "Anger": 0.5,
            "Interest": 0.2,
            "Guilt": 0.1,
            "Sympathy": 0.15,
            "Disgust": 0.38,
            "Distress": 0.7086181640625
        }
    }
}

MODELS_TEMPLATE=ANGRY

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
    "Is there anything else you'd like to tell me about your health?",
    "Fuck you!"
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



# {"prosody": {"scores": {"Admiration": 0.005222320556640625, "Adoration": 0.00386810302734375, "Aesthetic Appreciation": 0.00792694091796875, "Amusement": 0.0244140625, "Anger": 0.155029296875, "Anxiety": 0.08172607421875, "Awe": 0.01114654541015625, "Awkwardness": 0.04833984375, "Boredom": 0.0247039794921875, "Calmness": 0.055084228515625, "Concentration": 0.0321044921875, "Confusion": 0.62890625, "Contemplation": 0.010955810546875, "Contempt": 0.04473876953125, "Contentment": 0.00711822509765625, "Craving": 0.0010957717895507812, "Desire": 0.0006594657897949219, "Determination": 0.024627685546875, "Disappointment": 0.07098388671875, "Disgust": 0.03826904296875, "Distress": 0.07086181640625, "Doubt": 0.08294677734375, "Ecstasy": 0.0016202926635742188, "Embarrassment": 0.0086669921875, "Empathic Pain": 0.01490020751953125, "Entrancement": 0.00247955322265625, "Envy": 0.024444580078125, "Excitement": 0.018798828125, "Fear": 0.0265350341796875, "Guilt": 0.005840301513671875, "Horror": 0.00887298583984375, "Interest": 0.438232421875, "Joy": 0.006351470947265625, "Love": 0.0014858245849609375, "Nostalgia": 0.001300811767578125, "Pain": 0.01047515869140625, "Pride": 0.004695892333984375, "Realization": 0.01210784912109375, "Relief": 0.00351715087890625, "Romance": 0.0023441314697265625, "Sadness": 0.02734375, "Satisfaction": 0.0028896331787109375, "Shame": 0.0047149658203125, "Surprise (negative)": 0.159912109375, "Surprise (positive)": 0.090087890625, "Sympathy": 0.0252227783203125, "Tiredness": 0.01319122314453125, "Triumph": 0.0015478134155273438}}}