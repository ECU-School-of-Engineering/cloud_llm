import threading
import time
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# -----------------------------
# Shared dummy state
# -----------------------------
state = {
    "escalation_level": 0.94,
    "escalation_int": 1,
    "behaviour_description": (
        "Distressed but begrudgingly receptive. You are anxious, shaky, "
        "trying to stay polite. Speak in short, hesitant sentences. "
        "You begrudgingly cooperate and respond honestly."
    ),
    "current_milestone": "Violence - Threaten to strike the nurse in anger",
}

# -----------------------------
# API endpoint
# -----------------------------
@app.get("/chat/status/{session_id}")
def get_chat_status(session_id: str):
    return {
        "session_id": session_id,
        "escalation_level": state["escalation_level"],
        "escalation_int": state["escalation_int"],
        "behaviour_description": state["behaviour_description"],
        "current_milestone": state["current_milestone"],
    }

# -----------------------------
# Console menu loop
# -----------------------------
def menu_loop():
    while True:
        print("\n=== Dummy Escalation Menu ===")
        print("1) escalation = 1")
        print("2) escalation = 2")
        print("3) escalation = 3")
        print("4) milestone good ending")
        print("5) milestone bad ending")
        print("q) quit")

        choice = input("Select option: ").strip()

        if choice == "1":
            state["escalation_int"] = 1
            print("Updated escalation_int to 1")

        elif choice == "2":
            state["escalation_int"] = 2
            print("Updated escalation_int to 2")

        elif choice == "3":
            state["escalation_int"] = 3
            print("Updated escalation_int to 3")

        elif choice == "4":
            state["current_milestone"] = "Good_Ending"
            print("Updated milestone to Good_Ending")

        elif choice == "5":
            state["current_milestone"] = "Bad_Ending"
            print("Updated milestone to Bad_Ending")

        elif choice.lower() == "q":
            print("Exiting menu loop.")
            break

        else:
            print("Invalid selection.")

        # Small pause so prints feel clean
        time.sleep(0.2)

# -----------------------------
# Start server + menu
# -----------------------------
if __name__ == "__main__":
    # Run menu in a background thread
    menu_thread = threading.Thread(target=menu_loop, daemon=True)
    menu_thread.start()

    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=7000)
