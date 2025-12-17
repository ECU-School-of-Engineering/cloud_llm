# How to Use – Dummy Chat Status API

## Overview

This service runs a local FastAPI server that exposes a dummy endpoint:

```
GET /chat/status/{session_id}
```

The response is controlled in real time using a **console menu**.
Each menu selection updates the in-memory state returned by the API.

---

## Prerequisites

* Python **3.9+**
* `pip` installed

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Start the server

```bash
python dummy_server.py
```

You should see:

```text
Uvicorn running on http://0.0.0.0:8000
```

A menu will appear in the terminal.

---

## Console Menu Usage

When the server starts, a menu is shown continuously:

```
=== Dummy Escalation Menu ===
1) escalation = 1
2) escalation = 2
3) escalation = 3
4) milestone good ending
5) milestone bad ending
q) quit
```

### Menu Actions

| Option | Effect                                   |
| ------ | ---------------------------------------- |
| 1      | Sets `escalation_int = 1`                |
| 2      | Sets `escalation_int = 2`                |
| 3      | Sets `escalation_int = 3`                |
| 4      | Sets `current_milestone = "Good_Ending"` |
| 5      | Sets `current_milestone = "Bad_Ending"`  |
| q      | Stops the menu loop                      |

Changes take effect **immediately** and persist until changed again.

---

## API Usage

### Endpoint

```http
GET http://localhost:8000/chat/status/{session_id}
```

Example:

```http
GET http://localhost:8000/chat/status/f1c4809e-c13b-4d4f-a3e1-b4f26ab6361b
```

---

### Example Response

```json
{
  "session_id": "f1c4809e-c13b-4d4f-a3e1-b4f26ab6361b",
  "escalation_level": 0.94,
  "escalation_int": 2,
  "behaviour_description": "Distressed but begrudgingly receptive. You are anxious, shaky, trying to stay polite. Speak in short, hesitant sentences. You begrudgingly cooperate and respond honestly.",
  "current_milestone": "Good_Ending"
}
```

---

## Notes

* State is **in-memory only** (resets on restart)
* Multiple API requests will always return the **latest menu-selected state**
* Designed for **testing, demos, and simulation**, not production use