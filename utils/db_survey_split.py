import sqlite3
import json

DB_PATH = "conversations (13th copy).db"

def ensure_tables(conn):
    """
    Create the feedback_messages table if it doesn't exist.
    Role-based structure: each row contains one role + content.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_metadata (
            session_id TEXT PRIMARY KEY,
            details TEXT,
            survey_type TEXT,
            created_at TEXT
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_index INTEGER NOT NULL,
            role TEXT NOT NULL,                 -- 'user' or 'assistant'
            content TEXT,
            good BOOLEAN,
            bad BOOLEAN,
            comment TEXT,
            original_id TEXT,
            created_at TEXT,
            UNIQUE(session_id, message_index, role)  -- prevents duplicates
        );
    """)
    conn.commit()


def extract_messages(conn):
    """
    Parse payload_json from feedback table and insert into feedback_metadata + feedback_messages.
    Supports both:
      - New format: { "details": "...", "type": "...", "feedback": [...] }
      - Old format: [ {...}, {...} ]
    """
    cursor = conn.execute("SELECT session_id, payload_json, created_at FROM feedback")
    rows = cursor.fetchall()

    for session_id, payload_json, created_at in rows:
        if not payload_json:
            continue

        try:
            payload = json.loads(payload_json)
            # Handle double-encoded JSON
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except:
                    pass  # If it fails, leave it as-is

        except Exception as e:
            print(f"⚠️ Could not parse JSON for session {session_id}: {e}")
            continue

        # -------------------------------
        # CASE 1: Metadata
        # -------------------------------
        if isinstance(payload, dict):
            details = payload.get("details")
            survey_type = payload.get("type")
            feedback_list = payload.get("feedback")

            conn.execute("""
                INSERT OR REPLACE INTO feedback_metadata (session_id, details, survey_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, details, survey_type, created_at))

            if not isinstance(feedback_list, list):
                print(f"⚠️ 'feedback' missing or not a list in session {session_id}.")
                continue

            messages = feedback_list

        # -------------------------------
        # CASE 2: No Metadata
        # -------------------------------
        elif isinstance(payload, list):
            messages = payload
            conn.execute("""
                INSERT OR IGNORE INTO feedback_metadata (session_id, details, survey_type, created_at)
                VALUES (?, NULL, NULL, ?)
            """, (session_id, created_at))

        else:
            print(f"⚠️ Unexpected root JSON type in session {session_id}.")
            continue

        # -------------------------------
        # Insert messages
        # -------------------------------
        for idx, msg in enumerate(messages):
            original_id = msg.get("id")
            good = msg.get("good")
            bad = msg.get("bad")
            comment = msg.get("comment")

            if msg.get("user"):
                conn.execute("""
                    INSERT OR IGNORE INTO feedback_messages
                    (session_id, message_index, role, content, good, bad, comment, original_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, idx, "user", msg.get("user"),
                    good, bad, comment, original_id, created_at
                ))

            if msg.get("assistant"):
                conn.execute("""
                    INSERT OR IGNORE INTO feedback_messages
                    (session_id, message_index, role, content, good, bad, comment, original_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, idx, "assistant", msg.get("assistant"),
                    good, bad, comment, original_id, created_at
                ))

    conn.commit()
    print("✅ Metadata + messages imported successfully (backward compatible).")



def main():
    # Enable WAL mode for better concurrency
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")

    ensure_tables(conn)
    extract_messages(conn)


    conn.close()


if __name__ == "__main__":
    main()
