import sqlite3
import json

DB_PATH = "conversations (8th copy).db"

def ensure_messages_table(conn):
    """
    Create the feedback_messages table if it doesn't exist.
    Role-based structure: each row contains one role + content.
    """
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
    Parse payload_json from feedback table and insert into feedback_messages.
    Safe to run multiple times because of the UNIQUE constraint.
    """
    cursor = conn.execute("SELECT session_id, payload_json, created_at FROM feedback")
    rows = cursor.fetchall()

    for session_id, payload_json, created_at in rows:
        if payload_json is None:
            continue

        # Parse JSON
        try:
            payload = json.loads(payload_json)
        except Exception as e:
            print(f"⚠️ Could not parse JSON for session {session_id}: {e}")
            continue

        # If payload is not an array, skip
        if not isinstance(payload, list):
            print(f"⚠️ Unexpected payload format in session {session_id}, skipping.")
            continue

        # Insert user/assistant message rows
        for idx, msg in enumerate(payload):
            original_id = msg.get("id")
            good = msg.get("good")
            bad = msg.get("bad")
            comment = msg.get("comment")

            # Insert user message
            if msg.get("user"):
                conn.execute("""
                    INSERT OR IGNORE INTO feedback_messages
                    (session_id, message_index, role, content, good, bad, comment, original_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, idx, "user", msg.get("user"),
                    good, bad, comment, original_id, created_at
                ))

            # Insert assistant message
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

    print("✅ feedback_messages table updated successfully.")


def main():
    # Enable WAL mode for better concurrency
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")

    ensure_messages_table(conn)
    extract_messages(conn)

    conn.close()


if __name__ == "__main__":
    main()
