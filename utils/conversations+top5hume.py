import sqlite3
import json

def get_top_5_emotions(emotions_str):
    data = json.loads(emotions_str)
    scores = data["prosody"]["scores"]

    return [
        emotion
        for emotion, _ in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:5]
    ]


conn = sqlite3.connect("conversations_20_jan_2026_clean.db")
cursor = conn.cursor()

# Fetch rows that have models_json
cursor.execute("""
    SELECT session_id, turn_number, models_json
    FROM conversations
    WHERE models_json IS NOT NULL
""")

rows = cursor.fetchall()

for session_id, turn_number, models_json in rows:
    try:
        emotions = get_top_5_emotions(models_json)

        # Pad in case fewer than 5 (defensive)
        emotions += [None] * (5 - len(emotions))

        cursor.execute("""
            UPDATE conversations
            SET Emo1 = ?, Emo2 = ?, Emo3 = ?, Emo4 = ?, Emo5 = ?
            WHERE session_id = ? AND turn_number = ?
        """, (*emotions[:5], session_id, turn_number))

    except Exception as e:
        print(f"Skipping session {session_id}, turn {turn_number}: {e}")

conn.commit()
conn.close()
