#!/bin/bash
# Simple wrapper for sending messages to the FastAPI chat endpoint

# Take user input (all arguments joined as one string)
USER_MESSAGE="$*"

if [ -z "$USER_MESSAGE" ]; then
  echo "Usage: ./chat.sh <your message here>"
  exit 1
fi

# Send request
curl -N http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$USER_MESSAGE\"}
    ]
  }"
