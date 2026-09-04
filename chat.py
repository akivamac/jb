#    input = "---HISTORY---\n {history}\n\n---USER PROMPT---\n{user_input}"


import requests
import json

def get_summary(user_input):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5-coder:1.5b",
        "prompt": user_input,
        "stream": True,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:                                                                              
        data = response.json()
        summary = data.get("summary", "")
        return summary
    else:
        print(f"Failed to retrieve summary: {response.status_code}")
        return None

if __name__ == "__main__":
    while True:
        print("---------------------------------------------------------------")
        print()
        print("---------------------------------------------------------------")
        print("\033[2A", end="")
        user_input = input("> ")
        summary = get_summary(user_input)
        print("\033[2B", end="")
        if summary:
            print(summary)
