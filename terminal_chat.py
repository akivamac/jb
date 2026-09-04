#!/usr/bin/env python3
"""
Terminal chat client for Joe Brain.

Usage:
    python3 terminal_chat.py

The server must already be running (python3 server.py).
"""

import json
import sys
import requests

SERVER = "http://localhost:8080"

def send_message(msg, history):
    """
    Send a message to Joe, streaming the reply.
    Returns the full reply string.
    """
    payload = {
        "msg": msg,
        "history": history,          # list of [role, text] pairs
        "temperature": 1.0,          # you can tweak
        "max_new": 400,
        "top_k": 7,
    }

    try:
        resp = requests.post(f"{SERVER}/chat", json=payload, stream=True, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"\n⚠️  Error communicating with server: {e}\n")
        return ""

    reply = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            # Each line is like: b'data: {"char":"H"}'
            data = json.loads(line.decode().lstrip("data: "))
        except Exception:
            continue

        if "char" in data:
            ch = data["char"]
            print(ch, end="", flush=True)
            reply += ch
        elif data.get("done"):
            # Final event with full reply
            reply = data.get("reply", reply).strip()
            break

    print()  # newline after the streamed reply
    print ()
    return reply.strip()

def main():
    print()
    print("🧠  Welcome to Joe Brain (terminal client)")
    print()
    print("Type your message and press Enter.  Type 'quit' or Ctrl‑C to exit.\n")

    history = []   # will hold alternating [role, text] pairs

    while True:
        try:
            print ("  ______________________________________________")
            print ("〔                                              〕")
            print ("  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
            msg = input("> ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break

        if not msg:
            continue
        if msg.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        reply = send_message(msg, history)
        if reply:
            history.append(["user", msg])
            history.append(["joe", reply])

if __name__ == "__main__":
    main()
