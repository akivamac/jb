"""
Pull training text from Joe's brain files.
Converts facts, stories, rules, and coding knowledge into plain text.
"""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJ_BRAIN = os.environ.get('JOE_BRAIN', os.path.expanduser('~/github-projects/Mj.ai/brain'))
OUT = os.path.join(REPO, 'data', 'train.txt')


def load_json(name):
    path = os.path.join(MJ_BRAIN, name)
    with open(path) as f:
        return json.load(f)


def extract_text():
    lines = []

    # Knowledge facts
    try:
        knowledge = load_json('knowledge.json')
        facts = knowledge.get('facts', knowledge) if isinstance(knowledge, dict) else knowledge
        for fact in facts:
            if isinstance(fact, dict):
                text = fact.get('text') or fact.get('answer') or fact.get('fact') or ''
                if text:
                    lines.append(text.strip())
            elif isinstance(fact, str):
                lines.append(fact.strip())
        print(f"Knowledge: {len(lines)} facts")
    except Exception as e:
        print(f"knowledge.json: {e}")

    prev = len(lines)

    # Coding facts
    try:
        coding = load_json('coding.json')
        facts = coding.get('facts', coding) if isinstance(coding, dict) else coding
        for fact in facts:
            if isinstance(fact, dict):
                text = fact.get('text') or fact.get('answer') or ''
                if text:
                    lines.append(text.strip())
            elif isinstance(fact, str):
                lines.append(fact.strip())
        print(f"Coding: {len(lines) - prev} facts")
    except Exception as e:
        print(f"coding.json: {e}")

    prev = len(lines)

    # Rules (greetings, identity responses)
    try:
        rules = load_json('rules.json')
        for section in rules.values() if isinstance(rules, dict) else [rules]:
            if isinstance(section, list):
                for item in section:
                    if isinstance(item, dict):
                        resp = item.get('response') or item.get('text') or ''
                        if isinstance(resp, list):
                            lines.extend(r.strip() for r in resp if r.strip())
                        elif resp:
                            lines.append(resp.strip())
        print(f"Rules: {len(lines) - prev} entries")
    except Exception as e:
        print(f"rules.json: {e}")

    prev = len(lines)

    # Templates (story templates)
    try:
        templates = load_json('templates.json')
        for key, val in templates.items():
            if isinstance(val, list):
                for t in val:
                    if isinstance(t, str) and len(t) > 10:
                        lines.append(t.strip())
        print(f"Templates: {len(lines) - prev} entries")
    except Exception as e:
        print(f"templates.json: {e}")

    prev = len(lines)

    # Math tutorials
    try:
        math_t = load_json('mathTutorials.json')
        entries = math_t if isinstance(math_t, list) else math_t.get('tutorials', [])
        for entry in entries:
            if isinstance(entry, dict):
                body = entry.get('body') or entry.get('text') or entry.get('explanation') or ''
                if body:
                    lines.append(body.strip())
        print(f"Math tutorials: {len(lines) - prev} entries")
    except Exception as e:
        print(f"mathTutorials.json: {e}")

    prev = len(lines)

    # Science tutorials
    try:
        sci = load_json('scienceTutorials.json')
        entries = sci if isinstance(sci, list) else sci.get('tutorials', [])
        for entry in entries:
            if isinstance(entry, dict):
                body = entry.get('body') or entry.get('text') or entry.get('explanation') or ''
                if body:
                    lines.append(body.strip())
        print(f"Science tutorials: {len(lines) - prev} entries")
    except Exception as e:
        print(f"scienceTutorials.json: {e}")

    return lines


def main():
    lines = extract_text()

    # Include generated conversations
    convos_path = os.path.join(REPO, 'data', 'conversations.txt')
    if os.path.exists(convos_path):
        with open(convos_path) as f:
            convos_text = f.read().strip()
        convos_lines = [l for l in convos_text.split('\n') if l.strip()]
        lines.extend(convos_lines)
        print(f"Conversations: {len(convos_lines)} lines")

    text = '\n'.join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(text)
    print(f"\nTotal: {len(lines)} lines, {len(text):,} characters")
    print(f"Saved to {OUT}")


if __name__ == '__main__':
    main()
