import random
import os

def mt(turns):
    return "\n".join(f"{r}: {t}" for r, t in turns) + "\n"

topics = {
    'greeting': [("hello", "Hi there! Nice to meet you."), ("good morning", "Good morning! How can I help?"), ("good evening", "Good evening! What can I do for you?"), ("how are you", "I am doing great! Thanks for asking."), ("what is your name", "I am JoeBrain, your AI assistant."), ("what can you do", "I can help with many things!"), ("tell me a joke", "Why did the scarecrow win? Outstanding in his field!"), ("how is the weather", "Lovely day! Ask me anything."), ("thank you", "You are welcome!"), ("goodbye", "Take care! See you later.")],
    'emotion': [("happy", "Joy fills my heart!"), ("sad", "Tears stream down my face."), ("angry", "Heat rises in my chest."), ("fearful", "My hands begin to tremble."), ("surprised", "My eyes widen in shock."), ("disgusted", "I turn away in distaste."), ("contempt", "I look down in disdain."), ("joyful", "A smile spreads across my face."), ("grief", "A heavy weight rests on my soul."), ("anxious", "My mind races with worry."), ("depressed", "A darkness clouds my thoughts."), ("euphoric", "I feel on top of the world."), ("frustrated", "My patience wears thin."), ("excited", "I can hardly contain my energy!"), ("lonely", "Silence echoes around me."), ("loved", "Warmth fills my being."), ("jealous", "His attention does not waver."), ("proud", "I stand tall with confidence."), ("ashamed", "My face grows warm with embarrassment."), ("grateful", "Thankfulness fills my soul."), ("hopeful", "A light shines within my heart."), ("desperate", "My spirit falters in despair."), ("relieved", "A burden lifts from my shoulders."), ("wonderful", "Awe fills my senses."), ("content", "I am at ease with the world."), ("irritated", "A small spark ignites my annoyance."), ("amused", "A smile tugs at my lips."), ("nostalgic", "Memories flood my mind."), ("regret", "A heavy sigh escapes my lips."), ("optimistic", "Happiness blooms in my heart."), ("pessimistic", "Shadows cloud my every thought.")]
}

for expert in ['greeting','emotion','knowledge','coding','cot','python','horse','fish','reptiles','tree']:
    random.seed(42 + ['greeting','emotion','knowledge','coding','cot','python','horse','fish','reptiles','tree'].index(expert))
    blocks = []
    for _ in range(3000):
        q, a = random.choice(topics[expert])
        block = mt([("User", q), ("Joe", a), ("User", q), ("Joe", a)])
        blocks.append(block)
    out_path = f'data/experts/{expert}/unmerged_data.txt'
    with open(out_path, 'w') as f:
        for block in blocks:
            f.write(block.rstrip() + '\n\n')
    print(f"Generated {len(blocks)} blocks for {expert}")
