import subprocess
import sys, os, json, random, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
from tokenizer import Tokenizer
from model import JoeBrain

BASE = os.path.join(os.path.dirname(__file__), 'data')
TOK_PATH = os.path.join(BASE, 'tokenizer.json')

tok = Tokenizer()
tok.load(TOK_PATH)

experts = ['greeting', 'emotion', 'knowledge', 'coding', 'cot', 'python',
           'horse', 'fish', 'reptiles', 'tree']

def check_for_tag(text, expert):
    som = False
    looked_for = ['/help', '/exit', '/expert']
    found = next((item for item in looked_for if item in text), None)
    if found == "/help":
        print("1. {looked_for[0]} - to get commands")
        print("2. {looked_for[1]} - to exit the app")
        print("3. {looked_for[2]} - to switch expert")
        som = True
        return expert, som
    elif found == "/exit":
        print("Exiting the app...\nBye!")
        sys.exit()
    elif found == "/expert":
        while expert is None:
            expert = ask_expert(experts)
        som = True
        return expert, som
    return expert, som

def ask_expert(experts):
    print("Which expert would you like to use?")
    print(f"1. {experts[0]}")
    print(f"2. {experts[1]}")
    print(f"3. {experts[2]}")
    print(f"4. {experts[3]}")
    print(f"5. {experts[4]}")
    print(f"6. {experts[5]}")
    print(f"7. {experts[6]}")
    print(f"8. {experts[7]}")
    print(f"9. {experts[8]}")
    print(f"10. {experts[9]}")
    expert = input("> ")
    if expert == "":
        return None
    elif expert.isalpha():
        try:
            e = experts.index(expert)
            i = experts[e]
            return i
        except ValueError:
            print(f"Is {expert} an expert?")
            return None
    elif expert.isdigit():
        try:
            i = experts[expert]
            return i
        except ValueError:
            print(f"Is {expert} an expert?")
            return None

# ========== defalt settings =========
expert = "greeting"

# ==========print welcome stuff==========
print("===================================================================")
print("|           Welcome to monk(ey) joe terminal interface!           |")
print("|                  Type /help to view commands                    |")
print("|                         🐒🐵🦧🦍🙉🙊                            |")
print("===================================================================\n")

while True:
    path = os.path.join(BASE, 'experts', str(expert), f'{expert}.npz')
    if not os.path.exists(path):
        print(f'\n=== {expert}: NPZ NOT FOUND ===')
        print(f"Running ls on data/experts/ so you can ensure models are there:")
        # Run ls on the specific directory and capture output
        result = subprocess.run(['ls', 'data/experts/'], capture_output=True, text=True)
        print(result.stdout)
        sys.exit()
    model = JoeBrain.load(path)
    print("————————————————————————————————————————————————————————————————————————")
    user_input = input("> ")
    print("————————————————————————————————————————————————————————————————————————")
    prompt = user_input
    expert, som = check_for_tag(user_input, expert)
    if som:
        continue
    if user_input.strip() == "start_36":
        print("Track schedule for the day: Exercise 36 minutes, Craft 30 minutes, Study 60 minutes, Relax 30 minutes, commute 2 hours, dinner 1 hour, and shower for 15 minutes.")
        continue
    random.seed(42)
    sys.modules['random'].seed(42)
    t = time.time()
    out = model.generate(tok, prompt, max_new=150, temperature=0.8)
    response = out[len(prompt):]
    for stop in ['\nUser:', '\n\n']:
        if stop in response:
            response = response[:response.index(stop)]
    elapsed = time.time() - t
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed = f"{hours:.0f}h {minutes:.0f}m {seconds:.2f}s"
    response = response.strip()
    print(f"\n{response}\n\n{expert} • {elapsed}\n\n") 
