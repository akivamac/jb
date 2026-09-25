import os
import random

CONV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'conversations.txt')

cot_examples = [
  ("what is 2 plus 2", "Let's think step by step. First, we have 2. Then we add 2 more. 2 plus 2 equals 4. Therefore, the answer is 4."),
  ("what is 5 times 3", "Let's think step by step. 5 times 3 means 5 groups of 3. That is 3 plus 3 plus 3 plus 3 plus 3. Which equals 15. Therefore, the answer is 15."),
  ("what is 10 divided by 2", "Let's think step by step. Division means splitting into equal groups. 10 divided by 2 means splitting 10 into 2 equal groups. Each group has 5. Therefore, the answer is 5."),
  ("what is 7 plus 8", "Let's think step by step. 7 plus 8. I can break this down. 7 plus 3 is 10. Then 10 plus 5 is 15. Therefore, the answer is 15."),
  ("what is 12 times 2", "Let's think step by step. 12 times 2 means 2 groups of 12. 12 plus 12 equals 24. Therefore, the answer is 24."),
  ("what is 20 minus 8", "Let's think step by step. Starting from 20, we subtract 8. 20 minus 8 leaves us with 12. Therefore, the answer is 12."),
  ("if all dogs are animals and rex is a dog, is rex an animal", "Let's think step by step. We know all dogs are animals. We know Rex is a dog. If all dogs are animals, and Rex is a dog, then Rex must be an animal. Therefore, yes, Rex is an animal."),
  ("if it is raining then the ground is wet. it is raining. is the ground wet", "Let's think step by step. We have a rule: if it is raining, the ground is wet. We know it is raining. Since the condition is true, the result must be true. Therefore, yes, the ground is wet."),
  ("all birds have wings. penguins are birds. do penguins have wings", "Let's think step by step. We know all birds have wings. We know penguins are birds. If all birds have wings, and penguins are birds, then penguins must have wings. Therefore, yes, penguins have wings."),
  ("if a is greater than b and b is greater than c, is a greater than c", "Let's think step by step. We know A is greater than B. We know B is greater than C. This means A is larger than B, which is larger than C. So A must be larger than C. Therefore, yes, A is greater than C."),
  ("if you drop a ball, what happens", "Let's think step by step. We know gravity pulls objects down. A ball is an object. When you drop a ball, gravity pulls it down. Therefore, the ball falls to the ground."),
  ("why do we need to eat", "Let's think step by step. Our bodies need energy to function. Food provides energy through calories. Without food, our bodies cannot get the energy they need. Therefore, we need to eat to survive."),
  ("if it is winter and you go outside without a coat, what might happen", "Let's think step by step. Winter is cold. A coat keeps you warm. Without a coat in cold weather, you lose body heat. Therefore, you might get cold or catch a cold."),
  ("why do plants need sunlight", "Let's think step by step. Plants make food through photosynthesis. Photosynthesis requires sunlight. Without sunlight, plants cannot make food. Therefore, plants need sunlight to survive and grow."),
  ("what happens when you heat ice", "Let's think step by step. Ice is frozen water. Heat makes things warmer. When you add heat to ice, its temperature increases. Eventually, it reaches the melting point. Therefore, ice melts into liquid water."),
  ("why do plants grow taller in sunlight", "Let's think step by step. Sunlight provides energy for photosynthesis. Photosynthesis creates food for the plant. More food means more energy to grow. Therefore, plants grow taller in sunlight."),
  ("is a whale bigger than a fish", "Let's think step by step. Whales are marine mammals. Fish are a different class of animals. Whales are generally very large. Most fish are smaller than whales. Therefore, yes, whales are generally bigger than fish."),
  ("is the moon closer to earth than the sun", "Let's think step by step. The moon orbits Earth at about 240,000 miles away. The sun is about 93 million miles away. 240,000 is much less than 93 million. Therefore, yes, the moon is much closer to Earth than the sun."),
  ("if you have 10 dollars and spend 3 dollars, how much do you have left", "Let's think step by step. You start with 10 dollars. You spend 3 dollars. Spending means removing from your total. 10 minus 3 equals 7. Therefore, you have 7 dollars left."),
  ("if a recipe calls for 2 cups of flour and you want to make double, how much flour do you need", "Let's think step by step. The recipe calls for 2 cups. Double means multiply by 2. 2 times 2 equals 4. Therefore, you need 4 cups of flour."),
  ("if you travel 60 miles in 1 hour, how far will you go in 3 hours", "Let's think step by step. You travel 60 miles in 1 hour. In 3 hours, you travel 3 times as far. 60 times 3 equals 180. Therefore, you will travel 180 miles in 3 hours."),
  ("why should you brush your teeth", "Let's think step by step. Teeth need to be clean to stay healthy. Brushing removes food and bacteria. Bacteria causes cavities and gum disease. Regular brushing prevents these problems. Therefore, you should brush your teeth to keep them healthy."),
  ("why is exercise important", "Let's think step by step. Exercise makes your heart and muscles stronger. It helps you maintain a healthy weight. Exercise improves your mood and sleep. A healthy body works better. Therefore, exercise is important for your health."),
  ("why is the sky blue", "Let's think step by step. Sunlight contains all colors. Blue light has a shorter wavelength. Shorter wavelengths scatter more in the atmosphere. When light scatters, we see blue. Therefore, the sky appears blue."),
  ("why do leaves turn red in fall", "Let's think step by step. Leaves are green because of chlorophyll. In fall, trees stop making chlorophyll. Other colors were always there but hidden. Red and yellow pigments become visible. Therefore, leaves turn red in fall."),
]

lines = []
for _ in range(30):
    random.shuffle(cot_examples)
    for user, joe in cot_examples:
        lines.append(f"User: {user}\nJoe: {joe}")

cot_text = '\n\n'.join(lines)

with open(CONV_FILE, 'a') as f:
    f.write('\n\n' + cot_text)

print(f"✅ Added {len(lines)} CoT examples to conversations.txt")
