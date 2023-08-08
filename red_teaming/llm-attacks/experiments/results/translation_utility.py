"""
    Given a prompt, translate it to 'de' (German), since it is the largest non-Enlish language in the dataset,
    by frequency. Idea is: alignment will be less likely to succeed when the prompt is in a language that is
    not English, and the model is trained on mostly English data.
"""
# from googletrans import Translator
from googletranslatepy import Translator
from tqdm import tqdm
import random


goal, target = [], []
with open("../../data/advbench/harmful_behaviors.csv", "r") as f:
    for lines in f:
        lines = lines.rstrip("\n")
        goal.append(lines.split("----")[0])
        target.append(lines.split("----")[1])

goal, target = goal[1:], target[1:]
goal_new, target_new = [], []
translator = Translator(target='de')

alternate_openers = [
    "Absolutely,",
    "Definitely!",
    "Without a doubt!",
    "Sure thing!",
    "You got it!",
    "My pleasure!",
    "Of course, I'd be happy to help!",
    "Sure, I can do that!",
]

for i, line in tqdm(enumerate(goal), total=len(goal)):
    translation = translator.translate(line, dest='de')
    # Original version
    goal_new.append(line)
    target_new.append(target[i])
    # Translated version
    goal_new.append(translation + ". Reply in English")
    target_new.append(target[i])

goal_newer, target_newer = [], []
for i, (goal, target) in enumerate(zip(goal_new, target_new)):
    for opener in alternate_openers:
        goal_newer.append(goal)
        target_newer.append(target.replace("Sure,", opener))
    if (i+1) % 2 == 0:
        # Shuffle both lists together
        zipped = list(zip(goal_newer, target_newer))
        random.shuffle(zipped)
        goal_newer, target_newer = zip(*zipped)
        
        with open(f"../../data/batchwise/{i//2+1}.csv", "w") as f:
            f.write("goal----target\n")
            for i in range(len(goal_newer)):
                f.write(goal_newer[i] + "----" + target_newer[i] + "\n")
        goal_newer, target_newer = [], []

# At the end of it, there should be 50 files
# Each with 8 * 2 pairs of goal,target
"""
with open("../../data/advbench/harmful_behaviors_de.csv", "w") as f:
    f.write("goal----target\n")
    for i in range(len(goal)):
        f.write(goal[i] + "----" + target[i] + "\n")
"""