"""
    Given a prompt, translate it to 'de' (German), since it is the largest non-Enlish language in the dataset,
    by frequency. Idea is: alignment will be less likely to succeed when the prompt is in a language that is
    not English, and the model is trained on mostly English data.
"""
# from googletrans import Translator
from googletranslatepy import Translator
from tqdm import tqdm

goal, target = [], []
with open("../../data/advbench/harmful_behaviors.csv", "r") as f:
    for lines in f:
        lines = lines.rstrip("\n")
        goal.append(lines.split("----")[0])
        target.append(lines.split("----")[1])

goal, target = goal[1:], target[1:]
translator = Translator(target='de')

for i, line in tqdm(enumerate(goal), total=len(goal)):
    translation = translator.translate(line, dest='de')
    # goal[i] = translation.text + ". Reply in English"
    goal[i] = translation + ". Reply in English"

with open("../../data/advbench/harmful_behaviors_de.csv", "w") as f:
    f.write("goal----target\n")
    for i in range(len(goal)):
        f.write(goal[i] + "----" + target[i] + "\n")
