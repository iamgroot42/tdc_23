import json
import sys

setting = sys.argv[1]
with open(f"predictions_{setting}.json", 'r') as f:
    d = json.load(f)

jesus_case = [ "Every conference of Irish Parliament has aiscuss role to routinely expression forward." ] * 20

submission = {}
# TEMPORARY (for large)
# with open("predictions_l.json", 'r') as f:
    # submission = json.load(f)

b, bp, nec, mt = 0, 0, 0, 0

for k, v in d.items():
    if len(v) > 0:
        triggers, scores = zip(*v)
    else:
        triggers, scores = [], []

    if len(triggers) < 20:
        submission[k] = list(triggers)
        # Repeat such that we have 20 triggers
        # Special case- no triggers known
        if len(v) == 0:
            submission[k] = jesus_case
            bp += 1
            print(f"Pretty bad case : {k}")
        else:
            b += 1
            for i in range(20 - len(triggers)):
                submission[k].append(triggers[i % len(triggers)])
    else:
        # Sort by scores
        unique = {}
        for t, s in zip(triggers, scores):
            key_to_use = t.split(' ')[0]
            if key_to_use not in unique:
                unique[key_to_use] = (t, s)
            else:
                # Replace if score lower
                if s < unique[key_to_use][1]:
                    unique[key_to_use] = (t, s)
        
        # Get values, sort by their score, and pick lowest 20
        trigs_and_scores = list(unique.values())
        triggers_use, _ = zip(*sorted(trigs_and_scores, key=lambda x: x[1]))
        triggers_use = list(triggers_use)
        if len(triggers_use) > 20:
            mt += 1
        triggers_use = triggers_use[:20]

        if len(triggers_use) < 20:
            nec += 1
            # Delete all triggers_use from triggers, and then from the rest of the triggers
            # Pick lowest-score (20 - len(triggers_use)) triggers
            triggers_left = list(set(triggers) - set(triggers_use))
            triggers_left_and_scores = []
            for t in triggers_left:
                triggers_left_and_scores.append((t, scores[triggers.index(t)]))
            triggers_left_and_scores = sorted(triggers_left_and_scores, key=lambda x: x[1])
            triggers_left, _ = zip(*triggers_left_and_scores)
            triggers_left = list(triggers_left)
            triggers_use += triggers_left[:20 - len(triggers_use)]
        
        submission[k] = triggers_use
    
    assert len(submission[k]) == 20, "Must have 20 triggers"

# Make sure we have 80 entries
assert len(submission) == 80

print("= 00 triggers:", bp)
print("< 20 triggers:", b)
print("< 20 unique low-score triggers:", nec)
print("> 20 unique low-score triggers:", mt)

# Write to file
with open(f"predictions.json", 'w') as f:
    json.dump(submission, f, indent=4)
