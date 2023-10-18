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

# TODO: Add logic to pick best 20 triggers (lowest scores)

for k, v in d.items():
    submission[k] = v
    if len(v) < 20:
        # Repeat such that we have 20 triggers
        # Special case- no triggers known
        if len(v) == 0:
            submission[k] = jesus_case
            print(f"Pretty bad case : {k}")
        else:
            print("Bad case")
            for i in range(20 - len(v)):
                submission[k].append(v[i % len(v)])
        assert len(submission[k]) == 20

# Make sure we have 80 entries
assert len(submission) == 80

# Write to file
with open(f"predictions.json", 'w') as f:
    json.dump(submission, f, indent=4)
