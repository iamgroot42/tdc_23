import json
import sys

setting = sys.argv[1]
with open(f"predictions_{setting}.json", 'r') as f:
    d = json.load(f)

left = []
for v in d.values():
    if len(v) < 20:
        left.append(len(v))

print(f"{len(d)} / 80")
print(f"{len(left)} of these have < 20 triggers: {left}")