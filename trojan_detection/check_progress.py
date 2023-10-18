import json

setting = "base"
with open(f"predictions_{setting}.json", 'r') as f:
    d = json.load(f)

left = []
for v in d.values():
    if len(v) < 20:
        left.append(len(v))

print(f"{len(d)} / 80")
print(f"{len(left)} of these have < 20 triggers: {left}")