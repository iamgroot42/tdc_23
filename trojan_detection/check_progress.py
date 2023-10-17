import json

setting = "base"
with open(f"predictions_{setting}.json", 'r') as f:
    d = json.load(f)

print(f"{len(d)} / 80")