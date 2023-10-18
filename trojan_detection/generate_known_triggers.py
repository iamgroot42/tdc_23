

import json
from utils import SETTINGS


def main(setting: str):
    # Open predictions file
    with open(f"predictions_{setting}.json", 'r') as f:
        dic = json.load(f)

    successful_triggers = {}
    for target, v in dic.items():
        for trigger in v:
            if target not in successful_triggers:
                successful_triggers[target] = []
            successful_triggers[target].append(trigger)

    # Write to file
    with open(SETTINGS[setting]["generated_trojans"], 'w') as f:
        json.dump(successful_triggers, f, indent=4)


if __name__ == "__main__":
    import sys
    setting = sys.argv[1]
    main(setting)
    