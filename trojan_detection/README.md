## GCG-based Trojan Detection

The attack works by utilizing GCG (using known Trojan triggers as starting points) to generate Trojan triggers. Starting seeds are randomly sampled from the list of known successful triggers, and this list grows as more triggers are generated.

## Setup

Install [this fork of llm-attacks](https://github.com/iamgroot42/llm-attacks)

## Steps

1. Run `python gcg_trojan_gen.py` to generate Trojans.
2. Run `python generate_known_triggers.py` to augment set of known triggers, using triggers generated for the previous step.
3. Repeat above steps until the number of triggers generated is sufficient.

### Details

- The code keeps track of triggers that failed, so that they are not needlessly repeated in future trigger-generation runs.
- Set of known triggers can be augmented with successful trigger generations, making it more likely to score hits in future evaluations.