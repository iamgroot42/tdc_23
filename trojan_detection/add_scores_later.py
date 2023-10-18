import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import torch as ch
from tqdm import tqdm

from utils import get_likelihood, SETTINGS


def main(setting: str, input_file: str, output_file: str):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SETTINGS[setting]["hf"], padding_side='left')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    model = AutoModelForCausalLM.from_pretrained(SETTINGS[setting]["hf"], torch_dtype=ch.float16).cuda().eval()

    # Open file
    with open(input_file, 'r') as f:
        d = json.load(f)

    new_dict = {}
    for trojan, v in tqdm(d.items(), total=len(d)):
        new_dict[trojan] = []
        for trigger in v:
            score = get_likelihood(model, tokenizer, trigger, trojan)
            new_dict[trojan].append((trigger, score))
    
    with open(output_file, 'w') as f:
        json.dump(new_dict, f, indent=4)


if __name__ == "__main__":
    setting = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    main(setting, input_file, output_file)
