# My solution for TDC 2023 (LLM Edition)

**WARNING:** *The data folders in this repository contain files with material that may be disturbing, unpleasant, or repulsive.*

To get started, install my fork of `llm-attacks`

```bash
git clone https://github.com/iamgroot42/llm-attacks
cd llm-attacks
pip install -e .
```

Then, navigatee into the `trojan_detection` folder and execute the attack
```bash
cd trojan_detection
python gcg_trojan_gen.py --setting base --n_iters 50 --break_on_success --use_negative_feedback --n_tries 50 --n_want 20
```

For a detailed walkthrough of my solution, see [my blog post](https://www.anshumansuri.me/post/tdc/)
