import torch.nn.functional as F
import torch as ch
import numpy as np
from itertools import chain


class AlternativeTokenizationsGenerator:
    # Maintain cache for tokenization lookup
    def __init__(self, tokenizer):
        """
            chunk_factor (float): Splitting into substrings at each character level is too expensive, so we chunk the string into substrings of length chunk_factor * len(x)
            max_tokens (int): Reject any tokenization that exceeds max_tokens
            min_length (int): ANy substring with <= min_length is tokenized directly with tokenizer.
        """
        self.tokenizer = tokenizer
        self.max_tokens = 3000
        self.num_tokenizations = 512
        self.chunk_factor = 0.01
        self.min_length = 1

        # Cache to reduce recursions and calls to tokenizer
        # Reset on every call (otherwise machine runs out of memory)
        self.cache = {"": set()}

    def _recurse(self, y: str, chunk_size: int):
        # Use cache if possible
        if y in self.cache:
            return self.cache[y]

        # Only one character- one token
        if len(y) <= self.min_length:
            self.cache[y] = set([tuple(self.tokenizer(y)["input_ids"])])
            return self.cache[y]

        all_toks = []
        for i in range(chunk_size, len(y), chunk_size):
            # Tokenizations of left part
            toks_left = self._recurse(y[:i], chunk_size)
            # Tokenizations of right part
            toks_right = self._recurse(y[i:], chunk_size)
            for a in toks_left:
                for b in toks_right:
                    # Combine all possible combinations
                    combined = a + b
                    # As long as they don't exceed max_tokens
                    if len(combined) <= self.max_tokens:
                        all_toks.append(combined)
      
        # Add given string (directly tokenized) to mix as well
        just = self.tokenizer(y)["input_ids"]
        if len(just) <= self.max_tokens:
            all_toks.append(tuple(just))

        # Make a set out of it
        all_toks = set(all_toks)

        # Update lookup table
        self.cache[y] = all_toks
        return all_toks

    def get_random_split_crude_tokenizations(self,
                                             x: str,
                                             at_most: int = None):
        """
            Make random number of random splits along string and send to tokenizer
            Much faster than recursive approaches.
        """
        alt_tokenizations = []
        i = 0
        total = 0
        while i < self.num_tokenizations:
            total += 1
            if total > 5000:
                raise ValueError("Too many iterations, check if at_most is too small")

            # Pick random number of splits :at least len()/6, since average token length would be around 3-4 tokens
            # So a /10 factor should be sufficient to get in more tokens
            num_splits = np.random.randint(len(x) // 10, min(at_most - 1, len(x)-1))
            # Pick random split points
            split_points = np.random.choice(len(x), size=num_splits, replace=False)
            # Sort them
            split_points = sorted(split_points)

            # Collect strings to tokenize
            wanted_toks = [x[:split_points[0]]]
            for j in range(1, len(split_points)):
                wanted_toks.append(x[split_points[j - 1]:split_points[j]])
            wanted_toks.append(x[split_points[-1]:])

            # Tokenize
            curr_toks = self.tokenizer(wanted_toks)["input_ids"]
            curr_toks = tuple(chain.from_iterable(curr_toks))

            if at_most is not None and len(curr_toks) > at_most:
            # Reject tokenization if it exceeds at_most
                continue

            alt_tokenizations.append(curr_toks)

            i += 1

        return alt_tokenizations

    def get_tokenizations(self,
                          x: str,
                          chunk_size: int = None):
        """
            Intuition- a lot of the tokenizations will include character-level tokens etc, which
            might not be as interesting as subword-level tokenizations
        """
        # Reset cache to avoid OOM
        self.cache = {"": set()}
        if chunk_size is None:
            chunk_size = int(len(x) * self.chunk_factor)
        all_combos = self._recurse(x, chunk_size)

        wanted = sorted(all_combos, key=lambda x: len(x))

        return wanted

    def get_tokenizations(self, doc: str, stochastic: bool = True):
        if stochastic:
            tokenizations = self.get_random_split_crude_tokenizations(doc, at_most=self.max_tokens)
        else:
            tokenizations = self.get_tokenizations(doc)
        return tokenizations
