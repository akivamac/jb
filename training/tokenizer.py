"""
BPE (Byte Pair Encoding) tokenizer.
Starts with individual characters, then iteratively merges the most frequent pairs.
Compatible interface with the old character-level tokenizer.
"""

import json
import re


class Tokenizer:
    def __init__(self):
        self.merges = []         # list of (str, str) merge pairs in order
        self.vocab = {}          # token string -> id
        self.id_to_token = {}    # id -> token string
        self.size = 0

        # Backward compat aliases used by server.py
        self.char_to_id = self.vocab
        self.id_to_char = self.id_to_token

    def build(self, text, vocab_size=2000):
        """Build BPE vocabulary from training text."""
        # Start with all unique characters as base vocab
        chars = sorted(set(text))
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.id_to_token = {i: c for i, c in enumerate(chars)}
        self.merges = []
        next_id = len(chars)

        # Pre-tokenize: split into words (keeps spaces attached to following word)
        # This prevents merges across word boundaries
        words = re.findall(r'\S+|\s', text)

        # Represent each word as a tuple of characters
        word_freqs = {}
        for word in words:
            key = tuple(word)
            word_freqs[key] = word_freqs.get(key, 0) + 1

        num_merges = vocab_size - len(chars)
        print(f"Base vocab: {len(chars)} chars, planning {num_merges} merges")

        for step in range(num_merges):
            # Count all adjacent pairs
            pair_counts = {}
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            if best_count < 2:
                break

            # Merge the pair
            merged = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.vocab[merged] = next_id
            self.id_to_token[next_id] = merged
            next_id += 1

            # Update word_freqs: apply merge to all words
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = self._apply_merge(word, best_pair, merged)
                new_word_freqs[new_word] = new_word_freqs.get(new_word, 0) + freq
            word_freqs = new_word_freqs

            if (step + 1) % 200 == 0:
                print(f"  merge {step + 1}/{num_merges}: '{best_pair[0]}' + '{best_pair[1]}' -> '{merged}' (count={best_count})")

        self.size = next_id
        # Re-sync aliases
        self.char_to_id = self.vocab
        self.id_to_char = self.id_to_token
        print(f"Vocab size: {self.size} tokens ({len(chars)} chars + {len(self.merges)} merges)")

    def _apply_merge(self, word, pair, merged):
        """Apply a single merge to a word tuple."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def _build_merge_ranks(self):
        """Build a lookup: (a, b) -> merge rank (lower = applied first)."""
        self._merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    def _encode_chunk(self, chunk):
        """Encode a single pre-tokenized chunk using merge ranks."""
        if chunk in self._encode_cache:
            return self._encode_cache[chunk]

        tokens = list(chunk)

        while len(tokens) > 1:
            # Find the pair with the lowest merge rank
            best_rank = None
            best_i = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i

            if best_rank is None:
                break  # no more merges applicable

            # Merge the best pair
            tokens[best_i] = tokens[best_i] + tokens[best_i + 1]
            del tokens[best_i + 1]

        result = [self.vocab.get(t, 0) for t in tokens]
        self._encode_cache[chunk] = result
        return result

    def encode(self, text):
        """Encode a string into a list of token IDs."""
        if not text:
            return []

        # Lazy-init merge ranks and cache
        if not hasattr(self, '_merge_ranks'):
            self._build_merge_ranks()
        if not hasattr(self, '_encode_cache'):
            self._encode_cache = {}

        chunks = re.findall(r'\S+|\s', text)
        ids = []
        for chunk in chunks:
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids):
        """Decode a list of token IDs back to a string."""
        return ''.join(self.id_to_token.get(i, '?') for i in ids)

    def save(self, path):
        data = {
            'type': 'bpe',
            'merges': self.merges,
            'vocab': self.vocab,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        with open(path) as f:
            data = json.load(f)

        if data.get('type') == 'bpe':
            self.merges = [tuple(m) for m in data['merges']]
            self.vocab = data['vocab']
            self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        else:
            # Backward compat: load old char-level format
            self.merges = []
            self.vocab = data['char_to_id']
            self.id_to_token = {int(k): v for k, v in data['id_to_char'].items()}

        self.size = len(self.vocab)
        # Re-sync aliases
        self.char_to_id = self.vocab
        self.id_to_char = self.id_to_token
