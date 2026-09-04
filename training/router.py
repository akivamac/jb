"""
RouterNet — classifier that maps user messages to expert probabilities.

Architecture: Embedding → MeanPool → 2-layer MLP → Softmax
"""

import numpy as np
import os


class RouterNet:
    def __init__(self, vocab_size=2000, embed_dim=128, n_experts=7, seq_len=64):
        self.V = vocab_size
        self.C = embed_dim
        self.N = n_experts
        self.T = seq_len

        scale = np.sqrt(2.0 / (vocab_size + embed_dim))
        # D = pooled width (mean + max concatenation)
        D = 2 * embed_dim
        self.p = {
            'wte': (np.random.randn(vocab_size, embed_dim) * scale).astype(np.float32),
            'wpe': (np.random.randn(seq_len, embed_dim) * 0.02).astype(np.float32),
            'ln1_g': np.ones(D, dtype=np.float32),
            'ln1_b': np.zeros(D, dtype=np.float32),
            'fc1_w': (np.random.randn(D, D) * np.sqrt(2.0 / D)).astype(np.float32),
            'fc1_b': np.zeros(D, dtype=np.float32),
            'ln2_g': np.ones(D, dtype=np.float32),
            'ln2_b': np.zeros(D, dtype=np.float32),
            'fc2_w': (np.random.randn(D, D) * np.sqrt(2.0 / D)).astype(np.float32),
            'fc2_b': np.zeros(D, dtype=np.float32),
            'out_w': (np.random.randn(D, n_experts) * np.sqrt(2.0 / D)).astype(np.float32),
            'out_b': np.zeros(n_experts, dtype=np.float32),
        }

        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, idx):
        idx = np.asarray(idx, dtype=np.int32)
        T = len(idx)

        x = self.p['wte'][idx] + self.p['wpe'][:T]
        # Mask out padding (token id 0) from both mean and max pooling
        mask = (idx != 0).astype(np.float32)
        n = mask.sum()
        if n == 0:
            n = 1
        # Mean pooling (masked)
        h_mean = (x * mask[:, None]).sum(axis=0) / n
        # Max pooling (masked): pad real tokens with -inf so padding is ignored
        x_masked = np.where(mask[:, None] > 0, x, -1e9)
        h_max = x_masked.max(axis=0)
        h = np.concatenate([h_mean, h_max])

        # Block 1: LN + FC + ReLU + residual
        mean = h.mean()
        var = h.var()
        h1 = (h - mean) / np.sqrt(var + 1e-5) * self.p['ln1_g'] + self.p['ln1_b']
        z1 = np.maximum(h1 @ self.p['fc1_w'] + self.p['fc1_b'], 0)
        h = h + z1

        # Block 2: LN + FC + ReLU + residual
        mean = h.mean()
        var = h.var()
        h2 = (h - mean) / np.sqrt(var + 1e-5) * self.p['ln2_g'] + self.p['ln2_b']
        z2 = np.maximum(h2 @ self.p['fc2_w'] + self.p['fc2_b'], 0)
        h = h + z2

        # Output
        logits = h @ self.p['out_w'] + self.p['out_b']
        return logits

    def predict(self, idx):
        logits = self.forward(idx)
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs

    def save(self, path):
        tmp = path[:-4] + ".tmp.npz" if path.endswith('.npz') else path + ".tmp"
        arrays = {
            '__vocab_size__': np.array(self.V),
            '__embed_dim__': np.array(self.C),
            '__n_experts__': np.array(self.N),
            '__seq_len__': np.array(self.T),
            '__adam_t__': np.array(self.t),
        }
        for k, v in self.p.items():
            arrays[f'p_{k}'] = v
        for k, v in self.m.items():
            arrays[f'm_{k}'] = v
        for k, v in self.v.items():
            arrays[f'v_{k}'] = v

        np.savez_compressed(tmp, **arrays)
        import zipfile
        z = zipfile.ZipFile(tmp)
        bad = z.testzip()
        z.close()
        if bad is not None:
            os.remove(tmp)
            raise IOError(f"write failed at {bad}")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        V = int(data['__vocab_size__'])
        C = int(data['__embed_dim__'])
        N = int(data['__n_experts__'])
        T = int(data['__seq_len__'])

        model = cls(V, C, N, T)
        model.t = int(data['__adam_t__'])

        for key in data.files:
            if key.startswith('p_'):
                model.p[key[2:]] = data[key].astype(np.float32)
            elif key.startswith('m_'):
                model.m[key[2:]] = data[key].astype(np.float32)
            elif key.startswith('v_'):
                model.v[key[2:]] = data[key].astype(np.float32)

        return model

    def count_params(self):
        return sum(v.size for v in self.p.values())
