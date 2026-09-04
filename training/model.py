"""
Tiny Transformer — pure numpy, no external ML libs.
Architecture: embedding → N blocks (self-attention + FFN) → output logits

Each block:
  x = x + self_attention(layer_norm(x))
  x = x + ffn(layer_norm(x))

All weights stored in a flat dict so they're easy to export as JSON.
"""

import numpy as np
import json
import os


_GELU_C = np.sqrt(2 / np.pi).astype(np.float32)
_GELU_A = np.float32(0.044715)

def gelu(x):
    x3 = x * x * x
    return 0.5 * x * (1 + np.tanh(_GELU_C * (x + _GELU_A * x3)))

def gelu_grad(x):
    x2 = x * x
    x3 = x2 * x
    tanh_val = np.tanh(_GELU_C * (x + _GELU_A * x3))
    sech2 = 1 - tanh_val * tanh_val
    dtanh = _GELU_C * (1 + 3 * _GELU_A * x2)
    return 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * dtanh

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b

def layer_norm_grad(dout, x, g, eps=1e-5):
    C = x.shape[-1]
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xhat = (x - mean) / np.sqrt(var + eps)
    # sum over all dims except last for dg, db
    sum_axes = tuple(range(x.ndim - 1))
    dg = (dout * xhat).sum(axis=sum_axes)
    db = dout.sum(axis=sum_axes)
    dxhat = dout * g
    dvar = (dxhat * (x - mean) * -0.5 * (var + eps)**-1.5).sum(axis=-1, keepdims=True)
    dmean = (dxhat * -1 / np.sqrt(var + eps)).sum(axis=-1, keepdims=True) + dvar * (-2 * (x - mean)).mean(axis=-1, keepdims=True)
    dx = dxhat / np.sqrt(var + eps) + dvar * 2 * (x - mean) / C + dmean / C
    return dx, dg, db


class JoeBrain:
    """
    Tiny transformer language model.
    T = sequence length, C = embed_dim, H = n_heads, L = n_layers
    """

    def __init__(self, vocab_size, embed_dim=128, n_heads=4, n_layers=3, seq_len=128):
        self.vocab_size = vocab_size
        self.C = embed_dim
        self.H = n_heads
        self.L = n_layers
        self.T = seq_len
        assert embed_dim % n_heads == 0
        self.head_dim = embed_dim // n_heads

        self.p = {}   # parameters
        self.g = {}   # gradients
        self.m = {}   # Adam first moment
        self.v = {}   # Adam second moment
        self.t = 0    # Adam step counter
        self._init_weights()

    def _init_weights(self):
        C, V, T, L = self.C, self.vocab_size, self.T, self.L

        def W(shape, scale=None):
            if scale is None:
                scale = np.sqrt(2.0 / (shape[0] + shape[-1]))  # Xavier
            return np.random.randn(*shape).astype(np.float32) * scale

        self.p['wte'] = W((V, C), scale=0.02)
        self.p['wpe'] = W((T, C), scale=0.01)
        self.p['ln_f_g'] = np.ones(C, dtype=np.float32)
        self.p['ln_f_b'] = np.zeros(C, dtype=np.float32)
        # tie output projection to embedding (weight tying reduces params, helps training)
        self.p['proj_b'] = np.zeros(V, dtype=np.float32)

        for i in range(L):
            # attention
            self.p[f'ln1_g_{i}'] = np.ones(C, dtype=np.float32)
            self.p[f'ln1_b_{i}'] = np.zeros(C, dtype=np.float32)
            self.p[f'qkv_w_{i}'] = W((C, 3 * C))
            self.p[f'qkv_b_{i}'] = np.zeros(3 * C, dtype=np.float32)
            self.p[f'attn_proj_w_{i}'] = W((C, C), scale=0.02 / np.sqrt(2 * L))
            self.p[f'attn_proj_b_{i}'] = np.zeros(C, dtype=np.float32)
            # ffn
            self.p[f'ln2_g_{i}'] = np.ones(C, dtype=np.float32)
            self.p[f'ln2_b_{i}'] = np.zeros(C, dtype=np.float32)
            self.p[f'fc_w_{i}'] = W((C, 4 * C))
            self.p[f'fc_b_{i}'] = np.zeros(4 * C, dtype=np.float32)
            self.p[f'fc2_w_{i}'] = W((4 * C, C), scale=0.02 / np.sqrt(2 * L))
            self.p[f'fc2_b_{i}'] = np.zeros(C, dtype=np.float32)

        self.g = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._mask_cache = {}  # T -> causal mask (T, T) float32

    def _get_mask(self, T):
        m = self._mask_cache.get(T)
        if m is None:
            m = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
            self._mask_cache[T] = m
        return m

    def forward(self, idx):
        """idx: (B, T) or (T,) int array. Returns logits and cache for backward.
        Always works batched internally: idx (B, T) -> logits (B, T, V)
        If unbatched input, returns logits (T, V) and unbatched cache.
        """
        unbatched = idx.ndim == 1
        if unbatched:
            idx = idx[None, :]  # (1, T)

        B, T = idx.shape
        C, H, L = self.C, self.H, self.L
        hd = C // H
        p = self.p
        cache = {'idx': idx, 'B': B, 'T': T}

        # Embeddings: (B, T, C)  — no copy needed, x is never mutated in-place
        x = p['wte'][idx] + p['wpe'][np.arange(T)]

        block_caches = []
        for i in range(L):
            bc = {}

            # --- Attention ---
            bc['x_pre_ln1'] = x           # ref, not copy (x is reassigned, not mutated)
            x_ln = layer_norm(x, p[f'ln1_g_{i}'], p[f'ln1_b_{i}'])
            bc['x_ln1'] = x_ln

            qkv = x_ln @ p[f'qkv_w_{i}'] + p[f'qkv_b_{i}']  # (B, T, 3C)
            q, k, v = np.split(qkv, 3, axis=-1)

            # reshape to (B, H, T, hd)
            q = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

            scale = 1.0 / np.sqrt(hd)
            attn_scores = q @ k.transpose(0, 1, 3, 2) * scale  # (B, H, T, T)

            # causal mask (cached per-T; broadcasting handles batch dim)
            mask = self._get_mask(T)
            attn_scores = attn_scores + mask
            attn_w = softmax(attn_scores, axis=-1)  # (B, H, T, T)

            bc['q'] = q; bc['k'] = k; bc['v'] = v
            bc['attn_w'] = attn_w
            bc['scale'] = scale

            out = attn_w @ v  # (B, H, T, hd)
            out = out.transpose(0, 2, 1, 3).reshape(B, T, C)  # (B, T, C)
            bc['attn_merged'] = out          # ref, not copy

            out = out @ p[f'attn_proj_w_{i}'] + p[f'attn_proj_b_{i}']
            x = x + out

            # --- FFN ---
            bc['x_pre_ln2'] = x              # ref, not copy
            x_ln2 = layer_norm(x, p[f'ln2_g_{i}'], p[f'ln2_b_{i}'])
            bc['x_ln2'] = x_ln2

            h = x_ln2 @ p[f'fc_w_{i}'] + p[f'fc_b_{i}']
            bc['h_pre_act'] = h
            h_act = gelu(h)
            bc['h_act'] = h_act
            ffn_out = h_act @ p[f'fc2_w_{i}'] + p[f'fc2_b_{i}']
            x = x + ffn_out

            block_caches.append(bc)

        cache['block_caches'] = block_caches
        cache['x_final_pre_ln'] = x         # ref, not copy

        x = layer_norm(x, p['ln_f_g'], p['ln_f_b'])
        cache['x_final'] = x                 # ref, not copy

        # weight-tied output: reuse wte
        logits = x @ p['wte'].T + p['proj_b']  # (B, T, V)

        if unbatched:
            logits = logits[0]  # (T, V)
        return logits, cache

    def loss(self, logits, targets, mask=None, penalty_ids=None, penalty_weight=0.5):
        """Cross-entropy loss. logits (T, V) or (B, T, V), targets matching.
        mask: (B, T) or (T,) float — 1.0 for positions to train on, 0.0 to skip.
        penalty_ids: list of token ids to penalize (push logits down).
        penalty_weight: strength of penalty (added to dlogits for penalized ids).
        """
        if targets.ndim == 1:
            probs = softmax(logits, axis=-1)
            T = len(targets)
            log_probs = -np.log(probs[np.arange(T), targets] + 1e-9)
            if mask is not None:
                log_probs = log_probs * mask
                N = max(mask.sum(), 1)
            else:
                N = T
            loss = log_probs.sum() / N
            dlogits = probs.copy()
            dlogits[np.arange(T), targets] -= 1
            if mask is not None:
                dlogits *= mask[:, None]
            dlogits /= N
            if penalty_ids:
                for pid in penalty_ids:
                    dlogits[:, pid] += penalty_weight / N
            return loss, dlogits
        probs = softmax(logits, axis=-1)
        B, T = targets.shape
        b_idx = np.arange(B)[:, None]
        t_idx = np.arange(T)[None, :]
        log_probs = -np.log(probs[b_idx, t_idx, targets] + 1e-9)
        if mask is not None:
            log_probs = log_probs * mask
            N = max(mask.sum(), 1)
        else:
            N = B * T
        loss = log_probs.sum() / N
        dlogits = probs.copy()
        dlogits[b_idx, t_idx, targets] -= 1
        if mask is not None:
            dlogits *= mask[:, :, None]
        dlogits /= N
        if penalty_ids:
            for pid in penalty_ids:
                dlogits[:, :, pid] += penalty_weight / N
        return loss, dlogits

    def backward(self, dlogits, cache):
        """Accumulates gradients. Works with batched dlogits (B, T, V).
        Also handles unbatched (T, V) by adding a batch dim internally.
        """
        p, g = self.p, self.g
        idx = cache['idx']        # (B, T)
        B, T = idx.shape
        C, H, L = self.C, self.H, self.L
        hd = C // H

        # Ensure dlogits is (B, T, V)
        if dlogits.ndim == 2:
            dlogits = dlogits[None]  # (1, T, V)

        # Output projection (weight-tied to wte)
        x_final = cache['x_final']  # (B, T, C)
        g['proj_b'] += dlogits.sum(axis=(0, 1))
        # g['wte'] += dlogits.T @ x_final  -> (V, B*T) @ (B*T, C) = (V, C)
        BT = B * T
        g['wte'] += dlogits.reshape(BT, -1).T @ x_final.reshape(BT, C)
        dx = dlogits @ p['wte']  # (B, T, V) @ (V, C) -> (B, T, C)

        # Final layer norm
        dx, dg, db = layer_norm_grad(dx, cache['x_final_pre_ln'], p['ln_f_g'])
        g['ln_f_g'] += dg
        g['ln_f_b'] += db

        for i in reversed(range(L)):
            bc = cache['block_caches'][i]

            # --- FFN backward ---
            dffn = dx  # (B, T, C), no copy needed (dx is freshly computed each iter)
            g[f'fc2_b_{i}'] += dffn.sum(axis=(0, 1))
            # bc['h_act']: (B, T, 4C), dffn: (B, T, C)
            g[f'fc2_w_{i}'] += bc['h_act'].reshape(BT, -1).T @ dffn.reshape(BT, C)
            dh_act = dffn @ p[f'fc2_w_{i}'].T  # (B, T, 4C)
            dh = dh_act * gelu_grad(bc['h_pre_act'])  # (B, T, 4C)
            g[f'fc_b_{i}'] += dh.sum(axis=(0, 1))
            g[f'fc_w_{i}'] += bc['x_ln2'].reshape(BT, C).T @ dh.reshape(BT, -1)
            dx_ln2 = dh @ p[f'fc_w_{i}'].T  # (B, T, C)
            dx_ln2, dg, db = layer_norm_grad(dx_ln2, bc['x_pre_ln2'], p[f'ln2_g_{i}'])
            g[f'ln2_g_{i}'] += dg
            g[f'ln2_b_{i}'] += db
            dx = dx + dx_ln2  # residual

            # --- Attention backward ---
            dattn_out = dx  # (B, T, C), no copy needed
            g[f'attn_proj_b_{i}'] += dattn_out.sum(axis=(0, 1))
            g[f'attn_proj_w_{i}'] += bc['attn_merged'].reshape(BT, C).T @ dattn_out.reshape(BT, C)
            dout_merged = dattn_out @ p[f'attn_proj_w_{i}'].T  # (B, T, C)

            # reshape (B, T, C) -> (B, H, T, hd)
            dout_heads = dout_merged.reshape(B, T, H, hd).transpose(0, 2, 1, 3)  # (B, H, T, hd)

            attn_w = bc['attn_w']   # (B, H, T, T)
            v = bc['v']             # (B, H, T, hd)
            q = bc['q']             # (B, H, T, hd)
            k = bc['k']             # (B, H, T, hd)
            scale = bc['scale']

            dv = attn_w.transpose(0, 1, 3, 2) @ dout_heads        # (B, H, T, hd)
            dattn_w = dout_heads @ v.transpose(0, 1, 3, 2)       # (B, H, T, T)
            dattn_scores = attn_w * (dattn_w - (dattn_w * attn_w).sum(axis=-1, keepdims=True))
            dattn_scores *= scale

            dq = dattn_scores @ k                          # (B, H, T, hd)
            dk = dattn_scores.transpose(0, 1, 3, 2) @ q    # (B, H, T, hd)

            # (B, H, T, hd) -> (B, T, C)
            dq = dq.transpose(0, 2, 1, 3).reshape(B, T, C)
            dk = dk.transpose(0, 2, 1, 3).reshape(B, T, C)
            dv = dv.transpose(0, 2, 1, 3).reshape(B, T, C)

            dqkv = np.concatenate([dq, dk, dv], axis=-1)  # (B, T, 3C)
            g[f'qkv_b_{i}'] += dqkv.sum(axis=(0, 1))
            g[f'qkv_w_{i}'] += bc['x_ln1'].reshape(BT, C).T @ dqkv.reshape(BT, -1)
            dx_attn = dqkv @ p[f'qkv_w_{i}'].T  # (B, T, C)

            dx_attn, dg, db = layer_norm_grad(dx_attn, bc['x_pre_ln1'], p[f'ln1_g_{i}'])
            g[f'ln1_g_{i}'] += dg
            g[f'ln1_b_{i}'] += db
            dx = dx + dx_attn  # residual

        # Embeddings
        np.add.at(g['wte'], idx, dx)
        g['wpe'][:T] += dx.sum(axis=0)  # sum over batch dim

    def zero_grad(self):
        for k in self.g:
            self.g[k][:] = 0

    def step(self, lr, clip=1.0, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam with gradient clipping."""
        # Clip
        total_norm = np.sqrt(sum((v**2).sum() for v in self.g.values()))
        if total_norm > clip:
            scale = clip / (total_norm + 1e-8)
            for k in self.g:
                self.g[k] *= scale

        self.t += 1
        bc1 = 1 - beta1 ** self.t
        bc2 = 1 - beta2 ** self.t

        for k in self.p:
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * self.g[k]
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * self.g[k] ** 2
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            self.p[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def save(self, path):
        # Atomic write: write to a temp file, verify, then rename into place.
        # Writing directly to the final path risks a torn/corrupt checkpoint if the
        # process dies mid-write or another process reads while we write.
        # NOTE: numpy appends '.npz' to the filename unless it already ends in '.npz',
        # so the temp name must keep the '.npz' suffix.
        tmp = (path[:-4] + ".tmp.npz") if path.endswith('.npz') else (path + ".tmp")
        if path.endswith('.npz'):
            arrays = {}
            arrays['__vocab_size__'] = np.array(self.vocab_size)
            arrays['__embed_dim__'] = np.array(self.C)
            arrays['__n_heads__'] = np.array(self.H)
            arrays['__n_layers__'] = np.array(self.L)
            arrays['__seq_len__'] = np.array(self.T)
            arrays['__adam_t__'] = np.array(self.t)
            for k, v in self.p.items():
                arrays[f'p_{k}'] = v
            for k, v in self.m.items():
                arrays[f'm_{k}'] = v
            for k, v in self.v.items():
                arrays[f'v_{k}'] = v
            np.savez_compressed(tmp, **arrays)
            # verify every entry (CRC) before exposing the file
            import zipfile
            z = zipfile.ZipFile(tmp)
            bad = z.testzip()
            z.close()
            if bad is not None:
                os.remove(tmp)
                raise IOError(f"checkpoint write failed verification at {bad}; temp removed")
        else:
            data = {k: v.tolist() for k, v in self.p.items()}
            data['__config__'] = {
                'vocab_size': self.vocab_size,
                'embed_dim': self.C,
                'n_heads': self.H,
                'n_layers': self.L,
                'seq_len': self.T,
            }
            data['__adam__'] = {
                't': self.t,
                'm': {k: v.tolist() for k, v in self.m.items()},
                'v': {k: v.tolist() for k, v in self.v.items()},
            }
            with open(tmp, 'w') as f:
                json.dump(data, f)
        os.replace(tmp, path)
        print(f"Model saved to {path}")

    def prefill(self, idx):
        """
        Run full forward pass on prompt tokens, return logits for last position
        and KV cache for all layers.
        kv_cache: list of {'k': (H, T, hd), 'v': (H, T, hd)} per layer
        """
        logits, cache = self.forward(idx)
        # forward keeps batched cache as (1, H, T, hd) for unbatched input; squeeze batch
        kv_cache = [
            {'k': cache['block_caches'][i]['k'][0].copy(),   # (H, T, hd)
             'v': cache['block_caches'][i]['v'][0].copy()}
            for i in range(self.L)
        ]
        return logits[-1], kv_cache

    def forward_one(self, token_id, position, kv_cache):
        """
        Single-token forward pass using KV cache. No backward support.
        token_id: int
        position: int (absolute position index, capped at seq_len-1)
        kv_cache: list of {'k': (H, past, hd), 'v': (H, past, hd)}
        Returns: logits (V,), new_kv_cache
        """
        p = self.p
        C, H, L = self.C, self.H, self.L
        hd = C // H
        pos = position % self.T

        x = p['wte'][token_id] + p['wpe'][pos]  # (C,)
        x = x[None, :]  # (1, C)

        new_kv = []
        for i in range(L):
            x_ln = layer_norm(x, p[f'ln1_g_{i}'], p[f'ln1_b_{i}'])
            qkv = x_ln @ p[f'qkv_w_{i}'] + p[f'qkv_b_{i}']
            q, k, v = np.split(qkv, 3, axis=-1)  # each (1, C)

            q = q.reshape(1, H, hd).transpose(1, 0, 2)  # (H, 1, hd)
            k = k.reshape(1, H, hd).transpose(1, 0, 2)
            v = v.reshape(1, H, hd).transpose(1, 0, 2)

            # Append new k,v to cache
            k_full = np.concatenate([kv_cache[i]['k'], k], axis=1)  # (H, past+1, hd)
            v_full = np.concatenate([kv_cache[i]['v'], v], axis=1)

            # Trim to seq_len
            if k_full.shape[1] > self.T:
                k_full = k_full[:, -self.T:, :]
                v_full = v_full[:, -self.T:, :]

            new_kv.append({'k': k_full, 'v': v_full})

            scale = 1.0 / np.sqrt(hd)
            # q: (H,1,hd)  k_full: (H,past+1,hd) -> scores: (H,1,past+1)
            attn_scores = q @ k_full.transpose(0, 2, 1) * scale
            attn_w = softmax(attn_scores, axis=-1)
            out = attn_w @ v_full  # (H, 1, hd)
            out = out.transpose(1, 0, 2).reshape(1, C)
            out = out @ p[f'attn_proj_w_{i}'] + p[f'attn_proj_b_{i}']
            x = x + out

            x_ln2 = layer_norm(x, p[f'ln2_g_{i}'], p[f'ln2_b_{i}'])
            h = x_ln2 @ p[f'fc_w_{i}'] + p[f'fc_b_{i}']
            ffn_out = gelu(h) @ p[f'fc2_w_{i}'] + p[f'fc2_b_{i}']
            x = x + ffn_out

        x = layer_norm(x, p['ln_f_g'], p['ln_f_b'])
        logits = (x @ p['wte'].T + p['proj_b'])[0]  # (V,)
        return logits, new_kv

    @classmethod
    def load(cls, path):
        if path.endswith('.npz'):
            try:
                data = np.load(path, allow_pickle=False)
            except Exception as e:
                raise IOError(
                    f"checkpoint {path} is corrupt or unreadable ({e}). "
                    f"Restore from a backup / git, or retrain. Saves are now atomic "
                    f"so this should not recur."
                ) from e
            cfg = {
                'vocab_size': int(data['__vocab_size__']),
                'embed_dim': int(data['__embed_dim__']),
                'n_heads': int(data['__n_heads__']),
                'n_layers': int(data['__n_layers__']),
                'seq_len': int(data['__seq_len__']),
            }
            m = cls(**cfg)
            m.t = int(data['__adam_t__'])
            m.p = {k[2:]: data[k].astype(np.float32) for k in data if k.startswith('p_')}
            m.m = {k[2:]: data[k].astype(np.float32) for k in data if k.startswith('m_')}
            m.v = {k[2:]: data[k].astype(np.float32) for k in data if k.startswith('v_')}
            m.g = {k: np.zeros_like(v) for k, v in m.p.items()}
        else:
            with open(path) as f:
                data = json.load(f)
            cfg = data.pop('__config__')
            adam = data.pop('__adam__', None)
            m = cls(**cfg)
            m.p = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
            m.g = {k: np.zeros_like(v) for k, v in m.p.items()}
            if adam:
                m.t = adam['t']
                m.m = {k: np.array(v, dtype=np.float32) for k, v in adam['m'].items()}
                m.v = {k: np.array(v, dtype=np.float32) for k, v in adam['v'].items()}
            else:
                m.m = {k: np.zeros_like(v) for k, v in m.p.items()}
                m.v = {k: np.zeros_like(v) for k, v in m.p.items()}
        return m

    def generate(self, tokenizer, prompt, max_new=200, temperature=0.8):
        """Generate text from a prompt string."""
        ids = tokenizer.encode(prompt)
        for _ in range(max_new):
            ctx = ids[-self.T:]
            ctx_arr = np.array(ctx, dtype=np.int32)
            logits, _ = self.forward(ctx_arr)
            last_logits = logits[-1] / temperature
            probs = softmax(last_logits)
            next_id = np.random.choice(len(probs), p=probs)
            ids.append(next_id)
        return tokenizer.decode(ids)
