"""
Tiny Transformer — MLX (Apple GPU) backend. Drop-in port of model.py's JoeBrain.
Architecture identical: embedding → N blocks (self-attention + FFN) → output logits
  x = x + self_attention(layer_norm(x))
  x = x + ffn(layer_norm(x))

Key differences from model.py (numpy):
  - Computes on the Apple GPU via mlx.core (Metal). Weights are mx.array.
  - No in-place mutation: gradients accumulate via dict rebinding (g[k] = g[k] + dx),
    and embedding grads use .at[idx].add() (MLX's functional scatter-add).
  - loss uses one_hot built via where() and take_along_axis (mlx has no one_hot/scatter_add).
  - mx.eval() is called at the end of forward/backward/step to materialize values and
    prevent unbounded lazy-graph growth across training steps.

Checkpoints are 100% compatible with model.py: save writes numpy .npz (arrays converted
numpy->mx on load, mx->numpy on save), so model.py and model_mlx.py can load each other's saves.
"""

import numpy as np
import json
import os

import mlx.core as mx


_GELU_C = mx.array(np.sqrt(2 / np.pi), dtype=mx.float32)
_GELU_A = mx.array(0.044715, dtype=mx.float32)


def gelu(x):
    x3 = x * x * x
    return 0.5 * x * (1 + mx.tanh(_GELU_C * (x + _GELU_A * x3)))


def gelu_grad(x):
    x2 = x * x
    x3 = x2 * x
    tanh_val = mx.tanh(_GELU_C * (x + _GELU_A * x3))
    sech2 = 1 - tanh_val * tanh_val
    dtanh = _GELU_C * (1 + 3 * _GELU_A * x2)
    return 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * dtanh


def softmax(x, axis=-1):
    return mx.softmax(x, axis=axis)


def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return g * (x - mean) / mx.sqrt(var + eps) + b


def layer_norm_grad(dout, x, g, eps=1e-5):
    C = x.shape[-1]
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xhat = (x - mean) / mx.sqrt(var + eps)
    sum_axes = tuple(range(x.ndim - 1))
    dg = (dout * xhat).sum(axis=sum_axes)
    db = dout.sum(axis=sum_axes)
    dxhat = dout * g
    dvar = (dxhat * (x - mean) * -0.5 * (var + eps) ** -1.5).sum(axis=-1, keepdims=True)
    dmean = (dxhat * -1 / mx.sqrt(var + eps)).sum(axis=-1, keepdims=True) + dvar * (-2 * (x - mean)).mean(axis=-1, keepdims=True)
    dx = dxhat / mx.sqrt(var + eps) + dvar * 2 * (x - mean) / C + dmean / C
    return dx, dg, db


def _as_mx(x, dtype=mx.float32):
    """Convert numpy array / python list to an MLX array (fp32 default)."""
    if isinstance(x, mx.array):
        return x
    return mx.array(np.asarray(x), dtype=dtype)


class JoeBrain:
    """
    Tiny transformer language model (MLX/GPU backend).
    T = sequence length, C = embed_dim, H = n_heads, L = n_layers
    API-compatible with model.py's JoeBrain.
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
            return mx.array(np.random.randn(*shape).astype(np.float32) * scale)

        self.p['wte'] = W((V, C), scale=0.02)
        self.p['wpe'] = W((T, C), scale=0.01)
        self.p['ln_f_g'] = mx.ones(C, dtype=mx.float32)
        self.p['ln_f_b'] = mx.zeros(C, dtype=mx.float32)
        self.p['proj_b'] = mx.zeros(V, dtype=mx.float32)

        for i in range(L):
            self.p[f'ln1_g_{i}'] = mx.ones(C, dtype=mx.float32)
            self.p[f'ln1_b_{i}'] = mx.zeros(C, dtype=mx.float32)
            self.p[f'qkv_w_{i}'] = W((C, 3 * C))
            self.p[f'qkv_b_{i}'] = mx.zeros(3 * C, dtype=mx.float32)
            self.p[f'attn_proj_w_{i}'] = W((C, C), scale=0.02 / np.sqrt(2 * L))
            self.p[f'attn_proj_b_{i}'] = mx.zeros(C, dtype=mx.float32)
            self.p[f'ln2_g_{i}'] = mx.ones(C, dtype=mx.float32)
            self.p[f'ln2_b_{i}'] = mx.zeros(C, dtype=mx.float32)
            self.p[f'fc_w_{i}'] = W((C, 4 * C))
            self.p[f'fc_b_{i}'] = mx.zeros(4 * C, dtype=mx.float32)
            self.p[f'fc2_w_{i}'] = W((4 * C, C), scale=0.02 / np.sqrt(2 * L))
            self.p[f'fc2_b_{i}'] = mx.zeros(C, dtype=mx.float32)

        self.g = {k: mx.zeros_like(v) for k, v in self.p.items()}
        self.m = {k: mx.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: mx.zeros_like(v) for k, v in self.p.items()}
        self._mask_cache = {}  # T -> causal mask (T, T) float32

    def _get_mask(self, T):
        m = self._mask_cache.get(T)
        if m is None:
            m = mx.triu(mx.full((T, T), -1e9, dtype=mx.float32), k=1)
            self._mask_cache[T] = m
        return m

    def forward(self, idx):
        """idx: (B, T) or (T,) int array. Returns logits and cache for backward.
        idx may be numpy int array or mlx array.
        """
        idx = _as_mx(idx, dtype=mx.int32)
        unbatched = idx.ndim == 1
        if unbatched:
            idx = idx[None, :]

        B, T = idx.shape
        C, H, L = self.C, self.H, self.L
        hd = C // H
        p = self.p
        cache = {'idx': idx, 'B': B, 'T': T}

        x = p['wte'][idx] + p['wpe'][mx.arange(T)]

        block_caches = []
        for i in range(L):
            bc = {}
            bc['x_pre_ln1'] = x
            x_ln = layer_norm(x, p[f'ln1_g_{i}'], p[f'ln1_b_{i}'])
            bc['x_ln1'] = x_ln

            qkv = x_ln @ p[f'qkv_w_{i}'] + p[f'qkv_b_{i}']
            q, k, v = mx.split(qkv, 3, axis=-1)

            q = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

            scale = 1.0 / np.sqrt(hd)
            attn_scores = q @ k.transpose(0, 1, 3, 2) * scale

            mask = self._get_mask(T)
            attn_scores = attn_scores + mask
            attn_w = softmax(attn_scores, axis=-1)

            bc['q'] = q; bc['k'] = k; bc['v'] = v
            bc['attn_w'] = attn_w
            bc['scale'] = scale

            out = attn_w @ v
            out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
            bc['attn_merged'] = out

            out = out @ p[f'attn_proj_w_{i}'] + p[f'attn_proj_b_{i}']
            x = x + out

            bc['x_pre_ln2'] = x
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
        cache['x_final_pre_ln'] = x

        x = layer_norm(x, p['ln_f_g'], p['ln_f_b'])
        cache['x_final'] = x

        logits = x @ p['wte'].T + p['proj_b']

        if unbatched:
            logits = logits[0]
        mx.eval(logits)
        return logits, cache

    def loss(self, logits, targets, mask=None, penalty_ids=None, penalty_weight=0.5):
        """Cross-entropy loss. logits (T, V) or (B, T, V), targets matching.
        mask: (B, T) or (T,) float — 1.0 for positions to train on, 0.0 to skip.
        penalty_ids: list of token ids to penalize (push logits down).
        Returns (loss_scalar_float, dlogits_mx).
        """
        targets = _as_mx(targets, dtype=mx.int32)
        probs = softmax(logits, axis=-1)

        if targets.ndim == 1:
            T = targets.shape[0]
            taken = mx.take_along_axis(probs, targets[:, None], axis=-1)[:, 0]
            log_probs = -mx.log(taken + 1e-9)
            if mask is not None:
                mask = _as_mx(mask)
                log_probs = log_probs * mask
                N = mx.maximum(mask.sum(), 1).item()
            else:
                N = T
            loss = log_probs.sum() / N
            dlogits = probs - (mx.arange(probs.shape[-1])[None, :] == targets[:, None]).astype(mx.float32)
            if mask is not None:
                dlogits = dlogits * mask[:, None]
            dlogits = dlogits / N
        else:
            B, T = targets.shape
            taken = mx.take_along_axis(probs, targets[:, :, None], axis=-1)[:, :, 0]
            log_probs = -mx.log(taken + 1e-9)
            if mask is not None:
                mask = _as_mx(mask)
                log_probs = log_probs * mask
                N = mx.maximum(mask.sum(), 1).item()
            else:
                N = B * T
            loss = log_probs.sum() / N
            one_hot = (mx.arange(probs.shape[-1])[None, None, :] == targets[:, :, None]).astype(mx.float32)
            dlogits = probs - one_hot
            if mask is not None:
                dlogits = dlogits * mask[:, :, None]
            dlogits = dlogits / N

        if penalty_ids:
            pen = mx.zeros(probs.shape[-1], dtype=mx.float32)
            for pid in penalty_ids:
                pen = pen + (mx.arange(probs.shape[-1]) == pid).astype(mx.float32)
            if targets.ndim == 1:
                dlogits = dlogits + pen[None, :] * (penalty_weight / N)
            else:
                dlogits = dlogits + pen[None, None, :] * (penalty_weight / N)

        mx.eval(loss, dlogits)
        return loss.item(), dlogits

    def backward(self, dlogits, cache):
        """Accumulates gradients into self.g. cache from forward()."""
        p, g = self.p, self.g
        idx = cache['idx']
        B, T = idx.shape
        C, H, L = self.C, self.H, self.L
        hd = C // H

        if dlogits.ndim == 2:
            dlogits = dlogits[None, :]

        x_final = cache['x_final']
        g['proj_b'] = g['proj_b'] + dlogits.sum(axis=(0, 1))
        BT = B * T
        g['wte'] = g['wte'] + dlogits.reshape(BT, -1).T @ x_final.reshape(BT, C)
        dx = dlogits @ p['wte']

        dx, dg, db = layer_norm_grad(dx, cache['x_final_pre_ln'], p['ln_f_g'])
        g['ln_f_g'] = g['ln_f_g'] + dg
        g['ln_f_b'] = g['ln_f_b'] + db

        for i in reversed(range(L)):
            bc = cache['block_caches'][i]

            dffn = dx
            g[f'fc2_b_{i}'] = g[f'fc2_b_{i}'] + dffn.sum(axis=(0, 1))
            g[f'fc2_w_{i}'] = g[f'fc2_w_{i}'] + bc['h_act'].reshape(BT, -1).T @ dffn.reshape(BT, C)
            dh_act = dffn @ p[f'fc2_w_{i}'].T
            dh = dh_act * gelu_grad(bc['h_pre_act'])
            g[f'fc_b_{i}'] = g[f'fc_b_{i}'] + dh.sum(axis=(0, 1))
            g[f'fc_w_{i}'] = g[f'fc_w_{i}'] + bc['x_ln2'].reshape(BT, C).T @ dh.reshape(BT, -1)
            dx_ln2 = dh @ p[f'fc_w_{i}'].T
            dx_ln2, dg, db = layer_norm_grad(dx_ln2, bc['x_pre_ln2'], p[f'ln2_g_{i}'])
            g[f'ln2_g_{i}'] = g[f'ln2_g_{i}'] + dg
            g[f'ln2_b_{i}'] = g[f'ln2_b_{i}'] + db
            dx = dx + dx_ln2

            dattn_out = dx
            g[f'attn_proj_b_{i}'] = g[f'attn_proj_b_{i}'] + dattn_out.sum(axis=(0, 1))
            g[f'attn_proj_w_{i}'] = g[f'attn_proj_w_{i}'] + bc['attn_merged'].reshape(BT, C).T @ dattn_out.reshape(BT, C)
            dout_merged = dattn_out @ p[f'attn_proj_w_{i}'].T

            dout_heads = dout_merged.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

            attn_w = bc['attn_w']
            v = bc['v']
            q = bc['q']
            k = bc['k']
            scale = bc['scale']

            dv = attn_w.transpose(0, 1, 3, 2) @ dout_heads
            dattn_w = dout_heads @ v.transpose(0, 1, 3, 2)
            dattn_scores = attn_w * (dattn_w - (dattn_w * attn_w).sum(axis=-1, keepdims=True))
            dattn_scores = dattn_scores * scale

            dq = dattn_scores @ k
            dk = dattn_scores.transpose(0, 1, 3, 2) @ q

            dq = dq.transpose(0, 2, 1, 3).reshape(B, T, C)
            dk = dk.transpose(0, 2, 1, 3).reshape(B, T, C)
            dv = dv.transpose(0, 2, 1, 3).reshape(B, T, C)

            dqkv = mx.concatenate([dq, dk, dv], axis=-1)
            g[f'qkv_b_{i}'] = g[f'qkv_b_{i}'] + dqkv.sum(axis=(0, 1))
            g[f'qkv_w_{i}'] = g[f'qkv_w_{i}'] + bc['x_ln1'].reshape(BT, C).T @ dqkv.reshape(BT, -1)
            dx_attn = dqkv @ p[f'qkv_w_{i}'].T

            dx_attn, dg, db = layer_norm_grad(dx_attn, bc['x_pre_ln1'], p[f'ln1_g_{i}'])
            g[f'ln1_g_{i}'] = g[f'ln1_g_{i}'] + dg
            g[f'ln1_b_{i}'] = g[f'ln1_b_{i}'] + db
            dx = dx + dx_attn

        # Embeddings (functional scatter-add; idx may repeat -> accumulates)
        flat_idx = idx.reshape(BT)
        dx_flat = dx.reshape(BT, C)
        g['wte'] = g['wte'].at[flat_idx].add(dx_flat)
        g['wpe'] = g['wpe'].at[:T].add(dx.sum(axis=0))

        mx.eval(list(g.values()))

    def zero_grad(self):
        for k in self.g:
            self.g[k] = mx.zeros_like(self.g[k])

    def step(self, lr, clip=1.0, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam with gradient clipping."""
        total_norm = mx.sqrt(sum((v ** 2).sum() for v in self.g.values()))
        if total_norm.item() > clip:
            scale = clip / (total_norm + 1e-8)
            for k in self.g:
                self.g[k] = self.g[k] * scale

        self.t += 1
        bc1 = 1 - beta1 ** self.t
        bc2 = 1 - beta2 ** self.t

        for k in self.p:
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * self.g[k]
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * self.g[k] ** 2
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            self.p[k] = self.p[k] - lr * m_hat / (mx.sqrt(v_hat) + eps)

        mx.eval(list(self.p.values()))

    def save(self, path):
        """Save checkpoint as numpy .npz (atomic write + CRC verify), compatible with model.py."""
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
                arrays[f'p_{k}'] = np.array(v)
            for k, v in self.m.items():
                arrays[f'm_{k}'] = np.array(v)
            for k, v in self.v.items():
                arrays[f'v_{k}'] = np.array(v)
            np.savez_compressed(tmp, **arrays)
            import zipfile
            z = zipfile.ZipFile(tmp)
            bad = z.testzip()
            z.close()
            if bad is not None:
                os.remove(tmp)
                raise IOError(f"checkpoint write failed verification at {bad}; temp removed")
        else:
            data = {k: np.array(v).tolist() for k, v in self.p.items()}
            data['__config__'] = {
                'vocab_size': self.vocab_size,
                'embed_dim': self.C,
                'n_heads': self.H,
                'n_layers': self.L,
                'seq_len': self.T,
            }
            data['__adam__'] = {
                't': self.t,
                'm': {k: np.array(v).tolist() for k, v in self.m.items()},
                'v': {k: np.array(v).tolist() for k, v in self.v.items()},
            }
            with open(tmp, 'w') as f:
                json.dump(data, f)
        os.replace(tmp, path)
        print(f"Model saved to {path}")

    def prefill(self, idx):
        logits, cache = self.forward(idx)
        kv_cache = [
            {'k': cache['block_caches'][i]['k'][0],
             'v': cache['block_caches'][i]['v'][0]}
            for i in range(self.L)
        ]
        return logits[-1], kv_cache

    def forward_one(self, token_id, position, kv_cache):
        p = self.p
        C, H, L = self.C, self.H, self.L
        hd = C // H
        pos = position % self.T

        x = p['wte'][token_id] + p['wpe'][pos]
        x = x[None, :]

        new_kv = []
        for i in range(L):
            x_ln = layer_norm(x, p[f'ln1_g_{i}'], p[f'ln1_b_{i}'])
            qkv = x_ln @ p[f'qkv_w_{i}'] + p[f'qkv_b_{i}']
            q, k, v = mx.split(qkv, 3, axis=-1)

            q = q.reshape(1, H, hd).transpose(1, 0, 2)
            k = k.reshape(1, H, hd).transpose(1, 0, 2)
            v = v.reshape(1, H, hd).transpose(1, 0, 2)

            k_full = mx.concatenate([kv_cache[i]['k'], k], axis=1) if kv_cache else k
            v_full = mx.concatenate([kv_cache[i]['v'], v], axis=1) if kv_cache else v

            if k_full.shape[1] > self.T:
                k_full = k_full[:, -self.T:, :]
                v_full = v_full[:, -self.T:, :]

            new_kv.append({'k': k_full, 'v': v_full})

            scale = 1.0 / np.sqrt(hd)
            attn_scores = q @ k_full.transpose(0, 2, 1) * scale
            attn_w = softmax(attn_scores, axis=-1)
            out = attn_w @ v_full
            out = out.transpose(1, 0, 2).reshape(1, C)
            out = out @ p[f'attn_proj_w_{i}'] + p[f'attn_proj_b_{i}']
            x = x + out

            x_ln2 = layer_norm(x, p[f'ln2_g_{i}'], p[f'ln2_b_{i}'])
            h = x_ln2 @ p[f'fc_w_{i}'] + p[f'fc_b_{i}']
            ffn_out = gelu(h) @ p[f'fc2_w_{i}'] + p[f'fc2_b_{i}']
            x = x + ffn_out

        x = layer_norm(x, p['ln_f_g'], p['ln_f_b'])
        logits = (x @ p['wte'].T + p['proj_b'])[0]
        mx.eval(logits)
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
            m.p = {k[2:]: mx.array(data[k]) for k in data if k.startswith('p_')}
            m.m = {k[2:]: mx.array(data[k]) for k in data if k.startswith('m_')}
            m.v = {k[2:]: mx.array(data[k]) for k in data if k.startswith('v_')}
            m.g = {k: mx.zeros_like(v) for k, v in m.p.items()}
        else:
            with open(path) as f:
                data = json.load(f)
            cfg = data.pop('__config__')
            adam = data.pop('__adam__', None)
            m = cls(**cfg)
            m.p = {k: mx.array(np.array(v, dtype=np.float32)) for k, v in data.items()}
            m.g = {k: mx.zeros_like(v) for k, v in m.p.items()}
            if adam:
                m.t = adam['t']
                m.m = {k: mx.array(np.array(v, dtype=np.float32)) for k, v in adam['m'].items()}
                m.v = {k: mx.array(np.array(v, dtype=np.float32)) for k, v in adam['v'].items()}
            else:
                m.m = {k: mx.zeros_like(v) for k, v in m.p.items()}
                m.v = {k: mx.zeros_like(v) for k, v in m.p.items()}
        return m

    def generate(self, tokenizer, prompt, max_new=200, temperature=0.8):
        """Generate text from a prompt string (full-ctx forward per token)."""
        STOPS = ['\nUser:', '\nJoe:']
        ids = tokenizer.encode(prompt)
        prompt_len = len(ids)
        for _ in range(max_new):
            ctx = ids[-self.T:]
            ctx_arr = mx.array(np.array(ctx, dtype=np.int32))
            logits, _ = self.forward(ctx_arr)
            last_logits = logits[-1] / temperature
            probs = softmax(last_logits)
            p = np.array(probs).astype(np.float64)
            p = p / p.sum()
            if not np.all(np.isfinite(p)):
                p = np.ones(len(p)) / len(p)
            next_id = np.random.choice(len(p), p=p)
            ids.append(int(next_id))
            cut = self._find_stop(tokenizer, ids, prompt_len, STOPS)
            if cut is not None:
                return tokenizer.decode(ids[:cut])
        return tokenizer.decode(ids)

    def _find_stop(self, tokenizer, ids, prompt_len, STOPS):
        """Return the token index where a stop sequence ends, or None."""
        tail = len(ids) - prompt_len
        if tail < 1:
            return None
        gen = tokenizer.decode(ids[prompt_len:])
        for stop in STOPS:
            i = gen.find(stop)
            if i == -1:
                continue
            pos = prompt_len
            for tk in ids[prompt_len:]:
                t = tokenizer.id_to_token.get(tk, '')
                if len(t) > i:
                    return pos
                i -= len(t)
                pos += 1
            return pos
        return None

    def generate_fast(self, tokenizer, prompt, max_new=120, temperature=0.8):
        """Fast generation using KV cache (prefill + forward_one)."""
        STOPS = ['\nUser:', '\nJoe:']
        ids = tokenizer.encode(prompt)
        prompt_len = len(ids)
        logits, kv_cache = self.prefill(mx.array(np.array(ids[-self.T:], dtype=np.int32)))
        last_logits = logits / temperature
        probs = softmax(last_logits)
        p = np.array(probs).astype(np.float64)
        p = p / p.sum()
        if not np.all(np.isfinite(p)):
            p = np.ones(len(p)) / len(p)
        next_id = int(np.random.choice(len(p), p=p))
        ids.append(next_id)
        cut = self._find_stop(tokenizer, ids, prompt_len, STOPS)
        if cut is not None:
            return tokenizer.decode(ids[:cut])
        for _ in range(max_new - 1):
            pos = len(ids) - 1
            logits, kv_cache = self.forward_one(ids[-1], pos, kv_cache)
            last_logits = logits / temperature
            probs = softmax(last_logits)
            p = np.array(probs).astype(np.float64)
            p = p / p.sum()
            if not np.all(np.isfinite(p)):
                p = np.ones(len(p)) / len(p)
            next_id = int(np.random.choice(len(p), p=p))
            ids.append(next_id)
            cut = self._find_stop(tokenizer, ids, prompt_len, STOPS)
            if cut is not None:
                return tokenizer.decode(ids[:cut])
        return tokenizer.decode(ids)