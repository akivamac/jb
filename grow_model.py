"""
Grow JoeBrain's model size without retraining from scratch.

Supports:
  --new_embed_dim   Expand embedding dimension (e.g. 128 -> 256)
  --new_layers      Add more transformer layers (e.g. 3 -> 6)

Strategy:
  - Existing weights are preserved in the top-left block of expanded matrices
  - New rows/cols are near-zero so the model behaves almost identically at first
  - New layers use zero output projections (near-identity via residual stream)
  - Adam moments are zeroed for new weights so optimizer starts fresh on them

Usage:
  python3 grow_model.py --new_embed_dim 256
  python3 grow_model.py --new_layers 6
  python3 grow_model.py --new_embed_dim 256 --new_layers 6
"""

import numpy as np
import argparse
import os

MODEL_PATH = "data/model.npz"


def load_model(path):
    """Load .npz model file."""
    data = np.load(path, allow_pickle=False)
    
    cfg = {
        'vocab_size': int(data['__vocab_size__']),
        'embed_dim': int(data['__embed_dim__']),
        'n_heads': int(data['__n_heads__']),
        'n_layers': int(data['__n_layers__']),
        'seq_len': int(data['__seq_len__']),
    }
    
    params = {}
    for key in data.files:
        if key.startswith('p_'):
            params[key[2:]] = data[key].astype(np.float32)
    
    adam_m = {}
    adam_v = {}
    adam_t = int(data['__adam_t__'])
    
    for key in data.files:
        if key.startswith('m_'):
            adam_m[key[2:]] = data[key].astype(np.float32)
        elif key.startswith('v_'):
            adam_v[key[2:]] = data[key].astype(np.float32)
    
    adam = {'t': adam_t, 'm': adam_m, 'v': adam_v}
    return cfg, params, adam


def save_model(path, cfg, params, adam):
    """Save model to .npz."""
    arrays = {}
    arrays['__vocab_size__'] = np.array(cfg['vocab_size'])
    arrays['__embed_dim__'] = np.array(cfg['embed_dim'])
    arrays['__n_heads__'] = np.array(cfg['n_heads'])
    arrays['__n_layers__'] = np.array(cfg['n_layers'])
    arrays['__seq_len__'] = np.array(cfg['seq_len'])
    arrays['__adam_t__'] = np.array(adam['t'])
    
    for k, v in params.items():
        arrays[f'p_{k}'] = v
    for k, v in adam['m'].items():
        arrays[f'm_{k}'] = v
    for k, v in adam['v'].items():
        arrays[f'v_{k}'] = v
    
    # atomic write: temp file + rename, so a crash mid-write never corrupts the checkpoint
    tmp = path[:-4] + ".tmp.npz"
    try:
        np.savez_compressed(tmp, **arrays)
        import zipfile
        z = zipfile.ZipFile(tmp)
        bad = z.testzip()
        z.close()
        if bad is not None:
            os.remove(tmp)
            raise IOError(f"write failed verification at {bad}")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"Saved to {path}")


def tiny(shape, scale=0.01):
    return (np.random.randn(*shape) * scale).astype(np.float32)

def zeros(shape):
    return np.zeros(shape, dtype=np.float32)

def ones(n):
    return np.ones(n, dtype=np.float32)


def expand_dim(params, adam, old_C, new_C, n_heads, old_L):
    """Expand embed_dim from old_C to new_C, keeping existing weights."""
    assert new_C % n_heads == 0, f"new_embed_dim {new_C} must be divisible by n_heads {n_heads}"
    extra = new_C - old_C
    p = {}
    m = {}
    v = {}

    def pad_c(w):
        """Pad a (*, old_C) tensor to (*, new_C) along last axis."""
        pad_shape = list(w.shape)
        pad_shape[-1] = extra
        return np.concatenate([w, tiny(pad_shape)], axis=-1)

    def pad_r(w):
        """Pad a (old_C, *) tensor to (new_C, *) along first axis."""
        pad_shape = list(w.shape)
        pad_shape[0] = extra
        return np.concatenate([w, tiny(pad_shape)], axis=0)

    # wte: (V, old_C) -> (V, new_C)
    p['wte'] = pad_c(params['wte'])

    # wpe: (T, old_C) -> (T, new_C)
    p['wpe'] = pad_c(params['wpe'])

    # proj_b: (V,) unchanged
    p['proj_b'] = params['proj_b'].copy()

    # Final layer norm
    p['ln_f_g'] = np.concatenate([params['ln_f_g'], ones(extra)])
    p['ln_f_b'] = np.concatenate([params['ln_f_b'], zeros(extra)])

    for i in range(old_L):
        # Layer norms (C,) -> (new_C,)
        for key in [f'ln1_g_{i}', f'ln2_g_{i}']:
            p[key] = np.concatenate([params[key], ones(extra)])
        for key in [f'ln1_b_{i}', f'ln2_b_{i}']:
            p[key] = np.concatenate([params[key], zeros(extra)])

        # qkv_w: (old_C, 3*old_C) -> (new_C, 3*new_C)
        # Layout: [Q|K|V] in last dim
        old_w = params[f'qkv_w_{i}']  # (old_C, 3*old_C)
        new_w = zeros((new_C, 3 * new_C))
        # Copy each of Q, K, V blocks
        for h in range(3):
            new_w[:old_C, h*new_C:h*new_C+old_C] = old_w[:, h*old_C:(h+1)*old_C]
        # Fill new rows and new cols with tiny random
        new_w[old_C:, :] = tiny((extra, 3 * new_C))
        for h in range(3):
            new_w[:old_C, h*new_C+old_C:(h+1)*new_C] = tiny((old_C, extra))
        p[f'qkv_w_{i}'] = new_w

        # qkv_b: (3*old_C,) -> (3*new_C,) — keep Q/K/V biases in right slots
        old_b = params[f'qkv_b_{i}']
        new_b = zeros(3 * new_C)
        for h in range(3):
            new_b[h*new_C:h*new_C+old_C] = old_b[h*old_C:(h+1)*old_C]
        p[f'qkv_b_{i}'] = new_b

        # attn_proj_w: (old_C, old_C) -> (new_C, new_C)
        new_w = zeros((new_C, new_C))
        new_w[:old_C, :old_C] = params[f'attn_proj_w_{i}']
        p[f'attn_proj_w_{i}'] = new_w

        # attn_proj_b: (old_C,) -> (new_C,)
        p[f'attn_proj_b_{i}'] = np.concatenate([params[f'attn_proj_b_{i}'], zeros(extra)])

        # fc_w: (old_C, 4*old_C) -> (new_C, 4*new_C)
        old_w = params[f'fc_w_{i}']
        new_w = zeros((new_C, 4 * new_C))
        new_w[:old_C, :4*old_C] = old_w
        new_w[old_C:, :] = tiny((extra, 4 * new_C))
        new_w[:old_C, 4*old_C:] = tiny((old_C, 4 * extra))
        p[f'fc_w_{i}'] = new_w

        # fc_b: (4*old_C,) -> (4*new_C,)
        p[f'fc_b_{i}'] = np.concatenate([params[f'fc_b_{i}'], zeros(4 * extra)])

        # fc2_w: (4*old_C, old_C) -> (4*new_C, new_C)
        old_w = params[f'fc2_w_{i}']
        new_w = zeros((4 * new_C, new_C))
        new_w[:4*old_C, :old_C] = old_w
        p[f'fc2_w_{i}'] = new_w

        # fc2_b: (old_C,) -> (new_C,)
        p[f'fc2_b_{i}'] = np.concatenate([params[f'fc2_b_{i}'], zeros(extra)])

    # Zero Adam moments for all (shapes changed)
    for key in p:
        m[key] = zeros(p[key].shape)
        v[key] = zeros(p[key].shape)

    new_adam = None
    if adam:
        new_adam = {'t': adam['t'], 'm': {k: v2.tolist() for k, v2 in m.items()},
                    'v': {k: v2.tolist() for k, v2 in v.items()}}

    return p, new_adam


def add_layers(params, adam, old_L, new_L, C):
    """Add new_L - old_L transformer layers initialized as near-identity."""
    p = dict(params)
    m_dict = {}
    v_dict = {}

    for i in range(old_L, new_L):
        p[f'ln1_g_{i}'] = ones(C)
        p[f'ln1_b_{i}'] = zeros(C)
        p[f'ln2_g_{i}'] = ones(C)
        p[f'ln2_b_{i}'] = zeros(C)

        p[f'qkv_w_{i}'] = tiny((C, 3 * C))
        p[f'qkv_b_{i}'] = zeros(3 * C)

        # Zero output projection = layer starts as identity (residual passthrough)
        p[f'attn_proj_w_{i}'] = zeros((C, C))
        p[f'attn_proj_b_{i}'] = zeros(C)

        p[f'fc_w_{i}'] = tiny((C, 4 * C))
        p[f'fc_b_{i}'] = zeros(4 * C)

        # Zero output = no FFN contribution initially
        p[f'fc2_w_{i}'] = zeros((4 * C, C))
        p[f'fc2_b_{i}'] = zeros(C)

    # Zero Adam moments for new keys only
    new_keys = set(p.keys()) - set(params.keys())
    for key in new_keys:
        m_dict[key] = zeros(p[key].shape)
        v_dict[key] = zeros(p[key].shape)

    new_adam = None
    if adam:
        new_m = {k: np.array(adam['m'][k], dtype=np.float32) for k in adam['m']}
        new_v = {k: np.array(adam['v'][k], dtype=np.float32) for k in adam['v']}
        new_m.update(m_dict)
        new_v.update(v_dict)
        new_adam = {'t': adam['t'],
                    'm': {k: v2.tolist() for k, v2 in new_m.items()},
                    'v': {k: v2.tolist() for k, v2 in new_v.items()}}

    return p, new_adam


def count_params(params):
    return sum(v.size for v in params.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_embed_dim", type=int, default=0)
    parser.add_argument("--new_layers",    type=int, default=0)
    args = parser.parse_args()

    cfg, params, adam = load_model(MODEL_PATH)

    old_C = cfg['embed_dim']
    old_L = cfg['n_layers']
    n_heads = cfg['n_heads']
    T = cfg['seq_len']

    print(f"Loaded: embed_dim={old_C}, n_layers={old_L}, n_heads={n_heads}, seq_len={T}")
    print(f"Current params: {count_params(params):,}")

    changed = False

    if args.new_embed_dim and args.new_embed_dim != old_C:
        assert args.new_embed_dim > old_C, "new_embed_dim must be larger than current"
        print(f"\nExpanding embed_dim: {old_C} -> {args.new_embed_dim}")
        params, adam = expand_dim(params, adam, old_C, args.new_embed_dim, n_heads, old_L)
        cfg['embed_dim'] = args.new_embed_dim
        old_C = args.new_embed_dim
        changed = True

    if args.new_layers and args.new_layers != old_L:
        assert args.new_layers > old_L, "new_layers must be larger than current"
        print(f"\nAdding layers: {old_L} -> {args.new_layers}")
        params, adam = add_layers(params, adam, old_L, args.new_layers, old_C)
        cfg['n_layers'] = args.new_layers
        changed = True

    if not changed:
        print("Nothing to do. Pass --new_embed_dim or --new_layers.")
        return

    print(f"\nNew params: {count_params(params):,}")
    save_model(MODEL_PATH, cfg, params, adam)
    print("Done. Existing knowledge preserved. Resume training to adapt new capacity.")


if __name__ == "__main__":
    main()
