"""
Grow an expert model's capacity without retraining from scratch.

Preserves existing weights and Adam momentum. New weights are near-zero.

Usage:
  python3 grow_expert.py --name fish --new_embed_dim 256
  python3 grow_expert.py --name fish --new_layers 6
  python3 grow_expert.py --name fish --new_embed_dim 256 --new_layers 4

Current expert config: embed_dim=128, n_heads=4, n_layers=3, ~869K params
Recommended: --new_embed_dim 256 --new_layers 4 → ~4M params
"""

import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
from model import JoeBrain

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def tiny(shape, scale=0.01):
    return (np.random.randn(*shape) * scale).astype(np.float32)


def zeros(shape):
    return np.zeros(shape, dtype=np.float32)


def ones(n):
    return np.ones(n, dtype=np.float32)


def pad_last(w, new_last, fill='tiny'):
    """Pad last axis from old_size to new_last, preserving existing values.
    New entries use tiny() (weights) or zeros (Adam moments) via fill='zero'."""
    old_last = w.shape[-1]
    if new_last == old_last:
        return w.copy()
    pad_shape = list(w.shape)
    pad_shape[-1] = new_last - old_last
    filler = zeros if fill == 'zero' else tiny
    return np.concatenate([w, filler(pad_shape)], axis=-1)

def pad_first(w, new_first):
    """Pad first axis from old_size to new_first, preserving existing values."""
    old_first = w.shape[0]
    if new_first == old_first:
        return w.copy()
    pad_shape = list(w.shape)
    pad_shape[0] = new_first - old_first
    return np.concatenate([w, tiny(pad_shape)], axis=0)

def pad_both_2d(w, new_r, new_c, fill='tiny'):
    """Pad a 2D matrix from (old_r, old_c) to (new_r, new_c).
    New entries use tiny() (weights) or zeros (Adam moments) via fill='zero'."""
    old_r, old_c = w.shape
    result = zeros((new_r, new_c))
    result[:old_r, :old_c] = w
    filler = zeros if fill == 'zero' else tiny
    if old_r < new_r:
        result[old_r:, :old_c] = filler((new_r - old_r, old_c))
    if old_c < new_c:
        result[:old_r, old_c:] = filler((old_r, new_c - old_c))
    if old_r < new_r and old_c < new_c:
        result[old_r:, old_c:] = filler((new_r - old_r, new_c - old_c))
    return result


def expand_dim(model, new_C):
    """Expand embed_dim, preserving weights and Adam moments.
    New weight entries are tiny; new Adam moments are zero."""
    old_C = model.C
    L = model.L
    n_heads = model.H
    T = model.T
    V = model.vocab_size
    extra = new_C - old_C

    assert new_C % n_heads == 0, f"new_embed_dim {new_C} must be divisible by n_heads {n_heads}"
    assert new_C > old_C, "new_embed_dim must be larger"

    new_p = {}
    new_m = {}
    new_v = {}

    # Embeddings: (V, C) -> (V, new_C)
    new_p['wte'] = pad_last(model.p['wte'], new_C)
    new_m['wte'] = pad_last(model.m['wte'], new_C, fill='zero')
    new_v['wte'] = pad_last(model.v['wte'], new_C, fill='zero')

    # Position embeddings: (T, C) -> (T, new_C)
    new_p['wpe'] = pad_last(model.p['wpe'], new_C)
    new_m['wpe'] = pad_last(model.m['wpe'], new_C, fill='zero')
    new_v['wpe'] = pad_last(model.v['wpe'], new_C, fill='zero')

    # Output projection bias: (V,) unchanged
    new_p['proj_b'] = model.p['proj_b'].copy()
    new_m['proj_b'] = model.m['proj_b'].copy()
    new_v['proj_b'] = model.v['proj_b'].copy()

    # Final layer norm: (C,) -> (new_C,)
    for key in ['ln_f_g', 'ln_f_b']:
        new_p[key] = pad_last(model.p[key], new_C)
        new_m[key] = pad_last(model.m[key], new_C, fill='zero')
        new_v[key] = pad_last(model.v[key], new_C, fill='zero')

    for i in range(L):
        # Layer norms: (C,) -> (new_C,)
        for key in ['ln1_g', 'ln1_b', 'ln2_g', 'ln2_b']:
            full = f'{key}_{i}'
            new_p[full] = pad_last(model.p[full], new_C)
            new_m[full] = pad_last(model.m[full], new_C, fill='zero')
            new_v[full] = pad_last(model.v[full], new_C, fill='zero')

        # qkv_w: (C, 3*C) -> (new_C, 3*new_C)
        old_w = model.p[f'qkv_w_{i}']  # (old_C, 3*old_C)
        new_w = zeros((new_C, 3 * new_C))
        for h in range(3):
            new_w[:old_C, h*new_C:h*new_C+old_C] = old_w[:, h*old_C:(h+1)*old_C]
            # Copy Adam moments for existing block
        new_p[f'qkv_w_{i}'] = new_w
        # Adam: pad each Q,K,V block
        old_m = model.m[f'qkv_w_{i}']
        old_v_arr = model.v[f'qkv_w_{i}']
        new_m_w = zeros((new_C, 3 * new_C))
        new_v_w = zeros((new_C, 3 * new_C))
        for h in range(3):
            new_m_w[:old_C, h*new_C:h*new_C+old_C] = old_m[:, h*old_C:(h+1)*old_C]
            new_v_w[:old_C, h*new_C:h*new_C+old_C] = old_v_arr[:, h*old_C:(h+1)*old_C]
        new_m[f'qkv_w_{i}'] = new_m_w
        new_v[f'qkv_w_{i}'] = new_v_w

        # qkv_b: (3*C,) -> (3*new_C,)
        old_b = model.p[f'qkv_b_{i}']
        new_b = zeros(3 * new_C)
        new_mb = zeros(3 * new_C)
        new_vb = zeros(3 * new_C)
        for h in range(3):
            new_b[h*new_C:h*new_C+old_C] = old_b[h*old_C:(h+1)*old_C]
            new_mb[h*new_C:h*new_C+old_C] = model.m[f'qkv_b_{i}'][h*old_C:(h+1)*old_C]
            new_vb[h*new_C:h*new_C+old_C] = model.v[f'qkv_b_{i}'][h*old_C:(h+1)*old_C]
        new_p[f'qkv_b_{i}'] = new_b
        new_m[f'qkv_b_{i}'] = new_mb
        new_v[f'qkv_b_{i}'] = new_vb

        # attn_proj_w: (C, C) -> (new_C, new_C)
        new_p[f'attn_proj_w_{i}'] = pad_both_2d(model.p[f'attn_proj_w_{i}'], new_C, new_C)
        new_m[f'attn_proj_w_{i}'] = pad_both_2d(model.m[f'attn_proj_w_{i}'], new_C, new_C, fill='zero')
        new_v[f'attn_proj_w_{i}'] = pad_both_2d(model.v[f'attn_proj_w_{i}'], new_C, new_C, fill='zero')

        # attn_proj_b: (C,) -> (new_C,)
        new_p[f'attn_proj_b_{i}'] = pad_last(model.p[f'attn_proj_b_{i}'], new_C)
        new_m[f'attn_proj_b_{i}'] = pad_last(model.m[f'attn_proj_b_{i}'], new_C, fill='zero')
        new_v[f'attn_proj_b_{i}'] = pad_last(model.v[f'attn_proj_b_{i}'], new_C, fill='zero')

        # fc_w: (C, 4*C) -> (new_C, 4*new_C)
        new_p[f'fc_w_{i}'] = pad_both_2d(model.p[f'fc_w_{i}'], new_C, 4 * new_C)
        new_m[f'fc_w_{i}'] = pad_both_2d(model.m[f'fc_w_{i}'], new_C, 4 * new_C, fill='zero')
        new_v[f'fc_w_{i}'] = pad_both_2d(model.v[f'fc_w_{i}'], new_C, 4 * new_C, fill='zero')

        # fc_b: (4*C,) -> (4*new_C,)
        new_p[f'fc_b_{i}'] = pad_last(model.p[f'fc_b_{i}'], 4 * new_C)
        new_m[f'fc_b_{i}'] = pad_last(model.m[f'fc_b_{i}'], 4 * new_C, fill='zero')
        new_v[f'fc_b_{i}'] = pad_last(model.v[f'fc_b_{i}'], 4 * new_C, fill='zero')

        # fc2_w: (4*C, C) -> (4*new_C, new_C)
        new_p[f'fc2_w_{i}'] = pad_both_2d(model.p[f'fc2_w_{i}'], 4 * new_C, new_C)
        new_m[f'fc2_w_{i}'] = pad_both_2d(model.m[f'fc2_w_{i}'], 4 * new_C, new_C, fill='zero')
        new_v[f'fc2_w_{i}'] = pad_both_2d(model.v[f'fc2_w_{i}'], 4 * new_C, new_C, fill='zero')

        # fc2_b: (C,) -> (new_C,)
        new_p[f'fc2_b_{i}'] = pad_last(model.p[f'fc2_b_{i}'], new_C)
        new_m[f'fc2_b_{i}'] = pad_last(model.m[f'fc2_b_{i}'], new_C, fill='zero')
        new_v[f'fc2_b_{i}'] = pad_last(model.v[f'fc2_b_{i}'], new_C, fill='zero')

    model.C = new_C
    model.head_dim = new_C // n_heads
    model.p = new_p
    model.m = new_m
    model.v = new_v
    model.g = {k: np.zeros_like(v) for k, v in model.p.items()}


def add_layers(model, new_L):
    """Add transformer layers as near-identity (zero output projections)."""
    old_L = model.L
    C = model.C
    assert new_L > old_L

    for i in range(old_L, new_L):
        model.p[f'ln1_g_{i}'] = ones(C)
        model.p[f'ln1_b_{i}'] = zeros(C)
        model.p[f'ln2_g_{i}'] = ones(C)
        model.p[f'ln2_b_{i}'] = zeros(C)

        model.p[f'qkv_w_{i}'] = tiny((C, 3 * C))
        model.p[f'qkv_b_{i}'] = zeros(3 * C)
        model.p[f'attn_proj_w_{i}'] = zeros((C, C))  # zero = identity via residual
        model.p[f'attn_proj_b_{i}'] = zeros(C)

        model.p[f'fc_w_{i}'] = tiny((C, 4 * C))
        model.p[f'fc_b_{i}'] = zeros(4 * C)
        model.p[f'fc2_w_{i}'] = zeros((4 * C, C))  # zero = no FFN initially
        model.p[f'fc2_b_{i}'] = zeros(C)

        # Zero Adam moments for new layers
        for key in ['ln1_g', 'ln1_b', 'ln2_g', 'ln2_b',
                     'qkv_w', 'qkv_b', 'attn_proj_w', 'attn_proj_b',
                     'fc_w', 'fc_b', 'fc2_w', 'fc2_b']:
            full = f'{key}_{i}'
            model.m[full] = zeros(model.p[full].shape)
            model.v[full] = zeros(model.p[full].shape)

    model.L = new_L


def main():
    parser = argparse.ArgumentParser(description='Grow an expert model')
    parser.add_argument('--name', required=True, help='Expert name')
    parser.add_argument('--new_embed_dim', type=int, default=0)
    parser.add_argument('--new_layers', type=int, default=0)
    args = parser.parse_args()

    expert_dir = os.path.join(BASE, 'experts', args.name)
    model_path = os.path.join(expert_dir, f'{args.name}.npz')

    if not os.path.exists(model_path):
        print(f"ERROR: No checkpoint at {model_path}")
        sys.exit(1)

    model = JoeBrain.load(model_path)
    old_params = sum(v.size for v in model.p.values())
    print(f"Loaded '{args.name}': dim={model.C}, layers={model.L}, heads={model.H}, params={old_params:,}")

    if args.new_embed_dim and args.new_embed_dim != model.C:
        assert args.new_embed_dim > model.C
        print(f"\nExpanding embed_dim: {model.C} -> {args.new_embed_dim}")
        expand_dim(model, args.new_embed_dim)

    if args.new_layers and args.new_layers != model.L:
        assert args.new_layers > model.L
        print(f"Adding layers: {model.L} -> {args.new_layers}")
        add_layers(model, args.new_layers)

    new_params = sum(v.size for v in model.p.values())
    print(f"New config: dim={model.C}, layers={model.L}, heads={model.H}, params={new_params:,}")
    print(f"Growth: {old_params:,} -> {new_params:,} ({new_params/old_params:.1f}x)")

    model.save(model_path)
    print(f"Saved to {model_path}")
    print("Existing knowledge preserved. Resume training to adapt new capacity.")


if __name__ == '__main__':
    main()
