"""
Extend the positional embeddings (wpe) of a saved JoeBrain model
to support a longer sequence length — no retraining required.

Existing weights are preserved exactly. New position rows are
initialized by extrapolating from the last few existing rows,
so the model has a reasonable starting point for new positions.

Usage:
    python3 extend_seq_len.py --new_seq_len 256
    python3 extend_seq_len.py --new_seq_len 512
"""

import numpy as np
import argparse
import os

MODEL_PATH = "data/model.npz"

def load_model(path):
    """Load .npz model file."""
    data = np.load(path, allow_pickle=False)

    # Extract config from metadata
    cfg = {
        'vocab_size': int(data['__vocab_size__']),
        'embed_dim': int(data['__embed_dim__']),
        'n_heads': int(data['__n_heads__']),
        'n_layers': int(data['__n_layers__']),
        'seq_len': int(data['__seq_len__']),
    }

    # Extract parameters (p_ prefix)
    params = {}
    for key in data.files:
        if key.startswith('p_'):
            params[key[2:]] = data[key].astype(np.float32)

    # Extract Adam state
    adam_m = {}
    adam_v = {}
    adam_t = int(data['__adam_t__'])

    for key in data.files:
        if key.startswith('m_'):
            adam_m[key[2:]] = data[key].astype(np.float32)
        elif key.startswith('v_'):
            adam_v[key[2:]] = data[key].astype(np.float32)

    return cfg, params, adam_m, adam_v, adam_t

def save_model(path, cfg, params, adam_m, adam_v, adam_t):
    """Save model to .npz."""
    arrays = {}
    arrays['__vocab_size__'] = np.array(cfg['vocab_size'])
    arrays['__embed_dim__'] = np.array(cfg['embed_dim'])
    arrays['__n_heads__'] = np.array(cfg['n_heads'])
    arrays['__n_layers__'] = np.array(cfg['n_layers'])
    arrays['__seq_len__'] = np.array(cfg['seq_len'])
    arrays['__adam_t__'] = np.array(adam_t)

    for k, v in params.items():
        arrays[f'p_{k}'] = v
    for k, v in adam_m.items():
        arrays[f'm_{k}'] = v
    for k, v in adam_v.items():
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

def extend_wpe(wpe, new_T):
    old_T, C = wpe.shape
    if new_T <= old_T:
        print(f"seq_len is already {old_T}, nothing to do.")
        return wpe

    # Extrapolate: use the trend from the last few rows
    extra = new_T - old_T
    print(f"Extending wpe from {old_T} -> {new_T} (+{extra} rows)")

    # Linear extrapolation from last 8 rows (or fewer if model is tiny)
    window = min(8, old_T)
    tail = wpe[-window:]  # (window, C)
    # mean step direction over the tail
    step = (tail[-1] - tail[0]) / max(window - 1, 1)

    new_rows = np.stack([
        wpe[-1] + step * (i + 1) for i in range(extra)
    ], axis=0).astype(np.float32)

    return np.concatenate([wpe, new_rows], axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_seq_len", type=int, required=True,
                        help="Target sequence length (must be > current)")
    parser.add_argument("--input", type=str, default=MODEL_PATH,
                        help="Input .npz file path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz file path (defaults to input)")
    args = parser.parse_args()

    out_path = args.output if args.output else args.input

    print(f"Loading model from {args.input}...")
    # Unpack 5 values now
    cfg, params, adam_m, adam_v, adam_t = load_model(args.input)

    wpe = params["wpe"]
    old_T = wpe.shape[0]

    if args.new_seq_len <= old_T:
        print(f"Model already has seq_len={old_T}. Choose > {old_T}.")
        return

    # Extend wpe
    params["wpe"] = extend_wpe(wpe, args.new_seq_len)
    cfg["seq_len"] = args.new_seq_len
    print(f"Updated config seq_len: {old_T} -> {args.new_seq_len}")

    # Extend Adam moments for wpe if present
    if "wpe" in adam_m:
        old_m = adam_m["wpe"]
        old_v = adam_v["wpe"]
        extra = args.new_seq_len - old_T

        # Initialize new momentum/velocity to 0
        new_m = np.concatenate([old_m, np.zeros((extra, old_m.shape[1]), dtype=np.float32)], axis=0)
        new_v = np.concatenate([old_v, np.zeros((extra, old_v.shape[1]), dtype=np.float32)], axis=0)

        adam_m["wpe"] = new_m
        adam_v["wpe"] = new_v
        print("Extended Adam momentum and velocity buffers.")

    print(f"Saving extended model to {out_path}...")
    save_model(out_path, cfg, params, adam_m, adam_v, adam_t)
    print("Done.")

if __name__ == "__main__":
    main()
