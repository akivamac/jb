import json, numpy as np, os, io
with open('data/model.json') as f:
    data = json.load(f)
params = {k: np.array(v, dtype=np.float32) for k, v in data.items() if not k.startswith('__')}
buf = io.BytesIO()
np.savez_compressed(buf, **params)
npz_size = buf.tell()
json_size = os.path.getsize('data/model.json')
print(f'JSON: {json_size/1024/1024:.1f} MB, NPZ: {npz_size/1024/1024:.1f} MB, Reduction: {(1 - npz_size/json_size)*100:.1f}%')
