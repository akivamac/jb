"""
MoE Ensemble — ExpertRegistry.

Loads all enabled experts, runs quality filter, blends logits.
No router. Every expert always runs; quality filter silences junk.
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import JoeBrain


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class ExpertRegistry:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.experts = []        # list of {'name': str, 'model': JoeBrain}
        self.filter_cfg = {}     # quality filter config
        self._load()

    def _load(self):
        cfg_path = os.path.join(self.data_dir, 'experts.json')
        with open(cfg_path) as f:
            cfg = json.load(f)

        self.filter_cfg = cfg.get('quality_filter', {})

        for entry in cfg.get('experts', []):
            if not entry.get('enabled', True):
                continue
            name = entry['name']
            model_path = os.path.join(self.data_dir, entry['file'])
            if not os.path.exists(model_path):
                print(f"[ExpertRegistry] WARNING: expert '{name}' not found at {model_path}, skipping")
                continue
            model = JoeBrain.load(model_path)
            self.experts.append({'name': name, 'model': model})
            print(f"[ExpertRegistry] Loaded expert: {name} ({sum(v.size for v in model.p.values()):,} params)")

        if not self.experts:
            raise RuntimeError(f"No experts found in {self.data_dir}. Check experts.json and expert .npz files.")

        print(f"[ExpertRegistry] {len(self.experts)} expert(s) loaded")

    def quality_check(self, logits):
        """
        Score a single expert's logits. Returns (pass, score).
        pass=True means the expert is producing reasonable output.
        """
        if not self.filter_cfg.get('enabled', True):
            return True, 1.0

        probs = softmax(logits)
        V = len(logits)

        max_logit = float(logits.max())
        entropy = float(-(probs * np.log(probs + 1e-9)).sum())
        max_prob = float(probs.max())
        max_entropy = np.log(V)

        threshold = self.filter_cfg.get('max_logit_threshold', -2.0)
        ent_ratio = self.filter_cfg.get('entropy_ratio', 0.85)
        min_prob = self.filter_cfg.get('min_top_prob', 0.05)

        passed = True
        if max_logit < threshold:
            passed = False
        if entropy > ent_ratio * max_entropy:
            passed = False
        if max_prob < min_prob:
            passed = False

        # Simple repetition check: if top-3 token probs are all same
        sorted_probs = np.sort(probs)[::-1]
        if len(sorted_probs) >= 3 and sorted_probs[0] == sorted_probs[1] == sorted_probs[2] and sorted_probs[0] > 0.3:
            passed = False

        return passed, max_prob

    def blend_logits(self, all_logits):
        """
        Given list of (name, logits, passed) tuples, blend surviving experts.
        Weights each expert by its top probability (confidence) so confident
        experts dominate over confused ones.
        Returns blended logits.
        """
        survivors = [(name, logits) for name, logits, passed in all_logits if passed]

        if not survivors:
            # All experts failed quality check — fall back to equal blend of all
            all_l = np.stack([l for _, l, _ in all_logits])
            return all_l.mean(axis=0)

        if len(survivors) == 1:
            return survivors[0][1]

        # Weight by confidence: top probability of each survivor
        probs = []
        for _, logits in survivors:
            p = softmax(logits)
            probs.append(float(p.max()))
        weights = np.array(probs)
        weights = weights / weights.sum()

        stacked = np.stack([l for _, l in survivors])
        return (stacked * weights[:, None]).sum(axis=0)

    def run_ensemble(self, prompt_ids):
        """
        Run all experts on the same input, quality filter, blend.
        prompt_ids: (T,) int array
        Returns: (V,) blended logits
        """
        results = []
        for expert in self.experts:
            model = expert['model']
            logits, _ = model.forward(prompt_ids)  # (T, V)
            last_logits = logits[-1]  # (V,)
            passed, score = self.quality_check(last_logits)
            results.append((expert['name'], last_logits, passed))

        return self.blend_logits(results)

    def prefill_ensemble(self, prompt_ids):
        """
        Run all experts' prefill, quality filter last-token logits,
        return blended last-token logits and list of per-expert KV caches.
        Returns: (V,) blended logits, list of (name, model, kv_cache) for surviving experts
        """
        all_prefills = []
        for expert in self.experts:
            model = expert['model']
            logits, kv = model.prefill(prompt_ids)
            last_logits = logits  # (V,) — prefill already returns last position
            passed, score = self.quality_check(last_logits)
            all_prefills.append((expert['name'], last_logits, passed, model, kv))

        # Blend logits
        blended = self.blend_logits([(n, l, p) for n, l, p, _, _ in all_prefills])

        # Return all models + caches for forward_one (even failed ones, but mark them)
        surviving = [(n, m, kv) for n, _, p, m, kv in all_prefills if p]

        # If none survived, keep all
        if not surviving:
            surviving = [(n, m, kv) for n, _, _, m, kv in all_prefills]

        return blended, surviving

    def forward_one_ensemble(self, token_id, position, expert_caches):
        """
        Single-token forward for all surviving experts.
        expert_caches: list of (name, model, kv_cache)
        Returns: (V,) blended logits, updated expert_caches
        """
        all_logits = []
        updated = []
        for name, model, kv in expert_caches:
            logits, new_kv = model.forward_one(token_id, position, kv)
            all_logits.append((name, logits, True))  # trust surviving experts
            updated.append((name, model, new_kv))

        blended = self.blend_logits(all_logits)
        return blended, updated

    @property
    def T(self):
        """Sequence length — use smallest across experts."""
        return min(e['model'].T for e in self.experts)

    @property
    def expert_count(self):
        return len(self.experts)

    def expert_names(self):
        return [e['name'] for e in self.experts]
