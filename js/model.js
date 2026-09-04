/**
 * JoeBrain — browser inference engine
 * Loads model.json + tokenizer.json and generates text character by character.
 * Pure JS, no external dependencies.
 */

const JoeBrain = (() => {

  let p = {};      // model weights
  let cfg = {};    // model config
  let charToId = {};
  let idToChar = {};
  let ready = false;

  // ── Math helpers ──────────────────────────────────────────────────────────

  function matmul(A, B, rowsA, colsA, colsB) {
    // A: rowsA x colsA, B: colsA x colsB → out: rowsA x colsB
    const out = new Float32Array(rowsA * colsB);
    for (let i = 0; i < rowsA; i++) {
      for (let k = 0; k < colsA; k++) {
        const a = A[i * colsA + k];
        if (a === 0) continue;
        for (let j = 0; j < colsB; j++) {
          out[i * colsB + j] += a * B[k * colsB + j];
        }
      }
    }
    return out;
  }

  function addBias(x, b, T, C) {
    const out = new Float32Array(x);
    for (let t = 0; t < T; t++)
      for (let c = 0; c < C; c++)
        out[t * C + c] += b[c];
    return out;
  }

  function layerNorm(x, g, b, T, C, eps = 1e-5) {
    const out = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
      let mean = 0, variance = 0;
      for (let c = 0; c < C; c++) mean += x[t * C + c];
      mean /= C;
      for (let c = 0; c < C; c++) {
        const d = x[t * C + c] - mean;
        variance += d * d;
      }
      variance /= C;
      const std = Math.sqrt(variance + eps);
      for (let c = 0; c < C; c++)
        out[t * C + c] = g[c] * (x[t * C + c] - mean) / std + b[c];
    }
    return out;
  }

  function gelu(x) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      out[i] = 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v)));
    }
    return out;
  }

  function softmax(x, rows, cols) {
    const out = new Float32Array(x.length);
    for (let r = 0; r < rows; r++) {
      let max = -Infinity;
      for (let c = 0; c < cols; c++) max = Math.max(max, x[r * cols + c]);
      let sum = 0;
      for (let c = 0; c < cols; c++) {
        out[r * cols + c] = Math.exp(x[r * cols + c] - max);
        sum += out[r * cols + c];
      }
      for (let c = 0; c < cols; c++) out[r * cols + c] /= sum;
    }
    return out;
  }

  // ── Forward pass ──────────────────────────────────────────────────────────

  function forward(idx) {
    const T = idx.length;
    const { embed_dim: C, n_heads: H, n_layers: L } = cfg;
    const headDim = C / H;

    // Token + position embeddings
    let x = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
      const tok = p.wte[idx[t]];   // C floats
      const pos = p.wpe[t];         // C floats
      for (let c = 0; c < C; c++)
        x[t * C + c] = tok[c] + pos[c];
    }

    // Transformer blocks
    for (let i = 0; i < L; i++) {
      // --- Self-attention ---
      const xLn1 = layerNorm(x, p[`ln1_g_${i}`], p[`ln1_b_${i}`], T, C);

      // QKV projection
      const qkvW = p[`qkv_w_${i}`]; // C x 3C (row-major)
      const qkvB = p[`qkv_b_${i}`]; // 3C
      let qkv = matmul(xLn1, qkvW.flat ? qkvW : flattenWeights(qkvW, C, 3*C), T, C, 3 * C);
      qkv = addBias(qkv, qkvB, T, 3 * C);

      // Split Q K V and compute attention per head
      const attnOut = new Float32Array(T * C);
      for (let h = 0; h < H; h++) {
        // Extract head slices
        const Q = new Float32Array(T * headDim);
        const K = new Float32Array(T * headDim);
        const V = new Float32Array(T * headDim);
        for (let t = 0; t < T; t++) {
          for (let d = 0; d < headDim; d++) {
            Q[t * headDim + d] = qkv[t * 3 * C + h * headDim + d];
            K[t * headDim + d] = qkv[t * 3 * C + C + h * headDim + d];
            V[t * headDim + d] = qkv[t * 3 * C + 2 * C + h * headDim + d];
          }
        }

        // Attention scores: Q @ K^T / sqrt(headDim)
        const scale = 1.0 / Math.sqrt(headDim);
        const scores = new Float32Array(T * T);
        for (let ti = 0; ti < T; ti++) {
          for (let tj = 0; tj < T; tj++) {
            if (tj > ti) { scores[ti * T + tj] = -1e9; continue; } // causal mask
            let s = 0;
            for (let d = 0; d < headDim; d++)
              s += Q[ti * headDim + d] * K[tj * headDim + d];
            scores[ti * T + tj] = s * scale;
          }
        }

        const attnW = softmax(scores, T, T);

        // attnW @ V
        for (let ti = 0; ti < T; ti++) {
          for (let d = 0; d < headDim; d++) {
            let val = 0;
            for (let tj = 0; tj <= ti; tj++)
              val += attnW[ti * T + tj] * V[tj * headDim + d];
            attnOut[ti * C + h * headDim + d] += val;
          }
        }
      }

      // Attention output projection
      const apW = p[`attn_proj_w_${i}`];
      const apB = p[`attn_proj_b_${i}`];
      let attnProj = matmul(attnOut, apW.flat ? apW : flattenWeights(apW, C, C), T, C, C);
      attnProj = addBias(attnProj, apB, T, C);

      // Residual
      for (let j = 0; j < T * C; j++) x[j] += attnProj[j];

      // --- FFN ---
      const xLn2 = layerNorm(x, p[`ln2_g_${i}`], p[`ln2_b_${i}`], T, C);

      const fcW = p[`fc_w_${i}`];
      const fcB = p[`fc_b_${i}`];
      let h1 = matmul(xLn2, fcW.flat ? fcW : flattenWeights(fcW, C, 4 * C), T, C, 4 * C);
      h1 = addBias(h1, fcB, T, 4 * C);
      h1 = gelu(h1);

      const fc2W = p[`fc2_w_${i}`];
      const fc2B = p[`fc2_b_${i}`];
      let h2 = matmul(h1, fc2W.flat ? fc2W : flattenWeights(fc2W, 4 * C, C), T, 4 * C, C);
      h2 = addBias(h2, fc2B, T, C);

      // Residual
      for (let j = 0; j < T * C; j++) x[j] += h2[j];
    }

    // Final layer norm
    x = layerNorm(x, p.ln_f_g, p.ln_f_b, T, C);

    // Project to vocab — only need last token
    const last = x.slice((T - 1) * C, T * C);
    const projW = p.proj_w; // C x V
    const V = cfg.vocab_size;
    const logits = new Float32Array(V);
    for (let v = 0; v < V; v++) {
      let s = p.proj_b[v];
      for (let c = 0; c < C; c++) s += last[c] * projW[c][v];
      logits[v] = s;
    }
    return logits;
  }

  function flattenWeights(w, rows, cols) {
    // w is array of arrays → Float32Array row-major
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        out[r * cols + c] = w[r][c];
    return out;
  }

  function sample(logits, temperature = 0.8, recentIds = [], penalty = 1.3) {
    const V = logits.length;
    // Repetition penalty — divide logit of recently used tokens
    const penalized = new Float32Array(logits);
    for (const id of recentIds) {
      penalized[id] = penalized[id] > 0
        ? penalized[id] / penalty
        : penalized[id] * penalty;
    }
    // Apply temperature
    let max = -Infinity;
    for (let i = 0; i < V; i++) max = Math.max(max, penalized[i]);
    const probs = new Float32Array(V);
    let sum = 0;
    for (let i = 0; i < V; i++) {
      probs[i] = Math.exp((penalized[i] - max) / temperature);
      sum += probs[i];
    }
    for (let i = 0; i < V; i++) probs[i] /= sum;
    // Sample
    let r = Math.random(), cumul = 0;
    for (let i = 0; i < V; i++) {
      cumul += probs[i];
      if (r < cumul) return i;
    }
    return V - 1;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  async function load(modelUrl, tokenizerUrl) {
    const [modelData, tokData] = await Promise.all([
      fetch(modelUrl).then(r => r.json()),
      fetch(tokenizerUrl).then(r => r.json()),
    ]);

    cfg = modelData.__config__;

    // Pre-flatten 2D weight matrices for speed
    const C = cfg.embed_dim, V = cfg.vocab_size, L = cfg.n_layers, T = cfg.seq_len;
    p = {};
    for (const [k, v] of Object.entries(modelData)) {
      if (k === '__config__') continue;
      p[k] = Array.isArray(v[0]) ? v : new Float32Array(v);
    }

    charToId = tokData.char_to_id;
    idToChar = {};
    for (const [k, v] of Object.entries(tokData.id_to_char)) idToChar[parseInt(k)] = v;

    ready = true;
    console.log(`JoeBrain loaded: ${cfg.vocab_size} vocab, ${cfg.n_layers} layers, ${cfg.embed_dim} dim`);
  }

  function encode(text) {
    return [...text].map(c => charToId[c] ?? 0);
  }

  function decode(ids) {
    return ids.map(i => idToChar[i] ?? '?').join('');
  }

  // Characters to never generate
  const BANNED_CHARS = new Set(['😂','🍌','📁','💻','👨','👧','👦','👩','🔍','✏️','📷','💬','🐒','😂','⌋','⌊','∛','√','∫','≤','≥','∧','∨','∩','∪','⇒','↔','→','∂','ℵ','ℏ','Λ','Σ','Ω','Φ','Δ','Γ','Θ','α','β','γ','δ','ε','ζ','η','θ','λ','μ','ν','ξ','π','ρ','σ','τ','φ','χ','ψ','ω','⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹','₀','₁','₂','₃','₄','₅','₆','₇','₈','₉']);

  function* generateStream(prompt, { maxNew = 200, temperature = 0.8 } = {}) {
    if (!ready) throw new Error('Model not loaded');
    const ids = encode(prompt);
    const seqLen = cfg.seq_len;

    // Build set of banned token IDs — ban all non-ASCII chars
    const bannedIds = new Set();
    for (const [ch, id] of Object.entries(charToId)) {
      if (ch.charCodeAt(0) > 127) bannedIds.add(id);
    }

    for (let i = 0; i < maxNew; i++) {
      const ctx = ids.slice(-seqLen);
      const logits = forward(ctx);

      // Ban blacklisted chars
      for (const id of bannedIds) logits[id] = -1e9;

      // Ban any token seen 3+ times in last 10
      const recent10 = ids.slice(-10);
      const counts = {};
      for (const id of recent10) counts[id] = (counts[id] || 0) + 1;
      for (const [id, cnt] of Object.entries(counts)) {
        if (cnt >= 3) logits[parseInt(id)] = -1e9;
      }

      const next = sample(logits, temperature, ids.slice(-30), 2.0);
      ids.push(next);
      yield idToChar[next] ?? '?';
    }
  }

  async function generate(prompt, opts = {}) {
    let out = '';
    for (const ch of generateStream(prompt, opts)) out += ch;
    return out;
  }

  return { load, encode, decode, generate, generateStream, get ready() { return ready; } };
})();
