"""
Hidden-state extraction for encoder and decoder models.

Pooling is separated from the forward pass so it can be tested without weights
(this container has no model access; see README). The pad-masking in
`pool_hidden` is the part that silently corrupts results if it is wrong:
mean-pooling over padded positions makes every short sentence look alike and
manufactures cross-script similarity that is really just shared padding.
"""
import numpy as np


def pool_hidden(hidden, attention_mask, strategy="mean"):
    """
    hidden:         (batch, seq, dim)  array-like
    attention_mask: (batch, seq)       1 for real tokens, 0 for padding
    strategy:       "mean" | "last" | "cls"

    Returns (batch, dim).

      mean  - average over real tokens only. Default for encoders.
      last  - final real token. Default for causal decoders; this is what
              Karne (2026) used for the Serbian SAE comparison, so use it when
              contrasting against that work.
      cls   - position 0. Only meaningful for models with a CLS token.
    """
    h = np.asarray(hidden, dtype=np.float64)
    m = np.asarray(attention_mask)

    if h.ndim != 3:
        raise ValueError(f"hidden must be (batch, seq, dim), got {h.shape}")
    if m.shape != h.shape[:2]:
        raise ValueError(f"mask {m.shape} does not match hidden {h.shape[:2]}")
    if not np.all(m.sum(axis=1) > 0):
        raise ValueError("at least one sequence has zero real tokens")

    if strategy == "cls":
        return h[:, 0, :]

    if strategy == "last":
        # Position of the LAST real token, correct under left OR right padding.
        # (mask.sum()-1 is only right for right padding; several decoder
        # tokenisers pad on the left.)
        seq = m.shape[1]
        idx = seq - 1 - np.argmax(m[:, ::-1] > 0, axis=1)
        return h[np.arange(h.shape[0]), idx, :]

    if strategy == "mean":
        mf = m[:, :, None].astype(np.float64)
        return (h * mf).sum(axis=1) / mf.sum(axis=1)

    raise ValueError(f"unknown pooling strategy: {strategy}")


def extract_all_layers(texts, model_name, strategy="mean", batch_size=16,
                       max_length=512, device=None, dtype=None, cache_dir=None,
                       device_map=None, store_dtype=np.float16, random_init=False,
                       seed=0):
    """
    Returns (acts, info). acts is (n_layers+1, n_texts, dim); index 0 is the
    embedding layer. info reports truncation and non-finite values.

    - Raises if any activation is non-finite. Gemma-2 in fp16 is known to
      overflow; pass dtype=torch.float32 for it on T4 (no bf16 support).
    - Reports the fraction of texts truncated at max_length. High-fertility
      scripts truncate first, which would bias exactly the comparison under
      study, so this is logged per condition.
    - device_map="auto" splits large models (Aya-8B) across both T4s.
    - random_init=True builds the SAME architecture from its config with
      random weights, same tokeniser. This is the baseline every script
      effect must beat: a dry run showed a random-weight model already gives
      a large, BH-significant Gurmukhi/Shahmukhi displacement, simply because
      disjoint token inventories land in different places in any network.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoConfig

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if random_init:
        torch.manual_seed(seed)
        cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        cfg.output_hidden_states = True
        model = AutoModel.from_config(cfg).to(dtype)
        if device_map:
            model = model.to(device)
        model = model.to(device).eval()
        in_dev = device
    else:
        kw = dict(output_hidden_states=True, dtype=dtype, cache_dir=cache_dir)
        if device_map:
            model = AutoModel.from_pretrained(model_name, device_map=device_map, **kw).eval()
            in_dev = next(model.parameters()).device
        else:
            model = AutoModel.from_pretrained(model_name, **kw).to(device).eval()
            in_dev = device

    n_trunc = 0
    per_layer = None
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            full = tok(batch, add_special_tokens=True)["input_ids"]
            n_trunc += sum(1 for ids in full if len(ids) > max_length)

            enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(in_dev)
            out = model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].cpu().numpy()
            if per_layer is None:
                per_layer = [[] for _ in range(len(out.hidden_states))]
            for li, hs in enumerate(out.hidden_states):
                arr = hs.float().cpu().numpy()
                if not np.all(np.isfinite(arr[mask.astype(bool)])):
                    raise FloatingPointError(
                        f"{model_name}: non-finite activations at layer {li}. "
                        "Re-run with dtype=torch.float32.")
                per_layer[li].append(pool_hidden(arr, mask, strategy).astype(np.float32))

    # Release the model before returning. Without this, extracting N conditions
    # leaves N copies on the GPU and the 5th condition of a 1.5B model OOMs on a
    # 16GB T4 (observed on Kaggle).
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    acts = np.stack([np.concatenate(ch, axis=0) for ch in per_layer]).astype(store_dtype)
    info = {"n": len(texts), "truncated": n_trunc,
            "truncated_frac": n_trunc / max(len(texts), 1),
            "max_length": max_length, "n_layers": int(acts.shape[0]),
            "dim": int(acts.shape[2])}
    return acts, info


def save_activations(path, acts, meta=None):
    payload = {"acts": acts}
    if meta:
        payload["meta"] = np.array(str(meta), dtype=object)
    np.savez_compressed(path, **payload)


def load_activations(path):
    z = np.load(path, allow_pickle=True)
    return z["acts"]
