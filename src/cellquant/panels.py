"""Optional real-LLaMA layer diagnostics, not a completed CellQuant/GPTQ method.

Weights rows and token count may be sampled; the feature/group axis is NEVER
subsampled. down_proj is output-side global R1 and is not misrepresented as an
input-side Hadamard experiment. CPU reference analyses preserve all groups.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .runtime import atomic_json, config_hash, environment_report, utc_now


@dataclass
class ProbeConfig:
    model_id: str = "meta-llama/Llama-2-7b-hf"
    revision: str = "main"
    dataset_revision: str = "main"
    layers: tuple[int, ...] = (0,)
    projections: tuple[str, ...] = ("q_proj", "up_proj", "down_proj")
    sequence_length: int = 256
    sequences: int = 2
    capture_tokens: int = 64
    weight_rows: int = 8
    group_size: int = -1
    random_pairs: int = 8
    bits: int = 4
    seed: int = 20261115
    expected_model_gib: float = 12.6
    local_text_file: str | None = None
    save_panels: bool = True


def dequantize_last_axis(x, bits=4, group_size=-1):
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    g = d if group_size == -1 else int(group_size)
    if g <= 0 or d % g:
        raise ValueError("Group size must divide the ENTIRE feature axis")
    qmax, qmin = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
    z = x.reshape(*x.shape[:-1], d // g, g)
    scale = np.max(np.abs(z), axis=-1, keepdims=True) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    codes = np.clip(np.rint(z / scale), qmin, qmax)
    return (scale * codes).reshape(x.shape)


def normalized_hadamard_last_axis(x):
    z = np.array(x, dtype=np.float64, copy=True)
    d = z.shape[-1]
    if d < 1 or d & (d - 1):
        raise ValueError("This diagnostic only supports power-of-two full axes; no silent padding/truncation")
    h = 1
    while h < d:
        v = z.reshape(*z.shape[:-1], -1, 2 * h)
        left, right = v[..., :h].copy(), v[..., h:].copy()
        v[..., :h], v[..., h:] = left + right, left - right
        h *= 2
    return z / math.sqrt(d)


def nmse(error, reference):
    return float(np.sum(np.asarray(error) ** 2) /
                 max(float(np.sum(np.asarray(reference) ** 2)), np.finfo(float).tiny))


def group_diagnostics(x, group_size=-1):
    x = np.asarray(x, dtype=np.float64)
    g = x.shape[-1] if group_size == -1 else group_size
    z = x.reshape(-1, g)
    peaks = np.max(np.abs(z), axis=-1)
    rms = np.sqrt(np.mean(z * z, axis=-1))
    crest = np.divide(peaks, rms, out=np.zeros_like(peaks), where=rms > 0)
    return {"group_count": len(z), "group_size": g,
            "crest_mean": float(np.mean(crest)),
            "crest_p99": float(np.quantile(crest, .99)),
            "absmax_p50": float(np.quantile(peaks, .5)),
            "absmax_p99": float(np.quantile(peaks, .99))}


def stable_group_statistics(rows, *, group_size=-1, pairs=8, seed=20261115):
    rows = np.asarray(rows, dtype=np.float64)
    d = rows.shape[1]
    g = d if group_size == -1 else group_size
    if d % g or d < 2:
        raise ValueError("Invalid full feature/group axis")
    rng = np.random.default_rng(seed)
    unstable = affected = 0
    details = []
    for _ in range(pairs):
        i, j = map(int, rng.choice(d, size=2, replace=False))
        radius = np.hypot(rows[:, i], rows[:, j])
        n_u = n = 0
        for block in sorted({i // g, j // g}):
            ids = np.arange(block * g, (block + 1) * g)
            ids = ids[(ids != i) & (ids != j)]
            maximum = np.max(np.abs(rows[:, ids]), axis=1) if len(ids) else np.zeros(rows.shape[0])
            n_u += int(np.count_nonzero(maximum < radius))
            n += rows.shape[0]
        unstable += n_u
        affected += n
        details.append({"i": i, "j": j, "unstable_groups": n_u, "affected_groups": n})
    return {"status": "MEASURED", "criterion": "M >= hypot(u,v)",
            "scope": "symmetric AbsMax; input-side pair rotations only",
            "pair_selection": "seeded_random", "affected_groups": affected,
            "unstable_groups": unstable,
            "unstable_fraction": unstable / affected if affected else None,
            "pairs": details}


def analyze_input_panel(W, X_rows, *, group_size=-1, bits=4, pairs=8, seed=20261115):
    W, X_rows = np.asarray(W, dtype=np.float64), np.asarray(X_rows, dtype=np.float64)
    if W.shape[1] != X_rows.shape[1]:
        raise ValueError("Weight and activation feature axes differ")
    target = W @ X_rows.T
    Wq = dequantize_last_axis(W, bits, group_size)
    Xq = dequantize_last_axis(X_rows, bits, group_size)
    had_W = normalized_hadamard_last_axis(W)
    had_X = normalized_hadamard_last_axis(X_rows)
    HWq = dequantize_last_axis(had_W, bits, group_size)
    HXq = dequantize_last_axis(had_X, bits, group_size)
    return {
        "placement": "input_side_WQ_transpose_QX",
        "full_feature_axis_retained": True,
        "weight_shape": list(W.shape), "activation_rows_shape": list(X_rows.shape),
        "identity": {"output_nmse": nmse(Wq @ Xq.T - target, target),
                     "weight_nmse": nmse(Wq - W, W),
                     "activation_nmse": nmse(Xq - X_rows, X_rows),
                     "weight_groups": group_diagnostics(W, group_size),
                     "activation_groups": group_diagnostics(X_rows, group_size)},
        "hadamard": {"output_nmse": nmse(HWq @ HXq.T - target, target),
                     "weight_nmse": nmse(HWq - had_W, had_W),
                     "activation_nmse": nmse(HXq - had_X, had_X),
                     "weight_groups": group_diagnostics(had_W, group_size),
                     "activation_groups": group_diagnostics(had_X, group_size)},
        "full_precision_identity_relative_error": float(
            np.linalg.norm(had_W @ had_X.T - target) / max(np.linalg.norm(target), np.finfo(float).tiny)),
        "screening_weight": stable_group_statistics(had_W, group_size=group_size, pairs=pairs, seed=seed),
        "screening_activation": stable_group_statistics(had_X, group_size=group_size, pairs=pairs, seed=seed),
        "claim": "Real local RTN diagnostics only; no learned global R1, GPTQ, KV4, or model PPL",
    }


def _get_hf_token():
    value = os.environ.get("HF_TOKEN")
    if value:
        return value
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".panel-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_real_layer_probe(config: ProbeConfig, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "1.0", "created_at_utc": utc_now(),
              "status": "BLOCKED", "evidence_kind": "REAL_MODEL_LAYER_PROBE",
              "config": asdict(config), "config_sha256": config_hash(asdict(config)),
              "G1_status": "INSUFFICIENT_EVIDENCE", "panels": [],
              "limits": ["No full candidate solver or global-R1 optimization",
                         "No GPTQ or end-to-end PPL measurement",
                         "down_proj output-side R1 is captured but not optimized",
                         "A smoke probe cannot satisfy G1 or paper criteria"]}
    def finish():
        atomic_json(output / "probe_result.json", report)
        return report
    missing = [p for p in ["torch", "transformers", "accelerate"] if importlib.util.find_spec(p) is None]
    if missing:
        report["blocked_reason"] = "Optional packages missing: " + ", ".join(missing)
        return finish()
    import torch
    if not torch.cuda.is_available():
        report["blocked_reason"] = "CUDA is required for this real 7B/8B probe; no automatic low-bit/CPU substitute"
        return finish()
    free, total = torch.cuda.mem_get_info()
    if free / 2**30 < config.expected_model_gib + 1.5:
        report["blocked_reason"] = "Free GPU memory is below configured BF16/FP16 model + 1.5 GiB probe budget"
        report["free_gpu_gib"] = free / 2**30
        return finish()
    if config.capture_tokens > config.sequence_length * config.sequences:
        raise ValueError("capture_tokens exceeds available tokens")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    token = _get_hf_token()
    model = tokenizer = None
    handles = []
    try:
        torch.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, revision=config.revision, token=token,
            torch_dtype=dtype, device_map={"": "cuda:0"},
            low_cpu_mem_usage=True, attn_implementation="eager", trust_remote_code=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if getattr(model.config, "model_type", None) != "llama":
            raise ValueError("This tested interface targets HF Llama; port other architectures explicitly")
        if config.local_text_file:
            text = Path(config.local_text_file).read_text(encoding="utf-8")
            dataset_identity = {"source": "local_text", "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        else:
            if importlib.util.find_spec("datasets") is None:
                raise ImportError("Install datasets or provide local_text_file")
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
                              revision=config.dataset_revision)
            text = "\n\n".join(ds["text"])
            dataset_identity = {"source": "Salesforce/wikitext:wikitext-2-raw-v1:train",
                                "requested_revision": config.dataset_revision,
                                "dataset_fingerprint": ds._fingerprint}
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        needed = config.sequence_length * config.sequences
        if ids.shape[1] < needed:
            raise ValueError("Insufficient calibration text for requested sequences")
        ids = ids[:, :needed].contiguous()
        dataset_identity["token_ids_sha256"] = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        report.update({"model_revision_resolved": getattr(model.config, "_commit_hash", None),
                       "dataset": dataset_identity, "dtype": str(dtype),
                       "environment": environment_report()})
        captured, weights = {}, {}
        for layer_id in config.layers:
            layer = model.model.layers[layer_id]
            for projection in config.projections:
                module = getattr(layer.self_attn if projection == "q_proj" else layer.mlp, projection)
                name = f"layer_{layer_id}.{projection}"
                captured[name] = []
                selected = np.linspace(0, module.weight.shape[0] - 1,
                                       min(config.weight_rows, module.weight.shape[0]), dtype=int)
                row_ids = torch.tensor(selected, device=module.weight.device, dtype=torch.long)
                weights[name] = {"W": module.weight.detach().index_select(0, row_ids).float().cpu().numpy(),
                                 "row_ids": selected, "projection": projection,
                                 "full_weight_shape": list(module.weight.shape)}
                def capture(mod, args, key=name):
                    have = sum(v.shape[0] for v in captured[key])
                    remaining = config.capture_tokens - have
                    if remaining > 0:
                        x = args[0].detach().reshape(-1, args[0].shape[-1])[:remaining]
                        captured[key].append(x.float().cpu().numpy())
                handles.append(module.register_forward_pre_hook(capture))
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for idx in range(config.sequences):
                batch = ids[:, idx * config.sequence_length:(idx + 1) * config.sequence_length].to("cuda:0")
                # The backbone is enough for projection inputs; skip large lm_head logits.
                _ = model.model(input_ids=batch, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        report["probe_peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        for name, pieces in captured.items():
            X = np.concatenate(pieces, axis=0)
            entry = weights[name]
            W = entry["W"]
            if entry["projection"] == "down_proj":
                result = {"placement": "output_side_QB", "status": "CAPTURED_ONLY",
                          "full_feature_axis_retained": True,
                          "weight_shape": list(W.shape), "activation_rows_shape": list(X.shape),
                          "output_side_not_optimized": True,
                          "reason": "Global R1 mixes hidden output rows; sampled rows cannot represent full QB",
                          "activation_groups": group_diagnostics(X, config.group_size)}
            else:
                result = analyze_input_panel(W, X, group_size=config.group_size, bits=config.bits,
                                             pairs=config.random_pairs, seed=config.seed)
            result.update({"name": name, "full_weight_shape": entry["full_weight_shape"],
                           "sampled_weight_row_ids": entry["row_ids"].tolist()})
            if config.save_panels:
                panel_path = output / "panels" / f"{name}.npz"
                _atomic_npz(panel_path, W=W, X_rows=X, weight_row_ids=entry["row_ids"])
                result["panel_path"] = str(panel_path.relative_to(output))
            report["panels"].append(result)
        report["status"] = "MEASURED_DIAGNOSTICS_ONLY"
    except Exception as exc:
        report["status"] = "BLOCKED"
        report["blocked_reason"] = type(exc).__name__ + ": " + str(exc)[:1800]
        # The token is deliberately never put in config/logs.
        if token:
            report["blocked_reason"] = report["blocked_reason"].replace(token, "[REDACTED]")
    finally:
        for handle in handles:
            handle.remove()
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return finish()
