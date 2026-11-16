"""Download pinned model files and verify weights without allocating an LLM.

Run after prepare_experiment.py, then back up models/datasets/runs to Drive.
This operation supplies no G1/G2 quality or GPU compatibility evidence.
"""
from pathlib import Path
import argparse
import os
import uuid

from cellquant.data import immutable_revision
from cellquant.runtime import atomic_json, read_json, sha256_file, huggingface_cache_paths


def cache_model(config_path, output):
    config = read_json(config_path)
    if not all(immutable_revision(config.get(k)) for k in ("model_revision", "tokenizer_revision")):
        raise ValueError("Prepare immutable model/tokenizer revisions first")
    source_path = config.get("model_source_manifest_path")
    if not source_path or sha256_file(source_path) != config.get("model_source_manifest_sha256"):
        raise ValueError("Model source manifest checksum mismatch")
    model_id = config.get("model_id", config.get("formal_pilot", {}).get("model_id"))
    source = read_json(source_path)
    if source.get("model_id") != model_id or source.get("revision") != config["model_revision"]:
        raise ValueError("Model source identity mismatch")
    from huggingface_hub import snapshot_download
    from cellquant.experiment import _verify_weight_files
    output = Path(output).resolve() / uuid.uuid4().hex
    output.mkdir(parents=True)
    cache = str(huggingface_cache_paths(config.get("runtime"))[0])
    # Config/tokenizer files are small; weight shards use the existing checked
    # download path and are validated against HF LFS size/SHA-256 metadata.
    for revision in set((config["model_revision"], config["tokenizer_revision"])):
        snapshot_download(model_id, revision=revision, cache_dir=cache,
                          token=os.environ.get("HF_TOKEN") or None,
                          allow_patterns=["config.json", "generation_config.json", "tokenizer*",
                                          "special_tokens_map.json", "added_tokens.json"],
                          local_files_only=config.get("runtime", {}).get("local_cache_only", False))
    _verify_weight_files(config, output)
    record = {"status": "WEIGHTS_VERIFIED", "source_type": "not_run", "model_id": model_id,
              "model_revision": config["model_revision"], "cache_dir": cache,
              "config_sha256": sha256_file(config_path), "verification_dir": str(output),
              "note": "Download and file verification only; no LLM loaded or scientific gate evaluated"}
    atomic_json(output / "cache_receipt.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="runs/cache_checks")
    args = parser.parse_args()
    print(cache_model(args.config, args.output))
