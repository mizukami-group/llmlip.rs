#!/usr/bin/env python3
"""Unified front-end for converting upstream MLIP checkpoints into `.e3rs`.

This wraps the per-model exporters under ``tools/reference/`` and adds a
consistent UX: license display + explicit acceptance, checkpoint SHA256
verification, and output validation. The heavy conversion still runs in the
upstream exporter (which needs PyTorch and the upstream model library).

Examples
--------
    python converters/convert.py list
    python converters/convert.py info uma-s-1p2
    python converters/convert.py verify uma-s-1p2 /path/to/uma-s-1p2.pt
    python converters/convert.py convert uma-s-1p2 \\
        --checkpoint /path/to/uma-s-1p2.pt --out out/uma-s-1p2.e3rs --accept-license

The resulting `.e3rs` directory is then usable either at runtime
(``llmlip predict --model out/uma-s-1p2.e3rs ...``) or baked into a binary at
compile time (``LLMLIP_EMBED_MODEL=$PWD/out/uma-s-1p2.e3rs cargo build -p llmlip
--features embed-model``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REGISTRY_PATH = HERE / "models.json"


def load_registry() -> dict:
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def get_model(registry: dict, key: str) -> dict:
    models = registry["models"]
    if key not in models:
        available = ", ".join(sorted(models))
        raise SystemExit(f"unknown model '{key}'. Available: {available}")
    return models[key]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_list(registry: dict, _args: argparse.Namespace) -> int:
    print(f"{'model':<16} {'architecture':<16} {'restricted':<11} license")
    print("-" * 72)
    for key, model in sorted(registry["models"].items()):
        restricted = "yes" if model.get("restricted") else "no"
        lic = model["license"]["name"]
        print(f"{key:<16} {model['architecture']:<16} {restricted:<11} {lic}")
    return 0


def cmd_info(registry: dict, args: argparse.Namespace) -> int:
    model = get_model(registry, args.model)
    print(json.dumps(model, indent=2))
    return 0


def print_license(model: dict) -> None:
    lic = model["license"]
    src = model["source"]
    print("=" * 72)
    print(f"Model:   {model['title']}")
    print(f"Source:  {src.get('url', src)}")
    if src.get("gated"):
        print("         (GATED: you must request/accept access at the source)")
    print(f"License: {lic['name']}")
    print(f"         {lic.get('url', '')}")
    print(f"         project distribution policy: {model['redistribution']['policy']}")
    print("=" * 72)


def verify_checkpoint(model: dict, checkpoint: Path, strict: bool) -> None:
    expected = model.get("checkpoint", {}).get("sha256")
    expected_size = model.get("checkpoint", {}).get("size_bytes")
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if expected_size is not None and checkpoint.stat().st_size != expected_size:
        msg = (
            f"size mismatch: got {checkpoint.stat().st_size}, "
            f"expected {expected_size}"
        )
        if strict:
            raise SystemExit(msg)
        print(f"WARNING: {msg}")
    if expected is None:
        print("note: no pinned SHA256 in registry; skipping hash check.")
        return
    actual = sha256_file(checkpoint)
    if actual != expected:
        msg = f"SHA256 mismatch:\n  got      {actual}\n  expected {expected}"
        if strict:
            raise SystemExit(msg)
        print(f"WARNING: {msg}")
    else:
        print(f"SHA256 OK: {actual}")


def cmd_verify(registry: dict, args: argparse.Namespace) -> int:
    model = get_model(registry, args.model)
    verify_checkpoint(model, Path(args.checkpoint), strict=not args.warn_only)
    return 0


def find_llmlip_binary() -> Path | None:
    """Locate a built `llmlip` CLI for full model validation, if available."""
    import os
    import shutil

    env = os.environ.get("LLMLIP_BINARY")
    if env:
        candidate = Path(env)
        if candidate.exists():
            return candidate
    for profile in ("release", "debug"):
        candidate = REPO_ROOT / "target" / profile / "llmlip"
        if candidate.exists():
            return candidate
    on_path = shutil.which("llmlip")
    return Path(on_path) if on_path else None


def parse_manifest_minimal(manifest: Path) -> dict:
    """Parse the fields we validate from a generated manifest.toml.

    Uses tomllib when available (Python >= 3.11); otherwise falls back to a
    line-based scan that is sufficient for the machine-generated manifests the
    exporters write (no inline tables across lines, no multi-line strings).
    """
    try:
        import tomllib

        with manifest.open("rb") as handle:
            data = tomllib.load(handle)
        tensors = data.get("tensors", [])
        state_files = [
            section["state"]
            for section in data.values()
            if isinstance(section, dict) and isinstance(section.get("state"), str)
        ]
        return {
            "architecture": data.get("architecture"),
            "tensor_paths": [entry.get("path") for entry in tensors if entry.get("path")],
            "state_files": state_files,
        }
    except ModuleNotFoundError:
        pass

    import re

    architecture = None
    tensor_paths: list[str] = []
    state_files: list[str] = []
    in_tensors = False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[["):
            in_tensors = stripped == "[[tensors]]"
            continue
        if stripped.startswith("["):
            in_tensors = False
            continue
        match = re.match(r'^architecture\s*=\s*"([^"]+)"', stripped)
        if match and architecture is None:
            architecture = match.group(1)
            continue
        match = re.match(r'^state\s*=\s*"([^"]+)"', stripped)
        if match:
            state_files.append(match.group(1))
            continue
        if in_tensors:
            match = re.match(r'^path\s*=\s*"([^"]+)"', stripped)
            if match:
                tensor_paths.append(match.group(1))
    return {
        "architecture": architecture,
        "tensor_paths": tensor_paths,
        "state_files": state_files,
    }


def validate_output(out_dir: Path) -> None:
    manifest = out_dir / "manifest.toml"
    if not manifest.exists():
        raise SystemExit(
            f"conversion produced no manifest.toml at {out_dir}; conversion failed"
        )

    binary = find_llmlip_binary()
    if binary is not None:
        cmd = [str(binary), "validate-model", "--model", str(out_dir)]
        print("validating:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"`llmlip validate-model` failed with exit code {result.returncode}"
            )
        print(f"OK: wrote {out_dir} (validated with {binary})")
        return

    parsed = parse_manifest_minimal(manifest)
    problems: list[str] = []
    if not parsed["architecture"]:
        problems.append("manifest.toml has no `architecture` field")
    for state_file in parsed["state_files"]:
        if not (out_dir / state_file).exists():
            problems.append(f"state file referenced by manifest is missing: {state_file}")
    missing_tensors = [
        path for path in parsed["tensor_paths"] if not (out_dir / path).exists()
    ]
    if missing_tensors:
        preview = ", ".join(missing_tensors[:5])
        problems.append(
            f"{len(missing_tensors)} tensor payload(s) referenced by manifest are "
            f"missing (first few: {preview})"
        )
    if problems:
        raise SystemExit(
            "conversion output failed validation:\n  - " + "\n  - ".join(problems)
        )
    print(
        f"OK: wrote {out_dir} (architecture={parsed['architecture']}, "
        f"{len(parsed['tensor_paths'])} tensor file(s) present; build the `llmlip` "
        "CLI for full validation via `llmlip validate-model`)"
    )


def cmd_convert(registry: dict, args: argparse.Namespace) -> int:
    model = get_model(registry, args.model)
    print_license(model)
    if not args.accept_license:
        raise SystemExit(
            "refusing to convert without --accept-license. Review the license "
            "above; by passing --accept-license you confirm you have the right "
            "to use this checkpoint."
        )
    checkpoint = Path(args.checkpoint)
    if args.dry_run and not checkpoint.exists():
        print(f"note: checkpoint {checkpoint} not found; skipping verification (dry run)")
    else:
        verify_checkpoint(model, checkpoint, strict=not args.skip_verify)

    exporter = model["exporter"]
    script = REPO_ROOT / exporter["script"]
    if not script.exists():
        raise SystemExit(f"exporter not found: {script}")
    if exporter.get("requires"):
        print(f"note: this exporter requires {exporter['requires']}")
    out_dir = Path(args.out)

    # Manifest-driven exporters (PolarMACE) take a pinned artifact-manifest
    # JSON instead of the raw checkpoint. Instantiate the committed template
    # with the user's checkpoint path; the exporter re-verifies the pinned
    # SHA256 against that file before unpickling.
    checkpoint_value = checkpoint
    if exporter.get("manifest_template"):
        template_path = REPO_ROOT / exporter["manifest_template"]
        if not template_path.exists():
            raise SystemExit(f"artifact manifest template not found: {template_path}")
        if args.dry_run:
            print(
                f"(dry run) would instantiate {template_path} with "
                f"artifact_path={checkpoint} and pass it via "
                f"{exporter['checkpoint_arg']}"
            )
            checkpoint_value = Path(f"<generated {args.model} artifact manifest>")
        else:
            with template_path.open(encoding="utf-8") as handle:
                manifest_data = json.load(handle)
            manifest_data["artifact_path"] = str(checkpoint.resolve())
            # The runtime metadata parser rejects unknown fields; some committed
            # templates carry extra bookkeeping keys (model, state_export,
            # reference_environment) that must not flow into the export.
            allowed = {
                "artifact_path", "available_models", "byte_size",
                "cache_replaced", "downloaded_this_run", "expected_sha256",
                "family", "mace_commit", "mace_package_version",
                "model_file_tracked_in_git", "normal_cargo_test_dependency",
                "schema_version", "selected_model", "sha256", "source_file",
                "source_url",
            }
            manifest_data = {k: v for k, v in manifest_data.items() if k in allowed}
            # The exporters require the manifest to live inside the repo, so
            # write it next to the conversion output (also nice provenance).
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            generated_path = out_dir.parent / f"{args.model}.artifact-manifest.json"
            with generated_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest_data, handle, indent=2, sort_keys=True)
                handle.write("\n")
            checkpoint_value = generated_path
            print(f"generated artifact manifest: {checkpoint_value}")

    cmd = [
        sys.executable,
        str(script),
        exporter["checkpoint_arg"],
        str(checkpoint_value),
        exporter["out_arg"],
        str(out_dir),
        *exporter.get("extra_args", []),
        *args.passthrough,
    ]
    print("running:", " ".join(cmd))
    if args.dry_run:
        print("(dry run; not executing)")
        return 0
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"exporter failed with exit code {result.returncode}")
    validate_output(out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list known models")

    info = sub.add_parser("info", help="print registry entry for a model")
    info.add_argument("model")

    verify = sub.add_parser("verify", help="verify a checkpoint's SHA256")
    verify.add_argument("model")
    verify.add_argument("checkpoint")
    verify.add_argument(
        "--warn-only",
        action="store_true",
        help="print SHA256/size mismatches as warnings instead of failing",
    )

    convert = sub.add_parser("convert", help="convert a checkpoint to .e3rs")
    convert.add_argument("model")
    convert.add_argument("--checkpoint", required=True)
    convert.add_argument("--out", required=True)
    convert.add_argument("--accept-license", action="store_true")
    convert.add_argument("--skip-verify", action="store_true", help="do not fail on SHA256/size mismatch")
    convert.add_argument("--dry-run", action="store_true", help="print the exporter command without running it")
    convert.add_argument("passthrough", nargs="*", help="extra args forwarded to the exporter")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    registry = load_registry()
    handlers = {
        "list": cmd_list,
        "info": cmd_info,
        "verify": cmd_verify,
        "convert": cmd_convert,
    }
    return handlers[args.command](registry, args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
