#!/usr/bin/env python3
"""Pin MACE-POLAR-1 reference artifacts.

This script is reference-only. It downloads or records upstream PolarMACE
artifacts under ignored output paths and writes a small JSON manifest that Rust
tests may consume later as committed metadata. It must not become a Rust
runtime dependency.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


PINNED_MACE_COMMIT = "4d2da09413ac1407f37cdbb6b81fa28e4c15655e"
DEFAULT_MODEL = "polar-1-s"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def validate_mace_tree(mace_dir: Path) -> str:
    pinned_path = mace_dir.parent / "mace.PINNED_COMMIT"
    pinned = pinned_path.read_text(encoding="utf-8").strip()
    if pinned != PINNED_MACE_COMMIT:
        raise RuntimeError(
            f"{pinned_path} must contain {PINNED_MACE_COMMIT}, got {pinned}"
        )
    head = run_git(["rev-parse", "HEAD"], mace_dir)
    if head != PINNED_MACE_COMMIT:
        raise RuntimeError(f"{mace_dir} HEAD must be {PINNED_MACE_COMMIT}, got {head}")
    return head


def polar_urls(mace_dir: Path) -> dict[str, str]:
    source = mace_dir / "mace" / "calculators" / "foundations_models.py"
    text = source.read_text(encoding="utf-8")
    marker = "polar_model_urls = "
    start = text.index(marker) + len(marker)
    brace_start = text.index("{", start)
    depth = 0
    for index in range(brace_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                block = text[brace_start : index + 1]
                parsed = ast.literal_eval(block)
                return {str(key): str(value) for key, value in parsed.items()}
    raise RuntimeError(f"could not parse polar_model_urls from {source}")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_filename(url: str) -> str:
    return "".join(
        char for char in os.path.basename(url) if char.isalnum() or char in "._-"
    )


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "e3nn-core-rs-reference"})
    part = destination.with_suffix(destination.suffix + ".part")
    byte_count = 0
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError(f"model download failed with HTTP status {status} from {url}")
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type or "application/json" in content_type:
            raise RuntimeError(f"model download returned {content_type!r} from {url}")
        with part.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                byte_count += len(chunk)
                handle.write(chunk)
    if byte_count < 1024:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"model download from {url} was unexpectedly small: {byte_count} bytes")
    os.replace(part, destination)


def mace_version(mace_dir: Path) -> str | None:
    for candidate in [
        mace_dir / "mace" / "__version__.py",
        mace_dir / "mace" / "__init__.py",
    ]:
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            if line.startswith("__version__"):
                return str(ast.literal_eval(line.split("=", 1)[1].strip()))
    raise RuntimeError(f"could not determine MACE package version under {mace_dir}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    mace_dir = (root / args.mace_dir).resolve()
    commit = validate_mace_tree(mace_dir)
    urls = polar_urls(mace_dir)
    if args.model not in urls:
        raise RuntimeError(
            f"unknown PolarMACE model {args.model!r}; available: {sorted(urls)}"
        )
    url = urls[args.model]
    artifact_path = (root / args.artifacts_dir / artifact_filename(url)).resolve()

    downloaded = False
    cache_replaced = False
    if args.download and not artifact_path.exists():
        download(url, artifact_path)
        downloaded = True
    if artifact_path.exists() and args.expected_sha256:
        actual_sha256 = hash_file(artifact_path)
        if actual_sha256 != args.expected_sha256:
            if not args.download:
                raise RuntimeError(
                    f"{artifact_path} SHA256 mismatch: expected {args.expected_sha256}, got {actual_sha256}"
                )
            stale = artifact_path.with_suffix(artifact_path.suffix + ".sha256-mismatch")
            os.replace(artifact_path, stale)
            download(url, artifact_path)
            cache_replaced = True
            downloaded = True
            actual_sha256 = hash_file(artifact_path)
            if actual_sha256 != args.expected_sha256:
                raise RuntimeError(
                    f"{artifact_path} SHA256 mismatch after fresh download: expected {args.expected_sha256}, got {actual_sha256}"
                )

    artifact: dict[str, Any] = {
        "available_models": sorted(urls),
        "family": "PolarMACE",
        "mace_commit": commit,
        "mace_package_version": mace_version(mace_dir),
        "model_file_tracked_in_git": False,
        "normal_cargo_test_dependency": False,
        "schema_version": "polarmace-artifact-manifest-v1",
        "selected_model": args.model,
        "source_file": "third_party/mace/mace/calculators/foundations_models.py",
        "source_url": url,
    }
    if artifact_path.exists():
        artifact.update(
            {
                "artifact_path": str(artifact_path.relative_to(root)),
                "byte_size": artifact_path.stat().st_size,
                "cache_replaced": cache_replaced,
                "downloaded_this_run": downloaded,
                "expected_sha256": args.expected_sha256,
                "sha256": hash_file(artifact_path),
            }
        )
    else:
        artifact.update(
            {
                "artifact_path": str(artifact_path.relative_to(root)),
                "byte_size": None,
                "cache_replaced": cache_replaced,
                "downloaded_this_run": False,
                "expected_sha256": args.expected_sha256,
                "sha256": None,
                "status": "not_downloaded",
            }
        )
    return artifact


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mace-dir", default="third_party/mace", type=Path)
    parser.add_argument(
        "--artifacts-dir",
        default="outputs/reference-artifacts/mace",
        type=Path,
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args)
    write_json(repo_root() / args.out, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
