#!/usr/bin/env python3
"""
Pin or download official MACE-MH-1 reference artifacts.

This helper is reference-only. It never vendors source model binaries into Rust
crates and must not be required by normal `cargo test`.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mace-mh1-artifact-v1"
MACE_REPOSITORY = "https://github.com/ACEsuit/mace"
MACE_COMMIT = "4d2da09413ac1407f37cdbb6b81fa28e4c15655e"
FOUNDATIONS_REPOSITORY = "https://github.com/ACEsuit/mace-foundations"
FOUNDATIONS_RELEASE_TAG = "mace_mh_1"
FOUNDATIONS_RELEASE_URL = (
    f"{FOUNDATIONS_REPOSITORY}/releases/tag/{FOUNDATIONS_RELEASE_TAG}"
)
GITHUB_RELEASE_API = (
    f"https://api.github.com/repos/ACEsuit/mace-foundations/releases/tags/"
    f"{FOUNDATIONS_RELEASE_TAG}"
)
HUGGINGFACE_REPOSITORY = "https://huggingface.co/mace-foundations/mace-mh-1"
HUGGINGFACE_API = (
    "https://huggingface.co/api/models/mace-foundations/mace-mh-1?blobs=true"
)

GITHUB_MH1_URL = (
    f"{FOUNDATIONS_REPOSITORY}/releases/download/{FOUNDATIONS_RELEASE_TAG}/"
    "mace-mh-1.model"
)
GITHUB_MH0_URL = (
    f"{FOUNDATIONS_REPOSITORY}/releases/download/{FOUNDATIONS_RELEASE_TAG}/"
    "mace-mh-0.model"
)
HF_MH1_URL = f"{HUGGINGFACE_REPOSITORY}/resolve/main/mace-mh-1.model"

DOCUMENTED_HEADS = [
    "omat_pbe",
    "omol",
    "spice_wB97M",
    "oc20_usemppbe",
    "matpes_r2scan",
]

SOURCE_CANDIDATES: dict[str, dict[str, Any]] = {
    "github-release": {
        "source_kind": "github-release",
        "selected_model": "mh-1",
        "filename": "mace-mh-1.model",
        "url": GITHUB_MH1_URL,
        "sha256": "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47",
        "size_bytes": 59208139,
        "role": "primary_for_upstream_mace_registry",
    },
    "huggingface": {
        "source_kind": "huggingface",
        "selected_model": "mh-1",
        "filename": "mace-mh-1.model",
        "url": HF_MH1_URL,
        "sha256": "ec00a2705854622fbbd898ccfb7701072fcd674709102d009fb919c1b8cc5dde",
        "size_bytes": 59208715,
        "role": "official_model_card_payload_sha_differs_from_github_release",
    },
    "github-release-mh0": {
        "source_kind": "github-release",
        "selected_model": "mh-0",
        "filename": "mace-mh-0.model",
        "url": GITHUB_MH0_URL,
        "sha256": "d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d",
        "size_bytes": 39921174,
        "role": "predecessor_not_target",
    },
}


def checked_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.output.strip()) from exc


def validate_mace_tree(mace_dir: Path) -> str:
    if not mace_dir.exists():
        raise SystemExit(f"missing pinned MACE reference directory: {mace_dir}")
    commit = checked_output(["git", "-C", str(mace_dir), "rev-parse", "HEAD"])
    pin_path = mace_dir.with_name(f"{mace_dir.name}.PINNED_COMMIT")
    if not pin_path.exists():
        raise SystemExit(f"missing pinned commit marker: {pin_path}")
    pinned_commit = pin_path.read_text(encoding="utf-8").strip()
    if commit != pinned_commit:
        raise SystemExit(
            "MACE reference checkout does not match pinned commit marker\n"
            f"checkout: {commit}\n"
            f"marker:   {pinned_commit}"
        )
    if commit != MACE_COMMIT:
        raise SystemExit(
            "MACE reference checkout does not match the MACE-MH-1 pin\n"
            f"expected: {MACE_COMMIT}\n"
            f"actual:   {commit}"
        )
    return commit


def mace_version(mace_dir: Path) -> str:
    version_path = mace_dir / "mace" / "__version__.py"
    tree = ast.parse(version_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__version__"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    return node.value.value
    return "unknown"


def mace_mp_urls(mace_dir: Path) -> tuple[Path, dict[str, str]]:
    source_path = mace_dir / "mace" / "calculators" / "foundations_models.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "mace_mp_urls":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, dict):
                        raise SystemExit("mace_mp_urls is not a dictionary")
                    urls: dict[str, str] = {}
                    for key, url in value.items():
                        if not isinstance(key, str) or not isinstance(url, str):
                            raise SystemExit("mace_mp_urls contains non-string entries")
                        urls[key] = url
                    return source_path, urls
    raise SystemExit(f"could not find mace_mp_urls in {source_path}")


def validate_upstream_mace_mh1_url(mace_dir: Path) -> tuple[Path, str]:
    source_path, urls = mace_mp_urls(mace_dir)
    actual = urls.get("mh-1")
    if actual != GITHUB_MH1_URL:
        raise SystemExit(
            "pinned MACE checkout does not map mh-1 to the expected artifact URL\n"
            f"expected: {GITHUB_MH1_URL}\n"
            f"actual:   {actual}"
        )
    return source_path, actual


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_and_hash(url: str, destination: Path, force: bool) -> tuple[str, int]:
    if destination.exists() and not force:
        return hash_file(destination), destination.stat().st_size

    destination.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temp:
        temp_path = Path(temp.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temp.write(chunk)
                    hasher.update(chunk)
                    size += len(chunk)
            os.replace(temp_path, destination)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    return hasher.hexdigest(), size


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def refreshed_api_snapshot() -> dict[str, Any]:
    github = fetch_json(GITHUB_RELEASE_API)
    huggingface = fetch_json(HUGGINGFACE_API)
    github_assets = [
        {
            "name": asset.get("name"),
            "size": asset.get("size"),
            "browser_download_url": asset.get("browser_download_url"),
            "digest": asset.get("digest"),
            "created_at": asset.get("created_at"),
            "updated_at": asset.get("updated_at"),
        }
        for asset in github.get("assets", [])
    ]
    hf_siblings = [
        {
            "rfilename": sibling.get("rfilename"),
            "size": sibling.get("size"),
            "blobId": sibling.get("blobId"),
            "lfs": sibling.get("lfs"),
        }
        for sibling in huggingface.get("siblings", [])
    ]
    return {
        "github_release": {
            "tag_name": github.get("tag_name"),
            "name": github.get("name"),
            "published_at": github.get("published_at"),
            "target_commitish": github.get("target_commitish"),
            "html_url": github.get("html_url"),
            "assets": github_assets,
        },
        "huggingface": {
            "id": huggingface.get("id"),
            "sha": huggingface.get("sha"),
            "lastModified": huggingface.get("lastModified"),
            "cardData": huggingface.get("cardData"),
            "siblings": hf_siblings,
        },
    }


def verify_selected_artifact(source_key: str, sha256: str | None, size_bytes: int | None) -> None:
    candidate = SOURCE_CANDIDATES[source_key]
    if sha256 is not None and sha256 != candidate["sha256"]:
        raise SystemExit(
            "artifact SHA256 does not match expected value\n"
            f"expected: {candidate['sha256']}\n"
            f"actual:   {sha256}"
        )
    if size_bytes is not None and size_bytes != candidate["size_bytes"]:
        raise SystemExit(
            "artifact size does not match expected value\n"
            f"expected: {candidate['size_bytes']}\n"
            f"actual:   {size_bytes}"
        )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mace-dir", type=Path, default=Path("third_party/mace"))
    parser.add_argument(
        "--source",
        choices=["github-release", "huggingface"],
        default="github-release",
        help=(
            "artifact source to select for optional download. The default "
            "matches the pinned upstream MACE mh-1 registry URL."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/reference-artifacts/mace"),
        help="local cache for downloaded model binaries; ignored by git",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fixtures/mace/current/mace_mh1_artifact_manifest.json"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--refresh-api",
        action="store_true",
        help="query GitHub and Hugging Face APIs and embed a compact snapshot",
    )
    args = parser.parse_args()

    commit = validate_mace_tree(args.mace_dir)
    source_path, upstream_url = validate_upstream_mace_mh1_url(args.mace_dir)
    selected = SOURCE_CANDIDATES[args.source]
    artifact_path = args.cache_dir / "mace-mh1" / args.source / selected["filename"]
    sha256 = None
    size_bytes = None
    if args.download:
        sha256, size_bytes = download_and_hash(selected["url"], artifact_path, args.force)
    elif artifact_path.exists():
        sha256 = hash_file(artifact_path)
        size_bytes = artifact_path.stat().st_size
    verify_selected_artifact(args.source, sha256, size_bytes)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selected_model": "mh-1",
        "family": "MACE-MH-1",
        "purpose": "official MACE-MH-1 artifact pin for reference-only inspection and scoped Rust export",
        "reference": {
            "name": "mace",
            "repository": MACE_REPOSITORY,
            "commit": commit,
            "package_version": mace_version(args.mace_dir),
        },
        "upstream_mace_registry": {
            "source_file": str(source_path),
            "url_key": "mh-1",
            "url": upstream_url,
        },
        "official_sources": {
            key: dict(value) for key, value in sorted(SOURCE_CANDIDATES.items())
        },
        "selected_artifact": {
            "source_key": args.source,
            "source_kind": selected["source_kind"],
            "filename": selected["filename"],
            "url": selected["url"],
            "binary_tracked_by_git": False,
            "cache_path": str(artifact_path) if sha256 is not None else None,
            "downloaded": bool(args.download),
            "sha256": sha256 or selected["sha256"],
            "size_bytes": size_bytes or selected["size_bytes"],
        },
        "documented_heads": DOCUMENTED_HEADS,
        "requested_heads": DOCUMENTED_HEADS,
        "observed_heads": None,
        "observed_heads_note": (
            "Populate from tools/reference/inspect_mace_mh1_artifact.py; "
            "the public model card/release notes and the serialized model may differ."
        ),
        "size_variant_scope": {
            "status": "not_applicable_for_mace_mh1",
            "official_model_keys_found": ["mh-0", "mh-1"],
            "implementation_target": "mh-1",
            "note": (
                "MACE-MH-1 has no size axis in the checked official sources. "
                "Do not introduce synthetic small/medium/large or s/m/l labels "
                "for this family."
            ),
        },
        "scope": {
            "normal_cargo_test_dependency": False,
            "rust_runtime_dependency": False,
            "status": (
                "downloaded_reference_binary" if args.download else "pinned_reference_binary"
            ),
            "conversion_note": (
                "The binary itself is not committed. Future `.e3rs` conversion "
                "artifacts must serialize only explicitly supported fixed tensors "
                "and plans."
            ),
        },
        "source_notes": [
            (
                "The Hugging Face and GitHub release mace-mh-1.model payloads "
                "currently have different SHA256 values and byte sizes; treat "
                "them as separate source candidates, not mirrors."
            ),
            (
                "The primary implementation target is the GitHub release asset "
                "because the pinned upstream MACE foundations registry maps "
                "model key mh-1 to that URL."
            ),
        ],
        "python": {
            "python_version": platform.python_version(),
            "executable": Path(sys.executable).name,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.refresh_api:
        manifest["api_snapshot"] = refreshed_api_snapshot()
    write_json(args.out, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
