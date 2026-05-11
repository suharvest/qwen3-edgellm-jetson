#!/usr/bin/env python3
"""Stage Qwen3 artifacts for Hugging Face upload.

The manifest records the canonical relative layout consumed by Jetson Voice.
This tool copies files from a built engine tree into that layout and writes a
checksum inventory so uploads are reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def required_files(manifest: dict, set_name: str) -> list[str]:
    sets = manifest.get("artifact_sets", {})
    if set_name not in sets:
        raise KeyError(f"artifact set {set_name!r} not found; available: {', '.join(sorted(sets))}")
    return list(sets[set_name].get("required_files", []))


def parse_maps(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--map expects RELATIVE_PATH=SOURCE_PATH, got {value!r}")
        rel, src = value.split("=", 1)
        result[rel] = Path(src).expanduser()
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(rel: str, source_root: Path | None, explicit_maps: dict[str, Path]) -> Path | None:
    if rel in explicit_maps:
        return explicit_maps[rel]
    if source_root is None:
        return None
    return source_root / rel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="deploy/artifacts/qwen3_manifest.json")
    parser.add_argument("--set", dest="set_name", required=True)
    parser.add_argument("--source-root", help="Tree that already uses manifest-relative paths.")
    parser.add_argument("--out", required=True, help="Staging directory to upload with hf upload.")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="Override one file source as RELATIVE_PATH=SOURCE_PATH. Can be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    rel_files = required_files(manifest, args.set_name)
    source_root = Path(args.source_root).expanduser() if args.source_root else None
    explicit_maps = parse_maps(args.map)
    out_root = Path(args.out).expanduser()

    staged: list[dict[str, object]] = []
    missing: list[tuple[str, str]] = []

    for rel in rel_files:
        src = resolve_source(rel, source_root, explicit_maps)
        if src is None:
            missing.append((rel, "no source mapping"))
            continue
        if not src.exists():
            missing.append((rel, str(src)))
            continue
        dst = out_root / rel
        size = src.stat().st_size
        digest = sha256_file(src) if not args.dry_run else ""
        staged.append({"path": rel, "source": str(src), "size": size, "sha256": digest})
        print(f"{'would copy' if args.dry_run else 'copy'} {src} -> {dst} ({size} bytes)")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    if missing:
        print("\nMissing required files:")
        for rel, reason in missing:
            print(f"  {rel}: {reason}")
        return 2

    inventory = {
        "schema_version": 1,
        "artifact_set": args.set_name,
        "manifest": str(manifest_path),
        "files": staged,
    }
    inventory_path = out_root / "checksums" / f"{args.set_name}.json"
    print(f"{'would write' if args.dry_run else 'write'} {inventory_path}")
    if not args.dry_run:
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
