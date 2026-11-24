"""Read-only orphan sidecar scanner and non-destructive quarantine mover."""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".bin", ".pth")
SIDECAR_SUFFIXES = (
    ".preview.jpeg",
    ".preview.webp",
    ".preview.png",
    ".preview.jpg",
    ".civitai.info",
    ".info",
    ".json",
)
IGNORED_FILES = {
    "rename_history.json",
    "not_found_hashes.json",
    "config.local.json",
    "config.json",
}


def _sidecar_base(filename: str) -> str | None:
    lower = filename.lower()
    if lower in IGNORED_FILES:
        return None
    for suffix in SIDECAR_SUFFIXES:
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    return None


def _model_stem(filename: str) -> str | None:
    lower = filename.lower()
    for ext in MODEL_EXTS:
        if lower.endswith(ext):
            return filename[: -len(ext)]
    return None


def scan_orphans(directory: Path, recursive: bool = True) -> list[Path]:
    """Return sidecar files in directory whose stem has no matching model binary."""
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在：{directory}")

    orphans: list[Path] = []
    walker = (
        os.walk(directory)
        if recursive
        else [(str(directory), [], [p.name for p in directory.iterdir() if p.is_file()])]
    )
    for root_str, dirnames, filenames in walker:
        if recursive:
            dirnames[:] = sorted(
                d for d in dirnames
                if not d.startswith(".") and d not in ("__pycache__", "quarantine")
            )
        cur_dir = Path(root_str)
        model_stems: set[str] = set()
        for name in filenames:
            stem = _model_stem(name)
            if stem:
                model_stems.add(stem.casefold())

        for name in sorted(filenames, key=str.casefold):
            if name.startswith("."):
                continue
            base = _sidecar_base(name)
            if base is None:
                continue
            if base.casefold() not in model_stems:
                orphans.append(cur_dir / name)
    return orphans


def quarantine_orphans(
    orphan_paths: list[Path],
    base_dir: Path,
    quarantine_root: Path,
) -> dict[str, Any]:
    """Move orphan sidecars into a timestamped quarantine directory (never delete)."""
    if not orphan_paths:
        return {"moved": 0, "quarantine_dir": "", "errors": []}

    base_resolved = base_dir.resolve()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    batch_dir = (quarantine_root / stamp).resolve()
    moved = 0
    errors: list[str] = []

    for src in orphan_paths:
        try:
            src_resolved = src.resolve()
            if not src_resolved.is_file():
                continue
            try:
                rel = src_resolved.relative_to(base_resolved)
            except ValueError:
                rel = Path(src_resolved.name)
            dst = batch_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_resolved), str(dst))
            moved += 1
        except Exception as exc:
            errors.append(f"{src}: {exc}")

    return {
        "moved": moved,
        "quarantine_dir": str(batch_dir),
        "errors": errors,
    }
