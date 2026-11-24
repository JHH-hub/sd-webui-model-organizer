"""Safe preview image normalization preserving true image format or clean PNG transcoding."""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".bin", ".pth")
IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg")


def _model_stem(filename: str) -> str | None:
    lower = filename.lower()
    for ext in MODEL_EXTS:
        if lower.endswith(ext):
            return filename[: -len(ext)]
    return None


def _has_existing_preview(dir_files_lower: set[str], stem_lower: str) -> bool:
    for ext in IMAGE_EXTS:
        if f"{stem_lower}.preview{ext}" in dir_files_lower:
            return True
    return False


def _find_exact_stem_image(cur_dir: Path, files_by_lower: dict[str, Path], stem: str) -> Path | None:
    stem_lower = stem.casefold()
    for ext in IMAGE_EXTS:
        hit = files_by_lower.get(f"{stem_lower}{ext}")
        if hit is not None and hit.is_file():
            return hit
    return None


def _convert_to_real_png(src_path: Path, dst_path: Path) -> None:
    from PIL import Image

    tmp_path = dst_path.with_name(f".__mo_png_tmp_{uuid.uuid4().hex}.png")
    try:
        with Image.open(src_path) as im:
            has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
            converted = im.convert("RGBA" if has_alpha else "RGB")
            converted.save(tmp_path, format="PNG")
        tmp_path.replace(dst_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def collect_preview_normalizations(
    directory: Path,
    recursive: bool = True,
    mode: str = "keep_ext",
) -> tuple[list[dict[str, Any]], int]:
    """Scan model folders for <stem>.<ext> images missing the .preview suffix."""
    if mode not in ("keep_ext", "convert_png"):
        raise ValueError(f"Unsupported preview mode: {mode}")
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在：{directory}")

    tasks: list[dict[str, Any]] = []
    already_ok = 0

    walker = os.walk(directory) if recursive else [(str(directory), [], [p.name for p in directory.iterdir() if p.is_file()])]
    for root_str, dirnames, filenames in walker:
        if recursive:
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
        cur_dir = Path(root_str)
        files_by_lower = {name.casefold(): (cur_dir / name) for name in filenames}
        files_lower_set = set(files_by_lower.keys())

        stems: list[str] = []
        seen_stems: set[str] = set()
        for name in sorted(filenames, key=str.casefold):
            stem = _model_stem(name)
            if not stem:
                continue
            key = stem.casefold()
            if key not in seen_stems:
                seen_stems.add(key)
                stems.append(stem)

        for stem in stems:
            stem_lower = stem.casefold()
            if _has_existing_preview(files_lower_set, stem_lower):
                already_ok += 1
                continue
            candidate = _find_exact_stem_image(cur_dir, files_by_lower, stem)
            if candidate is None:
                continue
            ext = candidate.suffix.lower()
            target_ext = ".png" if mode == "convert_png" else ext
            target = cur_dir / f"{stem}.preview{target_ext}"
            if target.exists():
                already_ok += 1
                continue
            tasks.append({
                "src": candidate,
                "dst": target,
                "mode": mode,
                "convert": mode == "convert_png" and ext != ".png",
            })

    return tasks, already_ok


def apply_preview_normalizations(
    directory: Path,
    recursive: bool = True,
    mode: str = "keep_ext",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Normalize <stem>.<ext> previews to <stem>.preview.<ext> or lossless PNG."""
    tasks, already_ok = collect_preview_normalizations(directory, recursive=recursive, mode=mode)
    processed = 0
    errors: list[str] = []

    for item in tasks:
        src: Path = item["src"]
        dst: Path = item["dst"]
        if dry_run:
            continue
        try:
            if item["convert"]:
                _convert_to_real_png(src, dst)
                if src.resolve() != dst.resolve() and src.exists():
                    src.unlink()
            else:
                src.rename(dst)
            processed += 1
        except Exception as exc:
            errors.append(f"{src}: {exc}")

    return {
        "dry_run": dry_run,
        "mode": mode,
        "already_ok": already_ok,
        "planned": len(tasks),
        "processed": processed if not dry_run else len(tasks),
        "items": tasks,
        "errors": errors,
    }
