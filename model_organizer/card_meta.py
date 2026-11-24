"""Safe local model card metadata & example image extraction for Extra Networks UI."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .civitai_client import normalize_domain
from .paths import logical_abspath
from .trigger_sync import activation_text_from_info

MODEL_BINARY_EXTS = (".safetensors", ".ckpt", ".pt", ".bin", ".pth")


def _is_within_root(candidate: Path, root: Path) -> bool:
    """Case-aware containment check preserving NTFS junctions without dereferencing."""
    try:
        cand_norm = Path(os.path.normcase(str(logical_abspath(candidate))))
        root_norm = Path(os.path.normcase(str(logical_abspath(root))))
        cand_norm.relative_to(root_norm)
        return True
    except ValueError:
        return False


def _is_video_entry(img: dict[str, Any]) -> bool:
    if str(img.get("type") or "").strip().lower() == "video":
        return True
    url = str(img.get("url") or "").strip().lower().split("?", 1)[0]
    return url.endswith((".mp4", ".webm", ".mov"))


def _is_nsfw_entry(img: dict[str, Any]) -> bool:
    if img.get("nsfw") is True:
        return True
    level = img.get("nsfwLevel")
    if isinstance(level, (int, float)) and level > 1:
        return True
    if isinstance(img.get("nsfw"), str) and img["nsfw"].strip().lower() not in ("", "none", "false", "0"):
        return True
    return False


def resolve_model_file(
    model_dirs: dict[str, Path],
    model_type: str,
    name: str,
    path_hint: str = "",
) -> Path | None:
    """Resolve model binary strictly inside configured model_dirs (junction-safe)."""
    roots = [logical_abspath(p) for p in model_dirs.values() if p]
    if not roots:
        return None

    hint = (path_hint or "").strip()
    if hint:
        cand = logical_abspath(hint)
        if any(_is_within_root(cand, r) for r in roots) and cand.is_file():
            if cand.name.lower().endswith(MODEL_BINARY_EXTS):
                return cand
        # Do not escape containment even if path_hint was outside; still allow safe name lookup below only if no traversal?
        # If path_hint points outside allowed roots with explicit path separators, reject unless fallback name is non-empty.
        if "/" in hint or "\\" in hint:
            return None

    target_name = (name or "").strip().replace("\\", "/")
    if not target_name:
        return None
    target_norm = target_name.casefold()

    search_roots: list[Path] = []
    preferred = model_dirs.get((model_type or "").strip().lower())
    if preferred:
        search_roots.append(logical_abspath(preferred))
    for r in roots:
        if r not in search_roots:
            search_roots.append(r)

    for base_root in search_roots:
        if not base_root.is_dir():
            continue
        for root_str, dirnames, filenames in os.walk(base_root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
            cur_dir = Path(root_str)
            for fn in sorted(filenames, key=str.casefold):
                low = fn.lower()
                ext = next((e for e in MODEL_BINARY_EXTS if low.endswith(e)), None)
                if ext is None:
                    continue
                stem = fn[: -len(ext)]
                file_path = cur_dir / fn
                if stem.casefold() == target_norm:
                    return file_path
                try:
                    rel_stem = (cur_dir.relative_to(base_root) / stem).as_posix().lstrip("./")
                    if rel_stem.casefold() == target_norm:
                        return file_path
                except ValueError:
                    pass
    return None


def extract_example_images(info_data: dict[str, Any], max_images: int = 12) -> list[dict[str, Any]]:
    """Extract clean image URLs, NSFW flags, and generation parameters from .civitai.info."""
    raw_images = info_data.get("images") or []
    if not isinstance(raw_images, list):
        return []

    out: list[dict[str, Any]] = []
    for img in raw_images:
        if not isinstance(img, dict):
            continue
        url = str(img.get("url") or "").strip()
        if not url or _is_video_entry(img):
            continue
        meta = img.get("meta") if isinstance(img.get("meta"), dict) else {}
        w = img.get("width") or meta.get("width")
        h = img.get("height") or meta.get("height")
        size_val = str(meta.get("Size") or "").strip()
        if not size_val and w and h:
            size_val = f"{w}x{h}"

        level_raw = img.get("nsfwLevel")
        nsfw_level = int(level_raw) if isinstance(level_raw, (int, float)) else 0

        out.append({
            "url": url,
            "nsfw": _is_nsfw_entry(img),
            "nsfw_level": nsfw_level,
            "width": w,
            "height": h,
            "prompt": str(meta.get("prompt") or "").strip(),
            "negative_prompt": str(meta.get("negativePrompt") or "").strip(),
            "seed": str(meta.get("seed") or "").strip(),
            "steps": str(meta.get("steps") or "").strip(),
            "cfg_scale": str(meta.get("cfgScale") or "").strip(),
            "sampler": str(meta.get("sampler") or "").strip(),
            "model": str(meta.get("Model") or "").strip(),
            "size": size_val,
        })
        if len(out) >= max(1, int(max_images)):
            break
    return out


def _clean_trained_words(info_data: dict[str, Any]) -> list[str]:
    words = info_data.get("trainedWords") or []
    if isinstance(words, str):
        words = [words]
    if not isinstance(words, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for val in words:
        text = re.sub(r"<lora:[^>]+>", "", str(val or ""), flags=re.I)
        text = re.sub(r"\s+", " ", text).strip(" ,")
        if not text:
            continue
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def load_card_metadata(
    model_dirs: dict[str, Path],
    model_type: str,
    name: str,
    path_hint: str = "",
    domain: str = "civitai.com",
) -> dict[str, Any]:
    """Load triggers, Civitai URL, and example gallery metadata for one WebUI model card."""
    model_path = resolve_model_file(model_dirs, model_type=model_type, name=name, path_hint=path_hint)
    if model_path is None:
        return {"found": False, "error": "未找到对应的本地模型文件"}

    stem = model_path.stem
    parent = model_path.parent
    civitai_info_path = parent / f"{stem}.civitai.info"
    alt_info_path = parent / f"{stem}.info"
    info_data = _read_json_dict(civitai_info_path)
    if info_data is None:
        info_data = _read_json_dict(alt_info_path)
    has_info = info_data is not None

    json_path = parent / f"{stem}.json"
    json_data = _read_json_dict(json_path)
    has_json = json_data is not None

    info_obj = info_data or {}
    model_obj = info_obj.get("model") if isinstance(info_obj.get("model"), dict) else {}

    json_activation = str((json_data or {}).get("activation text") or "").strip()
    info_activation = activation_text_from_info(info_obj) if info_obj else ""
    activation_text = json_activation or info_activation

    model_id = info_obj.get("modelId")
    version_id = info_obj.get("id")
    norm_domain = normalize_domain(domain)
    civitai_url = ""
    if model_id is not None and str(model_id).strip().isdigit():
        civitai_url = f"https://{norm_domain}/models/{str(model_id).strip()}"
        if version_id is not None and str(version_id).strip().isdigit():
            civitai_url += f"?modelVersionId={str(version_id).strip()}"

    return {
        "found": True,
        "stem": stem,
        "model_name": str(model_obj.get("name") or stem),
        "version_name": str(info_obj.get("name") or ""),
        "base_model": str(info_obj.get("baseModel") or (json_data or {}).get("sd version") or ""),
        "model_type_civitai": str(model_obj.get("type") or ""),
        "model_id": str(model_id).strip() if model_id is not None else "",
        "version_id": str(version_id).strip() if version_id is not None else "",
        "civitai_url": civitai_url,
        "activation_text": activation_text,
        "trained_words": _clean_trained_words(info_obj),
        "has_info": has_info,
        "has_json": has_json,
        "images": extract_example_images(info_obj) if info_obj else [],
    }
