"""Model directory discovery and safe subfolder resolution."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MODEL_TYPE_SPECS: dict[str, dict[str, str]] = {
    "lora": {
        "label": "LoRA (models/Lora)",
        "default_rel": os.path.join("models", "Lora"),
    },
    "checkpoint": {
        "label": "Checkpoint 大模型 (models/Stable-diffusion)",
        "default_rel": os.path.join("models", "Stable-diffusion"),
    },
    "embedding": {
        "label": "Embedding / TI (embeddings)",
        "default_rel": "embeddings",
    },
    "lycoris": {
        "label": "LyCORIS (models/LyCORIS)",
        "default_rel": os.path.join("models", "LyCORIS"),
    },
}


def _first_existing_dir(*values: Any) -> Path | None:
    for val in values:
        if not val:
            continue
        if isinstance(val, (list, tuple, set)):
            found = _first_existing_dir(*val)
            if found is not None:
                return found
            continue
        try:
            candidate = Path(os.fspath(val))
        except TypeError:
            continue
        if candidate.is_dir():
            return candidate.resolve()
    return None


def resolve_model_dirs(webui_root: Path | None = None, cmd_opts: Any = None) -> dict[str, Path]:
    """Resolve model directories from WebUI cmd_opts or fallback to webui_root / cwd."""
    root = Path(webui_root).resolve() if webui_root is not None else Path.cwd().resolve()

    opt_map = {
        "lora": ("lora_dir", "lora_dirs"),
        "checkpoint": ("ckpt_dir", "ckpt_dirs"),
        "embedding": ("embeddings_dir",),
        "lycoris": ("lyco_dir", "lyco_dirs"),
    }

    dirs: dict[str, Path] = {}
    for key, spec in MODEL_TYPE_SPECS.items():
        override = None
        if cmd_opts is not None:
            vals = [getattr(cmd_opts, attr, None) for attr in opt_map.get(key, ())]
            override = _first_existing_dir(*vals)
        dirs[key] = override if override is not None else (root / spec["default_rel"]).resolve()
    return dirs


def list_subfolders(base_dir: Path) -> list[str]:
    """Return [""] plus sorted POSIX relative subfolder paths under base_dir."""
    if not base_dir.is_dir():
        return [""]
    subs: list[str] = []
    for root, dirnames, _ in os.walk(base_dir):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d != "__pycache__"
        )
        cur = Path(root)
        if cur != base_dir:
            rel = cur.relative_to(base_dir).as_posix()
            subs.append(rel)
    subs.sort()
    return [""] + subs


def safe_resolve_subdir(base_dir: Path, rel_sub: str) -> Path:
    """Resolve rel_sub safely inside base_dir, rejecting path traversal."""
    base_resolved = base_dir.resolve()
    clean = (rel_sub or "").strip().replace("\\", "/").strip("/")
    if not clean:
        return base_resolved
    target = (base_resolved / clean).resolve()
    try:
        target.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"非法子目录路径（禁止越权跳转）：{rel_sub}") from exc
    return target
