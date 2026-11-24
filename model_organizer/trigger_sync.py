"""Sync Civitai trainedWords into SD WebUI native .json activation text safely."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any


def activation_text_from_info(data: dict[str, Any]) -> str:
    """Convert Civitai trainedWords into deduplicated activation text without <lora:...> tags."""
    words = data.get("trainedWords") or []
    if isinstance(words, str):
        words = [words]
    result: list[str] = []
    seen: set[str] = set()
    for value in words:
        text = re.sub(r"<lora:[^>]+>", "", str(value or ""), flags=re.I)
        text = re.sub(r"\s+", " ", text).strip(" ,")
        if not text:
            continue
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            result.append(text)
    return ", ".join(result)


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON atomically via temp file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".__mo_json_tmp_{uuid.uuid4().hex}.json")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _info_base(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".civitai.info"):
        return filename[: -len(".civitai.info")]
    return ""


def _build_trigger_update(info_path: Path) -> tuple[Path, dict[str, Any], str, str] | None:
    with info_path.open("r", encoding="utf-8-sig") as handle:
        info_data = json.load(handle)
    if not isinstance(info_data, dict):
        return None
    activation = activation_text_from_info(info_data)
    if not activation:
        return None

    base = _info_base(info_path.name)
    if not base:
        return None
    json_path = info_path.parent / f"{base}.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8-sig") as handle:
            json_data = json.load(handle)
        if not isinstance(json_data, dict):
            return None
        if str(json_data.get("activation text") or "").strip():
            return None
        action = "fill"
    else:
        json_data = {
            "description": "",
            "sd version": str(info_data.get("baseModel") or ""),
            "activation text": "",
            "preferred weight": 1,
            "negative text": "",
            "notes": "Auto-generated from Civitai trainedWords by Model Organizer",
        }
        action = "create"

    json_data["activation text"] = activation
    return json_path, json_data, action, activation


def sync_single_trigger_json(info_path: Path) -> dict[str, Any] | None:
    """Create or fill a single model's native .json trigger file from its .civitai.info."""
    update = _build_trigger_update(info_path)
    if update is None:
        return None
    json_path, json_data, action, activation = update
    write_json_atomic(json_path, json_data)
    return {
        "json_path": json_path,
        "action": action,
        "activation_text": activation,
    }


def collect_trigger_updates(
    directory: Path, recursive: bool = True
) -> tuple[list[tuple[Path, dict[str, Any], str, str]], list[str]]:
    """Collect missing or empty activation text updates without overwriting existing ones."""
    updates: list[tuple[Path, dict[str, Any], str, str]] = []
    errors: list[str] = []
    if not directory.is_dir():
        return updates, [f"目录不存在：{directory}"]

    iterator = directory.rglob("*.civitai.info") if recursive else directory.glob("*.civitai.info")
    for info_path in sorted(iterator):
        if not info_path.is_file():
            continue
        try:
            item = _build_trigger_update(info_path)
            if item is not None:
                updates.append(item)
        except Exception as exc:
            errors.append(f"{info_path}: {exc}")
    return updates, errors


def apply_trigger_updates(
    directory: Path, recursive: bool = True, dry_run: bool = False
) -> dict[str, Any]:
    """Preview or write trigger word JSON updates."""
    updates, errors = collect_trigger_updates(directory, recursive=recursive)
    fill_count = sum(1 for _, _, action, _ in updates if action == "fill")
    create_count = sum(1 for _, _, action, _ in updates if action == "create")
    written = 0

    items = []
    for json_path, json_data, action, activation in updates:
        items.append({
            "json_path": json_path,
            "action": action,
            "activation_text": activation,
        })
        if not dry_run:
            try:
                write_json_atomic(json_path, json_data)
                written += 1
            except Exception as exc:
                errors.append(f"{json_path}: {exc}")

    return {
        "dry_run": dry_run,
        "fill": fill_count,
        "create": create_count,
        "updated": written if not dry_run else len(updates),
        "items": items,
        "errors": errors,
    }
