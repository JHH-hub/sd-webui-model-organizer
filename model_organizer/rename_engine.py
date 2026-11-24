"""Atomic model + sidecar batch renamer with collision resolution and persistent undo."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .trigger_sync import write_json_atomic

INFO_SUFFIXES = (".civitai.info", ".info")
GROUP_SUFFIXES = (
    ".preview.jpeg",
    ".preview.webp",
    ".preview.png",
    ".preview.jpg",
    ".civitai.info",
    ".safetensors",
    ".jpeg",
    ".webp",
    ".ckpt",
    ".json",
    ".yaml",
    ".info",
    ".png",
    ".jpg",
    ".pth",
    ".pt",
    ".txt",
)
INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def sanitize_filename(value: Any) -> str:
    """Strip characters disallowed in Windows/POSIX filenames."""
    clean = INVALID_FILENAME_CHARS.sub("", str(value or ""))
    return re.sub(r"\s+", " ", clean).strip().rstrip(". ")


def info_base_name(filename: str) -> str:
    """Return stem before .civitai.info or .info."""
    lower = filename.lower()
    for suffix in INFO_SUFFIXES:
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    return ""


def normalized_name(value: Any) -> str:
    """Canonical comparison key retaining ASCII alphanumerics and CJK characters."""
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", str(value or "").casefold())


def build_display_name(data: dict[str, Any], include_checkpoint_version: bool = True) -> str:
    """Checkpoint keeps 'Model - Version' when distinct; LoRA keeps Model name."""
    model = data.get("model") or {}
    if not isinstance(model, dict):
        model = {}
    model_name = str(model.get("name") or "").strip()
    version_name = str(data.get("name") or "").strip()
    model_type = str(model.get("type") or "").strip()
    if not model_name:
        model_name = version_name or "Unknown"
    if include_checkpoint_version and model_type.casefold() == "checkpoint":
        if version_name and version_name.casefold() not in {"unknown", "none"}:
            model_norm = normalized_name(model_name)
            version_norm = normalized_name(version_name)
            if version_norm and version_norm not in model_norm:
                return f"{model_name} - {version_name}"
    return model_name


def load_info(path: Path, include_checkpoint_version: bool = True) -> dict[str, str]:
    """Parse Civitai info JSON into normalized metadata."""
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    model = data.get("model") or {}
    if not isinstance(model, dict):
        model = {}
    return {
        "model_name": str(model.get("name") or data.get("name") or "Unknown"),
        "version_name": str(data.get("name") or ""),
        "model_type": str(model.get("type") or ""),
        "suggested_name": sanitize_filename(
            build_display_name(data, include_checkpoint_version=include_checkpoint_version)
        ),
    }


def group_files(directory: Path, base: str) -> list[tuple[Path, str]]:
    """Collect model file and all companion sidecar files sharing base stem in directory."""
    result: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for suffix in GROUP_SUFFIXES:
        candidate = directory / f"{base}{suffix}"
        if candidate.is_file() and candidate not in seen:
            seen.add(candidate)
            result.append((candidate, suffix))
    return result


def append_versions_for_collisions(records: list[dict[str, Any]]) -> None:
    """Append version_name when multiple entries in the same directory share suggested_name."""
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        if record.get("error"):
            continue
        parent_key = str(record.get("parent_dir") or "").casefold()
        name_key = normalized_name(record.get("suggested_name"))
        if name_key:
            counts[(parent_key, name_key)] = counts.get((parent_key, name_key), 0) + 1

    for record in records:
        if record.get("error"):
            continue
        parent_key = str(record.get("parent_dir") or "").casefold()
        suggested = str(record.get("suggested_name") or "")
        name_key = normalized_name(suggested)
        if counts.get((parent_key, name_key), 0) < 2:
            continue
        if str(record.get("model_type") or "").casefold() == "checkpoint":
            continue
        version = str(record.get("version_name") or "").strip()
        if not version or normalized_name(version) in normalized_name(suggested):
            continue
        record["suggested_name"] = sanitize_filename(f"{suggested} - {version}")


def scan_rename_candidates(
    directory: Path,
    recursive: bool = False,
    include_checkpoint_version: bool = True,
) -> list[dict[str, Any]]:
    """Scan .civitai.info/.info files and propose clean target filenames."""
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在：{directory}")

    base_resolved = directory.resolve()
    iterator = base_resolved.rglob("*") if recursive else base_resolved.iterdir()
    info_files = sorted(
        p for p in iterator
        if p.is_file() and any(p.name.lower().endswith(s) for s in INFO_SUFFIXES)
    )

    # Prefer .civitai.info if both .civitai.info and .info exist for same stem in same dir
    dedup: dict[tuple[Path, str], Path] = {}
    for p in info_files:
        base = info_base_name(p.name)
        if not base:
            continue
        key = (p.parent.resolve(), base)
        prev = dedup.get(key)
        if prev is None or p.name.lower().endswith(".civitai.info"):
            dedup[key] = p

    records: list[dict[str, Any]] = []
    for (parent_dir, base), info_path in sorted(dedup.items(), key=lambda kv: str(kv[1]).casefold()):
        rel_dir = "." if parent_dir == base_resolved else parent_dir.relative_to(base_resolved).as_posix()
        try:
            meta = load_info(info_path, include_checkpoint_version=include_checkpoint_version)
            rec: dict[str, Any] = {
                **meta,
                "error": "",
            }
        except Exception as exc:
            rec = {
                "model_name": "",
                "version_name": "",
                "model_type": "解析失败",
                "suggested_name": base,
                "error": str(exc),
            }
        companions = group_files(parent_dir, base)
        rec.update({
            "parent_dir": parent_dir,
            "rel_dir": rel_dir,
            "base": base,
            "info_path": info_path,
            "file_count": len(companions),
        })
        records.append(rec)

    append_versions_for_collisions(records)
    for rec in records:
        if rec.get("error"):
            rec["needs_rename"] = False
        else:
            rec["needs_rename"] = sanitize_filename(rec.get("suggested_name")) != rec["base"]
    return records


def plan_renames(items: list[dict[str, Any]]) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Build collision-checked [(src, dst)] move list from scan/table rows."""
    rows: list[tuple[Path, str, str, list[tuple[Path, str]]]] = []
    errors: list[str] = []

    for item in items:
        if not item.get("enabled", True):
            continue
        parent_dir = Path(item["parent_dir"]).resolve()
        base = str(item["base"])
        new_name = sanitize_filename(item.get("new_name"))
        if not new_name:
            continue
        if item.get("error"):
            errors.append(f"{base}：信息文件解析失败，无法重命名（{item['error']}）")
            continue
        files = group_files(parent_dir, base)
        if not files:
            errors.append(f"{base}：目录下未找到任何匹配文件")
            continue
        rows.append((parent_dir, base, new_name, files))

    if errors:
        return [], errors

    source_keys = {str(src.resolve()).casefold() for _, _, _, files in rows for src, _ in files}
    planned_dest_keys: set[str] = set()
    plan: list[tuple[Path, Path]] = []

    for parent_dir, _base, new_name, files in rows:
        for src, suffix in files:
            dst = (parent_dir / f"{new_name}{suffix}").resolve()
            dst_key = str(dst).casefold()
            if dst_key in planned_dest_keys:
                errors.append(f"目标文件名重复冲突：{dst.name}")
            elif dst.exists() and dst_key not in source_keys:
                errors.append(f"目标文件已存在：{dst.name}")
            else:
                planned_dest_keys.add(dst_key)
                if src.resolve() != dst or src.name != dst.name:
                    plan.append((src, dst))

    if errors:
        return [], errors
    return plan, []


def execute_atomic_moves(plan: list[tuple[Path, Path]]) -> None:
    """Two-phase temporary-file rename with full rollback on any failure."""
    if not plan:
        return
    staged: list[tuple[Path, Path, Path]] = []
    try:
        for source, destination in plan:
            temporary = source.with_name(f".__mo_tmp_{uuid.uuid4().hex}{source.suffix}")
            source.rename(temporary)
            staged.append((temporary, source, destination))
        for temporary, _source, destination in staged:
            temporary.rename(destination)
    except Exception as exc:
        rollback_errors: list[str] = []
        for temporary, source, destination in reversed(staged):
            try:
                if temporary.exists() and not source.exists():
                    temporary.rename(source)
                elif destination.exists() and not source.exists():
                    destination.rename(source)
            except Exception as rollback_exc:
                rollback_errors.append(f"{source.name}: {rollback_exc}")
        detail = str(exc)
        if rollback_errors:
            detail += "\n回滚过程出现异常：\n" + "\n".join(rollback_errors[:10])
        raise RuntimeError(detail) from exc


class RenameHistoryStore:
    """Persistent atomic JSON log enabling 1-click undo of any rename batch."""

    def __init__(self, history_path: Path, max_batches: int = 50):
        self.history_path = Path(history_path)
        self.max_batches = max_batches

    def _load(self) -> list[dict[str, Any]]:
        if not self.history_path.is_file():
            return []
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8-sig"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save(self, batches: list[dict[str, Any]]) -> None:
        trimmed = batches[-self.max_batches :]
        write_json_atomic(self.history_path, trimmed)

    def record(
        self,
        directory: Path,
        moves: list[tuple[Path, Path]],
        summary: str = "",
    ) -> dict[str, Any]:
        batches = self._load()
        now = datetime.now(timezone.utc).astimezone()
        batch = {
            "batch_id": now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
            "timestamp": now.isoformat(timespec="seconds"),
            "directory": str(directory),
            "summary": summary or f"更名 {len(moves)} 个文件",
            "reverted": False,
            "moves": [{"src": str(src), "dst": str(dst)} for src, dst in moves],
        }
        batches.append(batch)
        self._save(batches)
        return batch

    def list_batches(self, include_reverted: bool = False) -> list[dict[str, Any]]:
        batches = self._load()
        if not include_reverted:
            batches = [b for b in batches if not b.get("reverted")]
        return list(reversed(batches))

    def undo_batch(self, batch_id: str | None = None) -> dict[str, Any]:
        batches = self._load()
        target_idx: int | None = None
        if batch_id:
            for idx in range(len(batches) - 1, -1, -1):
                if batches[idx].get("batch_id") == batch_id:
                    target_idx = idx
                    break
            if target_idx is None:
                raise KeyError(f"未找到改名记录批次：{batch_id}")
        else:
            for idx in range(len(batches) - 1, -1, -1):
                if not batches[idx].get("reverted"):
                    target_idx = idx
                    break
            if target_idx is None:
                raise ValueError("没有可撤销的改名批次")

        batch = batches[target_idx]
        if batch.get("reverted"):
            raise ValueError(f"批次 {batch.get('batch_id')} 已经撤销过了")

        reverse_plan: list[tuple[Path, Path]] = [
            (Path(m["dst"]), Path(m["src"])) for m in batch.get("moves", [])
        ]
        if not reverse_plan:
            batch["reverted"] = True
            self._save(batches)
            return {"batch_id": batch["batch_id"], "restored": 0}

        dst_keys = {str(src.resolve()).casefold() for src, _ in reverse_plan}
        for current_file, original_file in reverse_plan:
            if not current_file.exists():
                raise FileNotFoundError(f"无法撤销，文件已不存在：{current_file}")
            orig_key = str(original_file.resolve()).casefold()
            if original_file.exists() and orig_key not in dst_keys:
                raise FileExistsError(f"无法撤销，原路径已被占用：{original_file}")

        execute_atomic_moves(reverse_plan)
        batch["reverted"] = True
        batch["reverted_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        self._save(batches)
        return {
            "batch_id": batch["batch_id"],
            "restored": len(reverse_plan),
            "directory": batch.get("directory", ""),
        }
