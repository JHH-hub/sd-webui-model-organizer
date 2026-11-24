"""Batch Civitai new-version checker using local .civitai.info metadata."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .civitai_client import CivitaiClient
from .paths import logical_abspath
from .rename_engine import INFO_SUFFIXES, info_base_name


def _safe_int(val: Any) -> int:
    try:
        return int(str(val or "").strip())
    except (TypeError, ValueError):
        return 0


def _parse_ts(val: Any) -> float:
    raw = str(val or "").strip()
    if not raw:
        return 0.0
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


def collect_local_civitai_models(directory: Path, recursive: bool = True) -> list[dict[str, Any]]:
    """Collect local models with numeric modelId grouped cleanly by modelId."""
    base_logical = logical_abspath(directory)
    if not base_logical.is_dir():
        raise FileNotFoundError(f"目录不存在：{base_logical}")

    info_files: list[Path] = []
    if recursive:
        for root_str, dirnames, filenames in os.walk(base_logical):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
            cur_dir = Path(root_str)
            for fn in filenames:
                if any(fn.lower().endswith(suf) for suf in INFO_SUFFIXES):
                    info_files.append(cur_dir / fn)
    else:
        for p in base_logical.iterdir():
            if p.is_file() and any(p.name.lower().endswith(suf) for suf in INFO_SUFFIXES):
                info_files.append(p)

    # Prefer .civitai.info over .info when both exist in the same directory for one stem.
    dedup: dict[tuple[Path, str], Path] = {}
    for p in sorted(info_files, key=lambda x: str(x).casefold()):
        base = info_base_name(p.name)
        if not base:
            continue
        key = (logical_abspath(p.parent), base)
        prev = dedup.get(key)
        if prev is None or p.name.lower().endswith(".civitai.info"):
            dedup[key] = p

    groups: dict[str, dict[str, Any]] = {}
    for (_parent_dir, base), info_path in sorted(dedup.items(), key=lambda kv: str(kv[1]).casefold()):
        try:
            data = json.loads(info_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        mid_raw = str(data.get("modelId") or "").strip()
        if not mid_raw.isdigit():
            continue

        model_obj = data.get("model") if isinstance(data.get("model"), dict) else {}
        vid_raw = str(data.get("id") or "").strip()
        vname = str(data.get("name") or vid_raw or "").strip()
        base_model = str(data.get("baseModel") or "").strip()
        ts_raw = data.get("publishedAt") or data.get("createdAt") or ""
        sort_key = (_parse_ts(ts_raw), _safe_int(vid_raw))

        entry = {
            "version_id": vid_raw,
            "version_name": vname,
            "base_model": base_model,
            "sort_key": sort_key,
            "local_file": base,
            "model_name": str(model_obj.get("name") or base),
        }

        grp = groups.get(mid_raw)
        if grp is None:
            grp = {
                "model_id": mid_raw,
                "model_name": entry["model_name"],
                "local_version_ids": set(),
                "local_base_models": set(),
                "best_local": entry,
            }
            groups[mid_raw] = grp
        if vid_raw:
            grp["local_version_ids"].add(vid_raw)
        if base_model:
            grp["local_base_models"].add(base_model.casefold())
        if entry["sort_key"] > grp["best_local"]["sort_key"]:
            grp["best_local"] = entry
            if model_obj.get("name"):
                grp["model_name"] = str(model_obj.get("name"))

    return sorted(groups.values(), key=lambda g: (str(g["model_name"]).casefold(), _safe_int(g["model_id"])))


def check_model_updates(
    directory: Path,
    client: CivitaiClient,
    recursive: bool = True,
    limit: int = 30,
    same_base_model_only: bool = True,
    delay_sec: float = 0.4,
) -> dict[str, Any]:
    """Check Civitai for newer published versions of local .civitai.info models."""
    groups = collect_local_civitai_models(directory, recursive=recursive)
    cap = max(1, int(limit))
    target_groups = groups[:cap]
    norm_domain = client.config.normalized_domain()

    updates: list[dict[str, Any]] = []
    errors: list[str] = []
    checked = 0
    up_to_date = 0

    for idx, grp in enumerate(target_groups):
        model_id = grp["model_id"]
        best_local = grp["best_local"]
        try:
            model_data = client.get_model(model_id)
            checked += 1
            versions = model_data.get("modelVersions") or []
            if not isinstance(versions, list):
                versions = []

            candidates: list[tuple[tuple[float, int], dict[str, Any]]] = []
            local_bases: set[str] = grp["local_base_models"]
            for v in versions:
                if not isinstance(v, dict):
                    continue
                status = str(v.get("status") or "").strip()
                if status and status != "Published":
                    continue
                v_base = str(v.get("baseModel") or "").strip()
                if same_base_model_only and local_bases and v_base.casefold() not in local_bases:
                    continue
                ts_val = _parse_ts(v.get("publishedAt") or v.get("createdAt") or "")
                vid_int = _safe_int(v.get("id"))
                candidates.append(((ts_val, vid_int), v))

            if not candidates:
                up_to_date += 1
            else:
                candidates.sort(key=lambda item: item[0], reverse=True)
                latest_sort, latest = candidates[0]
                latest_vid = str(latest.get("id") or "").strip()
                local_sort = best_local["sort_key"]

                is_newer = (
                    latest_vid
                    and latest_vid not in grp["local_version_ids"]
                    and (
                        latest_sort[0] > local_sort[0]
                        or (latest_sort[0] == local_sort[0] and latest_sort[1] > local_sort[1])
                    )
                )
                if is_newer:
                    pub_raw = str(latest.get("publishedAt") or latest.get("createdAt") or "").strip()
                    pub_date = pub_raw[:10] if len(pub_raw) >= 10 else pub_raw
                    model_title = str(model_data.get("name") or grp["model_name"] or model_id)
                    updates.append({
                        "model_id": model_id,
                        "model_name": model_title,
                        "local_version_name": best_local["version_name"],
                        "local_version_id": best_local["version_id"],
                        "latest_version_name": str(latest.get("name") or latest_vid),
                        "latest_version_id": latest_vid,
                        "base_model": str(latest.get("baseModel") or ""),
                        "published_at": pub_date,
                        "url": f"https://{norm_domain}/models/{model_id}?modelVersionId={latest_vid}",
                        "local_file": best_local["local_file"],
                    })
                else:
                    up_to_date += 1
        except Exception as exc:
            errors.append(f"{grp['model_name']} (#{model_id}): {exc}")

        if delay_sec > 0 and idx + 1 < len(target_groups):
            time.sleep(delay_sec)

    return {
        "total_models": len(groups),
        "checked": checked,
        "updates_found": len(updates),
        "up_to_date": up_to_date,
        "updates": updates,
        "errors": errors,
    }
