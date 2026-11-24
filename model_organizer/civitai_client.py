"""Lightweight Civitai (.com/.red) API & resumable download + SHA-256 backfill client."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from .rename_engine import build_display_name, execute_atomic_moves, group_files, sanitize_filename
from .trigger_sync import sync_single_trigger_json, write_json_atomic

ALLOWED_DOMAINS = ("civitai.com", "civitai.red")
MODEL_BINARY_EXTS = (".safetensors", ".ckpt", ".pt", ".bin")
PREVIEW_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class CivitaiConfig:
    domain: str = "civitai.com"
    api_key: str = ""
    proxy: str = ""
    skip_nsfw_preview: bool = False
    timeout: int = 30

    def normalized_domain(self) -> str:
        return normalize_domain(self.domain)


def normalize_domain(domain: str) -> str:
    """Validate and normalize Civitai host ('civitai.com' or 'civitai.red')."""
    d = (domain or "").strip().lower()
    if d.startswith("https://"):
        d = d[len("https://") :]
    elif d.startswith("http://"):
        d = d[len("http://") :]
    d = d.split("/")[0]
    if d in ALLOWED_DOMAINS:
        return d
    return "civitai.com"


def parse_civitai_url(url_or_id: str) -> tuple[str, str | None]:
    """Extract (model_id, optional_version_id) from Civitai URL or numeric model ID."""
    raw = (url_or_id or "").strip()
    if not raw:
        raise ValueError("请输入 Civitai 模型链接或数字 ID")
    if re.fullmatch(r"\d+", raw):
        return raw, None

    parsed = urllib.parse.urlsplit(raw if "://" in raw else f"https://{raw}")
    qs = urllib.parse.parse_qs(parsed.query)
    version_id = None
    for key in ("modelVersionId", "versionId"):
        if qs.get(key):
            val = qs[key][0].strip()
            if re.fullmatch(r"\d+", val):
                version_id = val
                break

    m = re.search(r"/(?:api/v\d+/)?models/(\d+)", parsed.path)
    if not m:
        raise ValueError(f"无法从输入中解析出 Civitai 模型 ID：{url_or_id}")
    return m.group(1), version_id


def rewrite_civitai_host(url: str, domain: str) -> str:
    """Switch primary civitai.com/civitai.red host while keeping image CDN domains intact."""
    target = normalize_domain(domain)
    parsed = urllib.parse.urlsplit(url)
    host = parsed.netloc.lower()
    if host in ("civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"):
        return urllib.parse.urlunsplit((parsed.scheme or "https", target, parsed.path, parsed.query, parsed.fragment))
    return url


def compute_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Compute streaming uppercase SHA-256 hex digest without loading model into RAM."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().upper()


def _is_nsfw_image(img: dict[str, Any]) -> bool:
    if img.get("nsfw") is True:
        return True
    level = img.get("nsfwLevel")
    if isinstance(level, (int, float)) and level > 1:
        return True
    if isinstance(img.get("nsfw"), str) and img["nsfw"].strip().lower() not in ("", "none", "false", "0"):
        return True
    return False


def _is_video_url(url: str) -> bool:
    path = urllib.parse.urlsplit(url).path.lower()
    return path.endswith((".mp4", ".webm", ".mov"))


class CivitaiClient:
    """Configurable per-request Civitai client supporting .com/.red, proxy & resumable downloads."""

    def __init__(self, config: CivitaiConfig | None = None, session: requests.Session | None = None):
        self.config = config or CivitaiConfig()
        self.session = session or requests.Session()

    def _headers(self, include_auth: bool = True) -> dict[str, str]:
        headers = {
            "User-Agent": "ModelOrganizer/1.0 (+https://pxlsan.cn)",
        }
        api_key = (self.config.api_key or "").strip()
        if include_auth and api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _proxies(self) -> dict[str, str] | None:
        proxy = (self.config.proxy or "").strip()
        if not proxy:
            return None
        return {"http": proxy, "https": proxy}

    def _base_url(self) -> str:
        return f"https://{self.config.normalized_domain()}/api/v1"

    def _safe_error(self, resp: requests.Response, prefix: str) -> RuntimeError:
        snippet = (resp.text or "").strip().replace("\n", " ")[:180]
        return RuntimeError(f"{prefix}（HTTP {resp.status_code}）：{snippet or resp.reason}")

    def get_model(self, model_id: str) -> dict[str, Any]:
        url = f"{self._base_url()}/models/{model_id}"
        resp = self.session.get(url, headers=self._headers(), proxies=self._proxies(), timeout=self.config.timeout)
        if not resp.ok:
            raise self._safe_error(resp, f"获取模型 #{model_id} 失败")
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("Civitai 返回格式异常")
        return data

    def get_model_version(self, version_id: str) -> dict[str, Any]:
        url = f"{self._base_url()}/model-versions/{version_id}"
        resp = self.session.get(url, headers=self._headers(), proxies=self._proxies(), timeout=self.config.timeout)
        if not resp.ok:
            raise self._safe_error(resp, f"获取版本 #{version_id} 失败")
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("Civitai 返回格式异常")
        return data

    def get_model_version_by_hash(self, sha256_hex: str) -> dict[str, Any] | None:
        clean_hash = (sha256_hex or "").strip()
        if not clean_hash:
            return None
        url = f"{self._base_url()}/model-versions/by-hash/{clean_hash}"
        resp = self.session.get(url, headers=self._headers(), proxies=self._proxies(), timeout=self.config.timeout)
        if resp.status_code == 404:
            return None
        if not resp.ok:
            raise self._safe_error(resp, f"Hash 查询失败 ({clean_hash[:10]}…)")
        data = resp.json()
        return data if isinstance(data, dict) else None

    def pick_preview_image(self, version_data: dict[str, Any]) -> dict[str, Any] | None:
        images = version_data.get("images") or []
        for img in images:
            if not isinstance(img, dict):
                continue
            img_url = str(img.get("url") or "").strip()
            if not img_url or _is_video_url(img_url) or str(img.get("type") or "").lower() == "video":
                continue
            if self.config.skip_nsfw_preview and _is_nsfw_image(img):
                continue
            return img
        return None

    def download_file(
        self,
        url: str,
        dst_path: Path,
        on_progress: Callable[[int, int | None], None] | None = None,
        rewrite_host: bool = True,
    ) -> Path:
        """Stream file to <dst_path>.part with HTTP Range resume and atomic commit."""
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dst_path.with_name(dst_path.name + ".part")
        req_url = rewrite_civitai_host(url, self.config.normalized_domain()) if rewrite_host else url

        offset = part_path.stat().st_size if part_path.is_file() else 0
        headers = self._headers()
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"

        resp = self.session.get(
            req_url,
            headers=headers,
            proxies=self._proxies(),
            stream=True,
            timeout=self.config.timeout,
        )
        if resp.status_code == 416 and offset > 0:
            resp.close()
            if part_path.exists():
                part_path.unlink()
            offset = 0
            resp = self.session.get(
                req_url,
                headers=self._headers(),
                proxies=self._proxies(),
                stream=True,
                timeout=self.config.timeout,
            )

        if not resp.ok:
            resp.close()
            if resp.status_code in (401, 403):
                raise RuntimeError("下载被拒绝（401/403），该模型可能需要在设置中填写 Civitai API Key")
            raise self._safe_error(resp, f"下载文件失败 ({dst_path.name})")

        append_mode = resp.status_code == 206 and offset > 0
        content_len = resp.headers.get("Content-Length")
        total = (offset + int(content_len)) if (append_mode and content_len and content_len.isdigit()) else (
            int(content_len) if (content_len and content_len.isdigit()) else None
        )
        written = offset if append_mode else 0

        with part_path.open("ab" if append_mode else "wb") as out_f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                out_f.write(chunk)
                written += len(chunk)
                if on_progress is not None:
                    on_progress(written, total)
        resp.close()

        part_path.replace(dst_path)
        return dst_path

    def build_civitai_info(self, model_data: dict[str, Any] | None, version_data: dict[str, Any]) -> dict[str, Any]:
        """Build standardized .civitai.info payload preserving model name & type."""
        info = copy.deepcopy(version_data)
        if model_data:
            info["modelId"] = model_data.get("id") or info.get("modelId")
            info["model"] = {
                "name": model_data.get("name", ""),
                "type": model_data.get("type", ""),
                "nsfw": bool(model_data.get("nsfw", False)),
                "poi": bool(model_data.get("poi", False)),
                "mode": model_data.get("mode"),
                "tags": model_data.get("tags") or [],
            }
        return info

    def _select_primary_file(self, version_data: dict[str, Any]) -> dict[str, Any]:
        files = [f for f in (version_data.get("files") or []) if isinstance(f, dict) and f.get("downloadUrl")]
        if not files:
            raise ValueError("该模型版本不包含可下载文件")
        for f in files:
            if f.get("primary") and str(f.get("type", "")).lower() != "training data":
                return f
        for f in files:
            name = str(f.get("name") or "").lower()
            if name.endswith(MODEL_BINARY_EXTS):
                return f
        return files[0]

    def _unique_target_stem(self, target_dir: Path, preferred: str, version_name: str, ignore_base: str) -> str:
        clean = sanitize_filename(preferred) or ignore_base
        if clean.casefold() == ignore_base.casefold():
            return clean

        def _exists_conflict(stem: str) -> bool:
            if stem.casefold() == ignore_base.casefold():
                return False
            for ext in MODEL_BINARY_EXTS + (".civitai.info", ".json"):
                if (target_dir / f"{stem}{ext}").exists():
                    return True
            return False

        if not _exists_conflict(clean):
            return clean
        if version_name:
            with_ver = sanitize_filename(f"{clean} - {version_name}")
            if with_ver and not _exists_conflict(with_ver):
                return with_ver
        idx = 2
        while True:
            numbered = sanitize_filename(f"{clean} ({idx})")
            if not _exists_conflict(numbered):
                return numbered
            idx += 1

    def download_model_version(
        self,
        model_data: dict[str, Any],
        version_data: dict[str, Any],
        target_dir: Path,
        auto_rename: bool = True,
        auto_sync_triggers: bool = True,
        include_checkpoint_version: bool = True,
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> dict[str, Any]:
        """Download model binary, .civitai.info, preview image, then auto-organize."""
        target_dir = Path(target_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        file_info = self._select_primary_file(version_data)
        raw_filename = sanitize_filename(file_info.get("name") or f"model_{version_data.get('id', 'dl')}.safetensors")
        if "." not in raw_filename:
            raw_filename += ".safetensors"
        base_stem = os.path.splitext(raw_filename)[0]

        model_path = target_dir / raw_filename
        self.download_file(str(file_info["downloadUrl"]), model_path, on_progress=on_progress, rewrite_host=True)

        info_payload = self.build_civitai_info(model_data, version_data)
        info_path = target_dir / f"{base_stem}.civitai.info"
        write_json_atomic(info_path, info_payload)

        preview_path: Path | None = None
        img_meta = self.pick_preview_image(version_data)
        if img_meta and img_meta.get("url"):
            img_url = str(img_meta["url"])
            ext = Path(urllib.parse.urlsplit(img_url).path).suffix.lower()
            if ext not in PREVIEW_EXTS:
                ext = ".png"
            preview_candidate = target_dir / f"{base_stem}.preview{ext}"
            try:
                self.download_file(img_url, preview_candidate, rewrite_host=False)
                preview_path = preview_candidate
            except Exception:
                preview_path = None

        final_stem = base_stem
        if auto_rename:
            suggested = build_display_name(info_payload, include_checkpoint_version=include_checkpoint_version)
            final_stem = self._unique_target_stem(
                target_dir,
                suggested,
                str(info_payload.get("name") or ""),
                base_stem,
            )
            if final_stem != base_stem:
                companions = group_files(target_dir, base_stem)
                moves = [(src, target_dir / f"{final_stem}{suf}") for src, suf in companions]
                execute_atomic_moves(moves)
                model_path = target_dir / f"{final_stem}{model_path.suffix}"
                info_path = target_dir / f"{final_stem}.civitai.info"
                if preview_path is not None:
                    preview_path = target_dir / f"{final_stem}.preview{preview_path.suffix}"

        json_result = None
        if auto_sync_triggers and info_path.is_file():
            json_result = sync_single_trigger_json(info_path)

        return {
            "model_path": model_path,
            "info_path": info_path,
            "preview_path": preview_path,
            "json_path": json_result["json_path"] if json_result else None,
            "activation_text": json_result["activation_text"] if json_result else "",
            "renamed_from": base_stem,
            "renamed_to": final_stem,
        }


def _load_hash_cache(cache_path: Path | None) -> set[str]:
    if cache_path is None or not cache_path.is_file():
        return set()
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return {str(x).upper() for x in data}
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return set()


def _save_hash_cache(cache_path: Path | None, hashes: set[str]) -> None:
    if cache_path is None:
        return
    write_json_atomic(cache_path, sorted(hashes))


def backfill_missing_info(
    directory: Path,
    client: CivitaiClient,
    recursive: bool = True,
    limit: int = 20,
    download_preview: bool = True,
    sync_triggers: bool = True,
    not_found_cache_path: Path | None = None,
    delay_sec: float = 0.5,
) -> dict[str, Any]:
    """Compute SHA-256 on models missing .civitai.info and fetch metadata from Civitai."""
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在：{directory}")

    not_found = _load_hash_cache(not_found_cache_path)
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    candidates: list[Path] = []
    for p in sorted(iterator, key=lambda x: str(x).casefold()):
        if not p.is_file():
            continue
        low = p.name.lower()
        if not low.endswith(MODEL_BINARY_EXTS):
            continue
        if ".vae." in low or p.stem.lower().endswith(".vae"):
            continue
        if (p.parent / f"{p.stem}.civitai.info").exists():
            continue
        candidates.append(p)

    matched = 0
    skipped_cached = 0
    not_found_count = 0
    triggers_synced = 0
    previews_downloaded = 0
    errors: list[str] = []
    details: list[str] = []
    queried = 0

    for model_path in candidates:
        if queried >= max(1, int(limit)):
            break
        try:
            sha = compute_sha256(model_path)
            if sha in not_found:
                skipped_cached += 1
                continue
            queried += 1
            version_data = client.get_model_version_by_hash(sha)
            if version_data is None:
                not_found.add(sha)
                not_found_count += 1
                details.append(f"[未收录] {model_path.name}")
            else:
                model_data = None
                mid = version_data.get("modelId")
                if mid and not version_data.get("model"):
                    try:
                        model_data = client.get_model(str(mid))
                    except Exception:
                        model_data = None
                info_payload = client.build_civitai_info(model_data, version_data)
                info_path = model_path.parent / f"{model_path.stem}.civitai.info"
                write_json_atomic(info_path, info_payload)
                matched += 1

                if download_preview and not any(
                    (model_path.parent / f"{model_path.stem}.preview{e}").exists() for e in PREVIEW_EXTS
                ):
                    img_meta = client.pick_preview_image(version_data)
                    if img_meta and img_meta.get("url"):
                        img_url = str(img_meta["url"])
                        ext = Path(urllib.parse.urlsplit(img_url).path).suffix.lower()
                        if ext not in PREVIEW_EXTS:
                            ext = ".png"
                        try:
                            client.download_file(img_url, model_path.parent / f"{model_path.stem}.preview{ext}", rewrite_host=False)
                            previews_downloaded += 1
                        except Exception:
                            pass

                if sync_triggers:
                    res = sync_single_trigger_json(info_path)
                    if res is not None:
                        triggers_synced += 1
                details.append(f"[已补全] {model_path.name}")

            if delay_sec > 0 and queried < len(candidates):
                time.sleep(delay_sec)
        except Exception as exc:
            errors.append(f"{model_path.name}: {exc}")

    _save_hash_cache(not_found_cache_path, not_found)
    return {
        "total_missing": len(candidates),
        "queried": queried,
        "matched": matched,
        "not_found": not_found_count,
        "skipped_cached": skipped_cached,
        "previews_downloaded": previews_downloaded,
        "triggers_synced": triggers_synced,
        "details": details,
        "errors": errors,
    }
