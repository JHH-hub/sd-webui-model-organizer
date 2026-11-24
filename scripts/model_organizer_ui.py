"""SD WebUI / Forge / Forge Neo extension entrypoint for Model Organizer."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gradio as gr
from modules import script_callbacks, shared

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_organizer import (  # noqa: E402
    MODEL_TYPE_SPECS,
    CivitaiClient,
    CivitaiConfig,
    RenameHistoryStore,
    apply_preview_normalizations,
    apply_trigger_updates,
    backfill_missing_info,
    execute_atomic_moves,
    list_subfolders,
    logical_abspath,
    normalize_domain,
    parse_civitai_url,
    plan_renames,
    quarantine_orphans,
    resolve_model_dirs,
    safe_resolve_subdir,
    sanitize_filename,
    scan_orphans,
    scan_rename_candidates,
)

DATA_DIR = ROOT / "data"
HISTORY_PATH = DATA_DIR / "rename_history.json"
HASH_CACHE_PATH = DATA_DIR / "not_found_hashes.json"
QUARANTINE_DIR = DATA_DIR / "quarantine"
ROOT_CHOICE = "(根目录)"
TYPE_CHOICES = [(spec["label"], key) for key, spec in MODEL_TYPE_SPECS.items()]
RENAME_TYPE_CHOICES = TYPE_CHOICES + [("自定义目录", "custom")]
RENAME_HEADERS = ["启用", "子目录", "当前基名", "类型", "C站模型名", "版本", "新名称（双击可改）"]


def _webui_root() -> Path:
    try:
        from modules import paths as webui_paths  # type: ignore

        p = getattr(webui_paths, "script_path", None)
        if p:
            return logical_abspath(p)
    except Exception:
        pass
    return logical_abspath(ROOT.parent.parent) if ROOT.parent.name.lower() == "extensions" else logical_abspath(Path.cwd())


def _model_dirs() -> dict[str, Path]:
    return resolve_model_dirs(_webui_root(), getattr(shared, "cmd_opts", None))


def _opts_get(key: str, default: Any = "") -> Any:
    data = getattr(shared, "opts", None)
    store = getattr(data, "data", None) if data is not None else None
    if isinstance(store, dict) and key in store:
        return store[key]
    return default


def _subfolder_choices(model_type: str) -> list[str]:
    if model_type == "custom":
        return [ROOT_CHOICE]
    dirs = _model_dirs()
    base = dirs.get(model_type)
    if base is None:
        return [ROOT_CHOICE]
    subs = [s for s in list_subfolders(base) if s]
    return [ROOT_CHOICE] + subs


def _resolve_target_dir(model_type: str, subfolder_choice: str, custom_dir: str = "") -> Path:
    if model_type == "custom":
        raw = (custom_dir or "").strip()
        if not raw:
            raise ValueError("请先填写自定义目录路径")
        p = logical_abspath(raw)
        if not p.is_dir():
            raise FileNotFoundError(f"自定义目录不存在：{p}")
        return p

    base = _model_dirs()[model_type]
    rel = "" if not subfolder_choice or subfolder_choice == ROOT_CHOICE else subfolder_choice
    target = safe_resolve_subdir(base, rel)
    return target


def _make_client(domain: str | None = None, proxy_override: str | None = None, skip_nsfw: bool | None = None) -> CivitaiClient:
    eff_domain = normalize_domain(domain or _opts_get("mo_civitai_domain", "civitai.com"))
    eff_proxy = (proxy_override if (proxy_override is not None and proxy_override.strip()) else str(_opts_get("mo_proxy", "") or "")).strip()
    eff_key = str(_opts_get("mo_civitai_api_key", "") or "").strip()
    eff_nsfw = bool(_opts_get("mo_skip_nsfw_preview", False)) if skip_nsfw is None else bool(skip_nsfw)
    return CivitaiClient(
        CivitaiConfig(
            domain=eff_domain,
            api_key=eff_key,
            proxy=eff_proxy,
            skip_nsfw_preview=eff_nsfw,
        )
    )


def _normalize_table_rows(raw_val: Any) -> list[list[Any]]:
    """Handle Gradio 3/4 Dataframe inputs consistently (list, dict, or pandas DataFrame)."""
    if raw_val is None:
        return []
    if isinstance(raw_val, dict) and "data" in raw_val:
        raw_val = raw_val["data"]
    elif hasattr(raw_val, "values") and hasattr(raw_val.values, "tolist"):
        raw_val = raw_val.values.tolist()
    if not isinstance(raw_val, (list, tuple)):
        return []
    return [list(r) for r in raw_val]


def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    return str(val or "").strip().lower() in ("true", "1", "yes", "y", "是")


def _undo_choices() -> tuple[list[tuple[str, str]], str | None]:
    batches = RenameHistoryStore(HISTORY_PATH).list_batches(include_reverted=False)
    choices = [
        (f"{b['timestamp']} | {b.get('summary', '')} ({b['batch_id']})", b["batch_id"])
        for b in batches
    ]
    return choices, (choices[0][1] if choices else None)


def _type_from_civitai_model(model_type: str) -> str:
    low = (model_type or "").lower()
    if low == "checkpoint":
        return "checkpoint"
    if low in ("textualinversion", "embedding"):
        return "embedding"
    if low == "locon":
        return "lycoris" if (_model_dirs()["lycoris"].is_dir()) else "lora"
    return "lora"


def _refresh_webui_models() -> str:
    notes: list[str] = []
    try:
        from modules import sd_models  # type: ignore

        if hasattr(sd_models, "list_models"):
            sd_models.list_models()
            notes.append("Checkpoint 列表已刷新")
    except Exception:
        pass
    try:
        from modules import ui_extra_networks  # type: ignore

        if hasattr(ui_extra_networks, "refresh"):
            ui_extra_networks.refresh()
            notes.append("Extra Networks 缓存已刷新")
    except Exception:
        pass
    if not notes:
        return "✅ 刷新请求已触发（若面板未立刻更新，请点击 WebUI 卡片区右上角「刷新」按钮）。"
    return "✅ " + "、".join(notes) + "。"


def on_ui_settings() -> None:
    section = ("model_organizer", "Model Organizer")
    shared.opts.add_option(
        "mo_civitai_domain",
        shared.OptionInfo(
            "civitai.com",
            "Civitai 域名切换（无需修改原 Civitai Helper 配置）",
            gr.Radio,
            {"interactive": True, "choices": ["civitai.com", "civitai.red"]},
            section=section,
        ),
    )
    shared.opts.add_option(
        "mo_civitai_api_key",
        shared.OptionInfo(
            "",
            "Civitai API Key（需登录下载的模型必填）",
            gr.Textbox,
            {"interactive": True, "lines": 1},
            section=section,
        ),
    )
    shared.opts.add_option(
        "mo_proxy",
        shared.OptionInfo(
            "",
            "HTTP / SOCKS5 代理地址（例如 socks5h://127.0.0.1:7890）",
            gr.Textbox,
            {"interactive": True, "lines": 1},
            section=section,
        ),
    )
    shared.opts.add_option(
        "mo_skip_nsfw_preview",
        shared.OptionInfo(
            False,
            "下载或补全预览图时跳过 NSFW 标记图片",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "mo_auto_rename_on_download",
        shared.OptionInfo(
            True,
            "C站下载完成后自动按规范名重命名",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "mo_auto_sync_triggers_on_download",
        shared.OptionInfo(
            True,
            "C站下载完成后自动写入触发词 .json",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )


# ── UI Handler Functions (testable standalone without Gradio browser server) ──

def handle_parse_civitai(url_or_id: str, domain: str, proxy_override: str):
    try:
        model_id, parsed_ver = parse_civitai_url(url_or_id)
        client = _make_client(domain=domain, proxy_override=proxy_override)
        model_data = client.get_model(model_id)
        versions = model_data.get("modelVersions") or []
        if not versions:
            raise ValueError(f"模型 #{model_id} 没有可用版本")

        ver_choices: list[tuple[str, str]] = []
        selected_ver_id: str | None = None
        for v in versions:
            vid = str(v.get("id") or "")
            vname = str(v.get("name") or vid)
            base_m = str(v.get("baseModel") or "")
            label = f"{vname} [{base_m}] (#{vid})" if base_m else f"{vname} (#{vid})"
            ver_choices.append((label, vid))
            if parsed_ver and vid == parsed_ver:
                selected_ver_id = vid
        if selected_ver_id is None:
            selected_ver_id = ver_choices[0][1]

        guessed_type = _type_from_civitai_model(str(model_data.get("type") or ""))
        sub_choices = _subfolder_choices(guessed_type)
        summary = (
            f"### {model_data.get('name', 'Unknown')} (`#{model_id}`)\n"
            f"- **类型**：`{model_data.get('type', 'Unknown')}` → 推荐入库目录：`{MODEL_TYPE_SPECS[guessed_type]['label']}`\n"
            f"- **可用版本数**：{len(ver_choices)} 个"
        )
        return (
            model_data,
            summary,
            gr.update(choices=ver_choices, value=selected_ver_id),
            gr.update(value=guessed_type),
            gr.update(choices=sub_choices, value=ROOT_CHOICE),
            "已就绪，确认子目录后点击「开始下载并自动入库」。",
        )
    except Exception as exc:
        return (
            {},
            f"⚠️ 解析失败：{exc}",
            gr.update(choices=[], value=None),
            gr.update(),
            gr.update(),
            f"❌ 解析失败：{exc}",
        )


def handle_dl_type_change(model_type: str):
    choices = _subfolder_choices(model_type)
    return gr.update(choices=choices, value=ROOT_CHOICE)


def handle_download_and_organize(
    model_state: dict[str, Any],
    url_or_id: str,
    domain: str,
    proxy_override: str,
    version_id: str | None,
    target_type: str,
    subfolder_choice: str,
    new_subfolder: str,
    auto_rename: bool,
    auto_sync_triggers: bool,
    skip_nsfw: bool,
):
    try:
        client = _make_client(domain=domain, proxy_override=proxy_override, skip_nsfw=skip_nsfw)
        model_data = model_state if isinstance(model_state, dict) and model_state.get("id") else None
        parsed_ver = None
        if model_data is None:
            model_id, parsed_ver = parse_civitai_url(url_or_id)
            model_data = client.get_model(model_id)

        versions = model_data.get("modelVersions") or []
        chosen_vid = str(version_id or parsed_ver or "")
        version_data = None
        if chosen_vid:
            for v in versions:
                if str(v.get("id")) == chosen_vid:
                    version_data = v
                    break
            if version_data is None:
                version_data = client.get_model_version(chosen_vid)
        elif versions:
            version_data = versions[0]
        if not version_data:
            raise ValueError("未找到可用的模型版本")

        target_dir = _resolve_target_dir(target_type, subfolder_choice)
        extra_sub = sanitize_filename((new_subfolder or "").strip())
        if extra_sub:
            target_dir = safe_resolve_subdir(target_dir, extra_sub)
        target_dir.mkdir(parents=True, exist_ok=True)

        res = client.download_model_version(
            model_data,
            version_data,
            target_dir,
            auto_rename=bool(auto_rename),
            auto_sync_triggers=bool(auto_sync_triggers),
            include_checkpoint_version=True,
        )
        lines = [
            f"✅ 下载并入库完成！",
            f"- **模型文件**：`{res['model_path']}`",
            f"- **Info 文件**：`{res['info_path'].name}`",
        ]
        if res["preview_path"]:
            lines.append(f"- **预览图**：`{res['preview_path'].name}`")
        if res["renamed_from"] != res["renamed_to"]:
            lines.append(f"- **已自动改名**：`{res['renamed_from']}` → `{res['renamed_to']}`")
        if res["activation_text"]:
            lines.append(f"- **已同步触发词**：`{res['activation_text']}`")
        return "\n".join(lines), gr.update(choices=_subfolder_choices(target_type), value=ROOT_CHOICE)
    except Exception as exc:
        return f"❌ 下载失败：{exc}", gr.update()


def handle_rename_type_change(model_type: str):
    is_custom = model_type == "custom"
    choices = _subfolder_choices(model_type)
    return gr.update(visible=is_custom), gr.update(choices=choices, value=ROOT_CHOICE, visible=not is_custom)


def handle_scan_rename(
    model_type: str,
    subfolder_choice: str,
    custom_dir: str,
    recursive: bool,
    include_ckpt_ver: bool,
    only_changed: bool,
):
    try:
        target_dir = _resolve_target_dir(model_type, subfolder_choice, custom_dir)
        if not target_dir.is_dir():
            return [], [], f"⚠️ 目录不存在：`{target_dir}`"
        records = scan_rename_candidates(
            target_dir,
            recursive=bool(recursive),
            include_checkpoint_version=bool(include_ckpt_ver),
        )
        rows: list[list[Any]] = []
        shown = []
        for r in records:
            if only_changed and not r.get("needs_rename") and not r.get("error"):
                continue
            shown.append({
                "parent_dir": str(r["parent_dir"]),
                "rel_dir": r["rel_dir"],
                "base": r["base"],
                "error": r.get("error", ""),
            })
            rows.append([
                bool(r.get("needs_rename") and not r.get("error")),
                r["rel_dir"],
                r["base"],
                "解析失败" if r.get("error") else (r.get("model_type") or "未知"),
                r.get("model_name") or "",
                r.get("version_name") or "",
                r.get("suggested_name") or r["base"],
            ])
        total_need = sum(1 for r in records if r.get("needs_rename"))
        msg = f"📂 扫描目录：`{target_dir}` | 总信息文件：{len(records)} | 建议改名：{total_need} | 当前表格显示：{len(rows)} 行（双击最右列可编辑新名称）"
        return rows, shown, msg
    except Exception as exc:
        return [], [], f"❌ 扫描失败：{exc}"


def handle_execute_rename(
    table_value: Any,
    scan_meta: list[dict[str, Any]],
    model_type: str,
    subfolder_choice: str,
    custom_dir: str,
):
    try:
        rows = _normalize_table_rows(table_value)
        if not rows:
            return "请先点击「扫描待更名列表」。", gr.update()

        target_dir = _resolve_target_dir(model_type, subfolder_choice, custom_dir)
        meta_by_key = {(m.get("rel_dir", "."), m.get("base", "")): m for m in (scan_meta or [])}
        items: list[dict[str, Any]] = []
        for r in rows:
            if len(r) < 7:
                continue
            enabled = _to_bool(r[0])
            if not enabled:
                continue
            rel_dir = str(r[1] or ".")
            base = str(r[2] or "")
            new_name = str(r[6] or "")
            meta = meta_by_key.get((rel_dir, base))
            parent_dir = (
                Path(meta["parent_dir"])
                if meta and meta.get("parent_dir")
                else (target_dir if rel_dir in ("", ".") else safe_resolve_subdir(target_dir, rel_dir))
            )
            items.append({
                "enabled": True,
                "parent_dir": parent_dir,
                "base": base,
                "new_name": new_name,
                "error": meta.get("error", "") if meta else "",
            })

        plan, errors = plan_renames(items)
        if errors:
            return "❌ 命名校验未通过（未改动任何文件）：\n" + "\n".join(f"- {e}" for e in errors[:20]), gr.update()
        if not plan:
            return "没有检测到需要更改的文件名。", gr.update()

        execute_atomic_moves(plan)
        batch = RenameHistoryStore(HISTORY_PATH).record(
            target_dir, plan, summary=f"更名 {len(plan)} 个文件"
        )
        choices, latest = _undo_choices()
        return (
            f"✅ 批量更名完成！共安全重命名 **{len(plan)}** 个文件。\n"
            f"- 撤销批次 ID：`{batch['batch_id']}`（若需恢复原名，可直接在下方选择批次一键撤销）",
            gr.update(choices=choices, value=latest),
        )
    except Exception as exc:
        return f"❌ 更名失败（已尝试自动回滚）：{exc}", gr.update()


def handle_undo_rename(batch_id: str | None):
    try:
        store = RenameHistoryStore(HISTORY_PATH)
        res = store.undo_batch(batch_id or None)
        choices, latest = _undo_choices()
        return (
            f"✅ 已成功撤销批次 `{res['batch_id']}`，共还原 **{res['restored']}** 个文件！",
            gr.update(choices=choices, value=latest),
        )
    except Exception as exc:
        return f"❌ 撤销失败：{exc}", gr.update()


def handle_maint_triggers(model_type: str, subfolder_choice: str, recursive: bool, dry_run: bool) -> str:
    try:
        target_dir = _resolve_target_dir(model_type, subfolder_choice)
        res = apply_trigger_updates(target_dir, recursive=bool(recursive), dry_run=bool(dry_run))
        tag = "【预检结果（未写入）】" if dry_run else "【执行完成】"
        lines = [
            f"### {tag} 触发词 JSON 补全",
            f"- 目标目录：`{target_dir}`",
            f"- 补全空白 JSON：{res['fill']} 个 | 新建 JSON：{res['create']} 个 | 错误：{len(res['errors'])} 个",
        ]
        for item in res["items"][:12]:
            lines.append(f"  - `{item['action']}` `{Path(item['json_path']).name}` → `{item['activation_text'][:70]}`")
        if len(res["items"]) > 12:
            lines.append(f"  - ……另有 {len(res['items']) - 12} 项")
        for err in res["errors"][:5]:
            lines.append(f"- ⚠️ {err}")
        return "\n".join(lines)
    except Exception as exc:
        return f"❌ 操作失败：{exc}"


def handle_maint_previews(model_type: str, subfolder_choice: str, recursive: bool, mode: str, dry_run: bool) -> str:
    try:
        target_dir = _resolve_target_dir(model_type, subfolder_choice)
        res = apply_preview_normalizations(
            target_dir, recursive=bool(recursive), mode=mode, dry_run=bool(dry_run)
        )
        tag = "【预检结果（未改动）】" if dry_run else "【执行完成】"
        lines = [
            f"### {tag} 预览图规范化（`{mode}`）",
            f"- 已有 `.preview.*`（跳过）：{res['already_ok']} 个",
            f"- 待处理/已处理：{res['processed']} 个 | 错误：{len(res['errors'])} 个",
        ]
        for item in res["items"][:12]:
            lines.append(f"  - `{item['src'].name}` → `{item['dst'].name}`")
        if len(res["items"]) > 12:
            lines.append(f"  - ……另有 {len(res['items']) - 12} 张")
        for err in res["errors"][:5]:
            lines.append(f"- ⚠️ {err}")
        return "\n".join(lines)
    except Exception as exc:
        return f"❌ 操作失败：{exc}"


def handle_maint_hash_backfill(
    model_type: str,
    subfolder_choice: str,
    recursive: bool,
    domain: str,
    limit: int,
    dl_preview: bool,
    sync_trig: bool,
) -> str:
    try:
        target_dir = _resolve_target_dir(model_type, subfolder_choice)
        client = _make_client(domain=domain)
        res = backfill_missing_info(
            target_dir,
            client,
            recursive=bool(recursive),
            limit=int(limit),
            download_preview=bool(dl_preview),
            sync_triggers=bool(sync_trig),
            not_found_cache_path=HASH_CACHE_PATH,
        )
        lines = [
            "### ✅ SHA-256 补全执行结果",
            f"- 缺失 `.civitai.info` 总数：{res['total_missing']} | 本次实际查询：{res['queried']} 条",
            f"- 命中并写入 `{res['matched']}` 个 | 未收录跳过缓存：{res['not_found']} 个 | 历史未收录跳过：{res['skipped_cached']} 个",
            f"- 下载封面：{res['previews_downloaded']} 张 | 补全触发词：{res['triggers_synced']} 个",
        ]
        for d in res["details"][:20]:
            lines.append(f"  - {d}")
        for err in res["errors"][:5]:
            lines.append(f"- ⚠️ {err}")
        return "\n".join(lines)
    except Exception as exc:
        return f"❌ Hash 补全失败：{exc}"


def handle_maint_orphans(model_type: str, subfolder_choice: str, recursive: bool, move_to_quarantine: bool) -> str:
    try:
        target_dir = _resolve_target_dir(model_type, subfolder_choice)
        orphans = scan_orphans(target_dir, recursive=bool(recursive))
        if not orphans:
            return f"✅ 目录 `{target_dir}` 未发现无主伴随文件。"
        if move_to_quarantine:
            out = quarantine_orphans(orphans, target_dir, QUARANTINE_DIR)
            return (
                f"### ✅ 已安全隔离 {out['moved']} 个孤儿伴随文件\n"
                f"- 隔离目录：`{out['quarantine_dir']}`（非永久删除，随时可取回）"
            )
        lines = [
            f"### 🔍 扫描到 {len(orphans)} 个孤儿伴随文件（仅列出，未移动）",
        ]
        for p in orphans[:25]:
            lines.append(f"- `{p}`")
        if len(orphans) > 25:
            lines.append(f"- ……另有 {len(orphans) - 25} 个")
        return "\n".join(lines)
    except Exception as exc:
        return f"❌ 孤儿扫描失败：{exc}"


def on_ui_tabs():
    default_domain = normalize_domain(str(_opts_get("mo_civitai_domain", "civitai.com")))
    default_auto_rename = bool(_opts_get("mo_auto_rename_on_download", True))
    default_auto_trig = bool(_opts_get("mo_auto_sync_triggers_on_download", True))
    default_skip_nsfw = bool(_opts_get("mo_skip_nsfw_preview", False))
    lora_subs = _subfolder_choices("lora")
    undo_choices, undo_default = _undo_choices()

    with gr.Blocks(analytics_enabled=False) as ui_block:
        gr.Markdown(
            "## 🗂️ Model Organizer · 模型批量重命名与整理助手\n"
            "不改动内置 Extra Networks 卡片，与 Civitai Helper 零冲突共存；支持批量规范更名、触发词/预览图/SHA-256 联网补全及 C 站 `.com / .red` 下载。"
        )

        # ── Tab 1: 批量规范重命名 ──
        with gr.Tab("1. 批量规范重命名"):
            rename_meta_state = gr.State([])
            with gr.Row():
                rn_type = gr.Dropdown(choices=RENAME_TYPE_CHOICES, value="lora", label="目标目录类型", scale=2)
                rn_subfolder = gr.Dropdown(choices=lora_subs, value=ROOT_CHOICE, label="子目录", scale=2)
                rn_custom_dir = gr.Textbox(
                    label="自定义目录绝对路径（支持 ComfyUI 模型目录）",
                    placeholder=r"D:\ComfyUI\models\loras",
                    visible=False,
                    scale=3,
                )
            with gr.Row():
                rn_recursive = gr.Checkbox(value=False, label="递归扫描所有子目录")
                rn_ckpt_ver = gr.Checkbox(value=True, label="Checkpoint 自动保留版本名")
                rn_only_changed = gr.Checkbox(value=True, label="仅显示需要更名的条目")
                rn_scan_btn = gr.Button("🔎 扫描待更名列表", variant="secondary")

            rn_scan_msg = gr.Markdown("点击「扫描待更名列表」预览将进行的改名。")
            rn_table = gr.Dataframe(
                headers=RENAME_HEADERS,
                datatype=["bool", "str", "str", "str", "str", "str", "str"],
                type="array",
                interactive=True,
                wrap=True,
                label="待更名列表（可取消勾选左侧复选框，或双击最右列自定义新文件名）",
            )

            with gr.Row():
                rn_run_btn = gr.Button("✅ 执行批量更名", variant="primary", scale=2)
                rn_undo_dd = gr.Dropdown(
                    choices=undo_choices,
                    value=undo_default,
                    label="历史更名批次（一键撤销还原）",
                    scale=4,
                )
                rn_undo_btn = gr.Button("↩ 一键撤销所选批次", variant="secondary", scale=2)

            rn_log = gr.Markdown("")

        # ── Tab 2: 一键整理体检 ──
        with gr.Tab("2. 一键整理体检"):
            with gr.Row():
                mt_type = gr.Dropdown(choices=TYPE_CHOICES, value="lora", label="模型目录", scale=2)
                mt_subfolder = gr.Dropdown(choices=lora_subs, value=ROOT_CHOICE, label="子目录", scale=2)
                mt_recursive = gr.Checkbox(value=True, label="递归处理全部子目录", scale=1)

            with gr.Accordion("① 递归补全触发词 .json（永不覆盖已有手写提示词）", open=True):
                with gr.Row():
                    mt_trig_dry = gr.Button("🔍 预检可补全数量")
                    mt_trig_run = gr.Button("⚡ 一键补全触发词", variant="primary")
                mt_trig_out = gr.Markdown("")

            with gr.Accordion("② 预览图规范化（.preview.<真实扩展名> 或安全转码 PNG）", open=True):
                with gr.Row():
                    mt_prev_mode = gr.Radio(
                        choices=[("保留原图真实扩展名（推荐，零伪 PNG）", "keep_ext"), ("转码为真 PNG 格式 (.preview.png)", "convert_png")],
                        value="keep_ext",
                        label="命名与格式模式",
                        scale=3,
                    )
                    mt_prev_dry = gr.Button("🔍 预检预览图", scale=1)
                    mt_prev_run = gr.Button("🖼️ 执行规范化", variant="primary", scale=1)
                mt_prev_out = gr.Markdown("")

            with gr.Accordion("③ SHA-256 联网补全缺失 .civitai.info 与封面", open=False):
                with gr.Row():
                    mt_hash_domain = gr.Radio(choices=["civitai.com", "civitai.red"], value=default_domain, label="查询域名", scale=2)
                    mt_hash_limit = gr.Slider(minimum=1, maximum=100, step=1, value=20, label="单次最多查询条数", scale=2)
                    mt_hash_prev = gr.Checkbox(value=True, label="同时下载预览图", scale=1)
                    mt_hash_trig = gr.Checkbox(value=True, label="同时写入触发词", scale=1)
                    mt_hash_btn = gr.Button("🌐 开始 Hash 补全", variant="primary", scale=1)
                mt_hash_out = gr.Markdown("")

            with gr.Accordion("④ 孤儿伴随文件清理（仅移入扩展内 quarantine 隔离，不删除）", open=False):
                with gr.Row():
                    mt_orph_scan = gr.Button("🔍 仅扫描孤儿文件")
                    mt_orph_move = gr.Button("📦 安全移入隔离目录", variant="secondary")
                mt_orph_out = gr.Markdown("")

        # ── Tab 3: C站下载与自动入库 ──
        with gr.Tab("3. C站下载与自动入库"):
            parsed_model_state = gr.State({})
            with gr.Row():
                dl_url = gr.Textbox(
                    label="Civitai 模型链接或数字 ID",
                    placeholder="https://civitai.com/models/123456?modelVersionId=789012 或 civitai.red 链接",
                    scale=4,
                )
                dl_domain = gr.Radio(
                    choices=["civitai.com", "civitai.red"],
                    value=default_domain,
                    label="站点域名",
                    scale=2,
                )
            with gr.Row():
                dl_proxy = gr.Textbox(
                    label="临时代理覆盖（留空则使用 Settings 中的代理设置）",
                    placeholder="socks5h://127.0.0.1:7890 或 http://127.0.0.1:7890",
                    scale=4,
                )
                dl_parse_btn = gr.Button("🔍 解析模型信息", variant="secondary", scale=1)

            dl_summary = gr.Markdown("输入模型链接并点击「解析模型信息」，也可直接选择类型与子目录后下载。")

            with gr.Row():
                dl_version = gr.Dropdown(choices=[], value=None, label="选择模型版本", scale=3)
                dl_type = gr.Dropdown(choices=TYPE_CHOICES, value="lora", label="保存模型类型", scale=2)
                dl_subfolder = gr.Dropdown(choices=lora_subs, value=ROOT_CHOICE, label="目标子目录", scale=2)
                dl_new_sub = gr.Textbox(label="或新建子目录（可选）", placeholder="如：画风 / 角色", scale=2)

            with gr.Row():
                dl_auto_rename = gr.Checkbox(value=default_auto_rename, label="下载后自动按 C 站模型名规范重命名")
                dl_auto_trig = gr.Checkbox(value=default_auto_trig, label="下载后自动生成触发词 .json")
                dl_skip_nsfw = gr.Checkbox(value=default_skip_nsfw, label="跳过 NSFW 预览图")
                dl_start_btn = gr.Button("⬇ 开始下载并自动入库", variant="primary")

            dl_log = gr.Markdown("")

        with gr.Row():
            refresh_btn = gr.Button("🔄 刷新 WebUI 模型列表", variant="secondary", scale=1)
            refresh_out = gr.Markdown(
                "开源工具：Model Organizer v1.0 · MIT License · [像素离散 Pxlsan](https://pxlsan.cn)",
            )

        # ── Event Bindings ──
        dl_parse_btn.click(
            fn=handle_parse_civitai,
            inputs=[dl_url, dl_domain, dl_proxy],
            outputs=[parsed_model_state, dl_summary, dl_version, dl_type, dl_subfolder, dl_log],
        )
        dl_type.change(fn=handle_dl_type_change, inputs=[dl_type], outputs=[dl_subfolder])
        dl_start_btn.click(
            fn=handle_download_and_organize,
            inputs=[
                parsed_model_state,
                dl_url,
                dl_domain,
                dl_proxy,
                dl_version,
                dl_type,
                dl_subfolder,
                dl_new_sub,
                dl_auto_rename,
                dl_auto_trig,
                dl_skip_nsfw,
            ],
            outputs=[dl_log, dl_subfolder],
        )

        rn_type.change(fn=handle_rename_type_change, inputs=[rn_type], outputs=[rn_custom_dir, rn_subfolder])
        rn_scan_btn.click(
            fn=handle_scan_rename,
            inputs=[rn_type, rn_subfolder, rn_custom_dir, rn_recursive, rn_ckpt_ver, rn_only_changed],
            outputs=[rn_table, rename_meta_state, rn_scan_msg],
        )
        rn_run_btn.click(
            fn=handle_execute_rename,
            inputs=[rn_table, rename_meta_state, rn_type, rn_subfolder, rn_custom_dir],
            outputs=[rn_log, rn_undo_dd],
        )
        rn_undo_btn.click(fn=handle_undo_rename, inputs=[rn_undo_dd], outputs=[rn_log, rn_undo_dd])

        mt_type.change(fn=handle_dl_type_change, inputs=[mt_type], outputs=[mt_subfolder])
        mt_trig_dry.click(
            fn=lambda t, s, r: handle_maint_triggers(t, s, r, True),
            inputs=[mt_type, mt_subfolder, mt_recursive],
            outputs=[mt_trig_out],
        )
        mt_trig_run.click(
            fn=lambda t, s, r: handle_maint_triggers(t, s, r, False),
            inputs=[mt_type, mt_subfolder, mt_recursive],
            outputs=[mt_trig_out],
        )
        mt_prev_dry.click(
            fn=lambda t, s, r, m: handle_maint_previews(t, s, r, m, True),
            inputs=[mt_type, mt_subfolder, mt_recursive, mt_prev_mode],
            outputs=[mt_prev_out],
        )
        mt_prev_run.click(
            fn=lambda t, s, r, m: handle_maint_previews(t, s, r, m, False),
            inputs=[mt_type, mt_subfolder, mt_recursive, mt_prev_mode],
            outputs=[mt_prev_out],
        )
        mt_hash_btn.click(
            fn=handle_maint_hash_backfill,
            inputs=[mt_type, mt_subfolder, mt_recursive, mt_hash_domain, mt_hash_limit, mt_hash_prev, mt_hash_trig],
            outputs=[mt_hash_out],
        )
        mt_orph_scan.click(
            fn=lambda t, s, r: handle_maint_orphans(t, s, r, False),
            inputs=[mt_type, mt_subfolder, mt_recursive],
            outputs=[mt_orph_out],
        )
        mt_orph_move.click(
            fn=lambda t, s, r: handle_maint_orphans(t, s, r, True),
            inputs=[mt_type, mt_subfolder, mt_recursive],
            outputs=[mt_orph_out],
        )
        refresh_btn.click(fn=_refresh_webui_models, inputs=[], outputs=[refresh_out])

    return [(ui_block, "模型整理助手", "model_organizer")]


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
