"""Standalone CLI for Model Organizer (works for SD WebUI, Forge, and ComfyUI model folders)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_organizer import (
    CivitaiClient,
    CivitaiConfig,
    RenameHistoryStore,
    apply_preview_normalizations,
    apply_trigger_updates,
    backfill_missing_info,
    logical_abspath,
    normalize_domain,
    parse_civitai_url,
    plan_renames,
    execute_atomic_moves,
    quarantine_orphans,
    scan_orphans,
    scan_rename_candidates,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def _default_history() -> RenameHistoryStore:
    return RenameHistoryStore(DATA_DIR / "rename_history.json")


def _cmd_scan(args: argparse.Namespace) -> int:
    target = logical_abspath(args.directory)
    records = scan_rename_candidates(
        target,
        recursive=args.recursive,
        include_checkpoint_version=not args.no_checkpoint_version,
    )
    need = [r for r in records if r.get("needs_rename")]
    errs = [r for r in records if r.get("error")]
    print(f"扫描目录: {target}")
    print(f"信息文件总数: {len(records)} | 建议更名: {len(need)} | 解析失败: {len(errs)}")
    for r in need[: args.limit]:
        loc = f"[{r['rel_dir']}] " if r["rel_dir"] != "." else ""
        print(f"  - {loc}{r['base']} -> {r['suggested_name']} ({r['file_count']} files)")
    if len(need) > args.limit:
        print(f"  ... 另有 {len(need) - args.limit} 项未展开")
    for r in errs[:10]:
        print(f"  [ERR] {r['info_path']}: {r['error']}")
    return 0


def _cmd_rename(args: argparse.Namespace) -> int:
    target = logical_abspath(args.directory)
    records = scan_rename_candidates(
        target,
        recursive=args.recursive,
        include_checkpoint_version=not args.no_checkpoint_version,
    )
    items = [
        {
            "parent_dir": r["parent_dir"],
            "base": r["base"],
            "new_name": r["suggested_name"],
            "enabled": bool(r.get("needs_rename")),
            "error": r.get("error", ""),
        }
        for r in records
        if r.get("needs_rename") or r.get("error")
    ]
    plan, errors = plan_renames(items)
    if errors:
        print("发现冲突或错误，已中止：", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    if not plan:
        print("没有需要更名的文件。")
        return 0
    if args.dry_run:
        print(f"[DRY-RUN] 计划更名 {len(plan)} 个文件：")
        for src, dst in plan[:50]:
            print(f"  {src.name} -> {dst.name}")
        return 0

    execute_atomic_moves(plan)
    batch = _default_history().record(target, plan, summary=f"CLI 批量更名 {len(plan)} 个文件")
    print(f"完成！共更名 {len(plan)} 个文件（撤销批次 ID: {batch['batch_id']}）。")
    return 0


def _cmd_undo(args: argparse.Namespace) -> int:
    res = _default_history().undo_batch(args.batch_id)
    print(f"已撤销批次 {res['batch_id']}，恢复 {res['restored']} 个文件。")
    return 0


def _cmd_sync_triggers(args: argparse.Namespace) -> int:
    res = apply_trigger_updates(logical_abspath(args.directory), recursive=args.recursive, dry_run=args.dry_run)
    prefix = "[DRY-RUN] " if args.dry_run else ""
    print(
        f"{prefix}触发词同步完成：补全空白 {res['fill']} 个 | 新建 JSON {res['create']} 个 | 错误 {len(res['errors'])} 个"
    )
    for item in res["items"][:20]:
        print(f"  - [{item['action']}] {item['json_path']}: {item['activation_text'][:80]}")
    for err in res["errors"][:10]:
        print(f"  [ERR] {err}", file=sys.stderr)
    return 1 if res["errors"] else 0


def _cmd_normalize_previews(args: argparse.Namespace) -> int:
    res = apply_preview_normalizations(
        logical_abspath(args.directory),
        recursive=args.recursive,
        mode=args.mode,
        dry_run=args.dry_run,
    )
    prefix = "[DRY-RUN] " if args.dry_run else ""
    print(
        f"{prefix}预览图规范化完成（mode={res['mode']}）：已规范 {res['already_ok']} 个 | 待处理/已处理 {res['processed']} 个 | 错误 {len(res['errors'])} 个"
    )
    for item in res["items"][:20]:
        print(f"  - {item['src'].name} -> {item['dst'].name}")
    for err in res["errors"][:10]:
        print(f"  [ERR] {err}", file=sys.stderr)
    return 1 if res["errors"] else 0


def _cmd_scan_orphans(args: argparse.Namespace) -> int:
    target = logical_abspath(args.directory)
    orphans = scan_orphans(target, recursive=args.recursive)
    print(f"扫描目录: {target}")
    print(f"发现孤儿伴随文件: {len(orphans)} 个")
    for p in orphans[:30]:
        print(f"  - {p}")
    if len(orphans) > 30:
        print(f"  ... 另有 {len(orphans) - 30} 个")
    if args.quarantine and orphans:
        out = quarantine_orphans(orphans, target, logical_abspath(args.quarantine))
        print(f"已安全移入隔离目录 {out['quarantine_dir']}：{out['moved']} 个文件。")
    return 0


def _make_client(args: argparse.Namespace) -> CivitaiClient:
    cfg = CivitaiConfig(
        domain=normalize_domain(args.domain),
        api_key=args.api_key or "",
        proxy=args.proxy or "",
        skip_nsfw_preview=getattr(args, "skip_nsfw", False),
    )
    return CivitaiClient(cfg)


def _cmd_fetch_hash(args: argparse.Namespace) -> int:
    client = _make_client(args)
    res = backfill_missing_info(
        logical_abspath(args.directory),
        client,
        recursive=args.recursive,
        limit=args.limit,
        download_preview=not args.no_preview,
        sync_triggers=not args.no_triggers,
        not_found_cache_path=DATA_DIR / "not_found_hashes.json",
    )
    print(json.dumps({k: v for k, v in res.items() if k != "details"}, ensure_ascii=False, indent=2))
    for line in res["details"][:30]:
        print(" ", line)
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    client = _make_client(args)
    model_id, parsed_ver = parse_civitai_url(args.url_or_id)
    model_data = client.get_model(model_id)
    versions = model_data.get("modelVersions") or []
    if not versions:
        raise RuntimeError(f"模型 #{model_id} 没有可用版本")
    ver_id = args.version_id or parsed_ver
    version_data = None
    if ver_id:
        for v in versions:
            if str(v.get("id")) == str(ver_id):
                version_data = v
                break
        if version_data is None:
            version_data = client.get_model_version(str(ver_id))
    else:
        version_data = versions[0]

    res = client.download_model_version(
        model_data,
        version_data,
        Path(args.target_dir),
        auto_rename=not args.no_rename,
        auto_sync_triggers=not args.no_triggers,
        include_checkpoint_version=not args.no_checkpoint_version,
    )
    print(f"下载入库完成: {res['model_path']}")
    if res["activation_text"]:
        print(f"触发词已同步: {res['activation_text']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model Organizer CLI (SD WebUI / Forge / ComfyUI)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="预览规范重命名建议（只读）")
    p_scan.add_argument("directory")
    p_scan.add_argument("--recursive", action="store_true")
    p_scan.add_argument("--no-checkpoint-version", action="store_true")
    p_scan.add_argument("--limit", type=int, default=50)
    p_scan.set_defaults(func=_cmd_scan)

    p_ren = sub.add_parser("rename", help="按 .civitai.info 批量规范重命名模型与伴随文件")
    p_ren.add_argument("directory")
    p_ren.add_argument("--recursive", action="store_true")
    p_ren.add_argument("--no-checkpoint-version", action="store_true")
    p_ren.add_argument("--dry-run", action="store_true")
    p_ren.set_defaults(func=_cmd_rename)

    p_undo = sub.add_parser("undo", help="撤销最近或指定批次的批量重命名")
    p_undo.add_argument("--batch-id", default=None)
    p_undo.set_defaults(func=_cmd_undo)

    p_trig = sub.add_parser("sync-triggers", help="从 .civitai.info 递归补全 WebUI .json 触发词")
    p_trig.add_argument("directory")
    p_trig.add_argument("--recursive", action="store_true")
    p_trig.add_argument("--dry-run", action="store_true")
    p_trig.set_defaults(func=_cmd_sync_triggers)

    p_prev = sub.add_parser("normalize-previews", help="规范化同名预览图为 .preview.<ext> 或转码 PNG")
    p_prev.add_argument("directory")
    p_prev.add_argument("--recursive", action="store_true")
    p_prev.add_argument("--mode", choices=("keep_ext", "convert_png"), default="keep_ext")
    p_prev.add_argument("--dry-run", action="store_true")
    p_prev.set_defaults(func=_cmd_normalize_previews)

    p_orph = sub.add_parser("scan-orphans", help="扫描无主伴随文件，可选安全移入隔离目录")
    p_orph.add_argument("directory")
    p_orph.add_argument("--recursive", action="store_true")
    p_orph.add_argument("--quarantine", default=None, help="将孤儿文件移入指定隔离目录")
    p_orph.set_defaults(func=_cmd_scan_orphans)

    p_hash = sub.add_parser("fetch-hash", help="通过 SHA-256 为缺失 .civitai.info 的模型补全信息与封面")
    p_hash.add_argument("directory")
    p_hash.add_argument("--domain", default="civitai.com")
    p_hash.add_argument("--proxy", default="")
    p_hash.add_argument("--api-key", default="")
    p_hash.add_argument("--limit", type=int, default=20)
    p_hash.add_argument("--recursive", action="store_true")
    p_hash.add_argument("--no-preview", action="store_true")
    p_hash.add_argument("--no-triggers", action="store_true")
    p_hash.set_defaults(func=_cmd_fetch_hash)

    p_dl = sub.add_parser("download", help="从 Civitai (.com/.red) 下载模型并自动改名/补全触发词")
    p_dl.add_argument("url_or_id")
    p_dl.add_argument("target_dir")
    p_dl.add_argument("--domain", default="civitai.com")
    p_dl.add_argument("--proxy", default="")
    p_dl.add_argument("--api-key", default="")
    p_dl.add_argument("--version-id", default=None)
    p_dl.add_argument("--skip-nsfw", action="store_true")
    p_dl.add_argument("--no-rename", action="store_true")
    p_dl.add_argument("--no-triggers", action="store_true")
    p_dl.add_argument("--no-checkpoint-version", action="store_true")
    p_dl.set_defaults(func=_cmd_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
