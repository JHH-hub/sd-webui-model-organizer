"""SD WebUI / Forge Model Organizer pure-Python core."""

__version__ = "1.0.0"

from .paths import MODEL_TYPE_SPECS, list_subfolders, logical_abspath, resolve_model_dirs, safe_resolve_subdir
from .rename_engine import (
    GROUP_SUFFIXES,
    INFO_SUFFIXES,
    RenameHistoryStore,
    append_versions_for_collisions,
    build_display_name,
    execute_atomic_moves,
    group_files,
    info_base_name,
    load_info,
    plan_renames,
    sanitize_filename,
    scan_rename_candidates,
)
from .trigger_sync import (
    activation_text_from_info,
    apply_trigger_updates,
    collect_trigger_updates,
    sync_single_trigger_json,
    write_json_atomic,
)
from .preview_normalizer import (
    apply_preview_normalizations,
    collect_preview_normalizations,
)
from .orphan_scanner import quarantine_orphans, scan_orphans
from .civitai_client import (
    CivitaiClient,
    CivitaiConfig,
    backfill_missing_info,
    compute_sha256,
    normalize_domain,
    parse_civitai_url,
    rewrite_civitai_host,
)

__all__ = [
    "__version__",
    "MODEL_TYPE_SPECS",
    "logical_abspath",
    "resolve_model_dirs",
    "list_subfolders",
    "safe_resolve_subdir",
    "INFO_SUFFIXES",
    "GROUP_SUFFIXES",
    "sanitize_filename",
    "info_base_name",
    "build_display_name",
    "load_info",
    "group_files",
    "append_versions_for_collisions",
    "scan_rename_candidates",
    "plan_renames",
    "execute_atomic_moves",
    "RenameHistoryStore",
    "activation_text_from_info",
    "write_json_atomic",
    "collect_trigger_updates",
    "apply_trigger_updates",
    "sync_single_trigger_json",
    "collect_preview_normalizations",
    "apply_preview_normalizations",
    "scan_orphans",
    "quarantine_orphans",
    "CivitaiConfig",
    "CivitaiClient",
    "normalize_domain",
    "parse_civitai_url",
    "rewrite_civitai_host",
    "compute_sha256",
    "backfill_missing_info",
]
