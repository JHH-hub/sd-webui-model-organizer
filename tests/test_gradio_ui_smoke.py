import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
from PIL import Image

from model_organizer.trigger_sync import write_json_atomic


class _FakeOptionInfo:
    def __init__(self, default, label, component, component_args=None, section=None):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args or {}
        self.section = section


class _FakeOpts:
    def __init__(self):
        self.data = {}
        self.options = {}

    def add_option(self, key, info):
        self.options[key] = info
        self.data.setdefault(key, info.default)


def _install_fake_webui_modules(lora_dir: Path):
    modules_pkg = types.ModuleType("modules")
    shared_mod = types.ModuleType("modules.shared")
    shared_mod.cmd_opts = types.SimpleNamespace(lora_dir=str(lora_dir))
    shared_mod.opts = _FakeOpts()

    cb_mod = types.ModuleType("modules.script_callbacks")
    cb_mod._settings = []
    cb_mod._tabs = []
    cb_mod._app_started = []
    cb_mod.on_ui_settings = lambda fn: cb_mod._settings.append(fn)
    cb_mod.on_ui_tabs = lambda fn: cb_mod._tabs.append(fn)
    cb_mod.on_app_started = lambda fn: cb_mod._app_started.append(fn)

    modules_pkg.shared = shared_mod
    modules_pkg.script_callbacks = cb_mod
    shared_mod.OptionInfo = _FakeOptionInfo
    return {"modules": modules_pkg, "modules.shared": shared_mod, "modules.script_callbacks": cb_mod}


class GradioUISmokeTests(unittest.TestCase):
    def test_ui_registration_and_handlers_work_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            lora_dir = Path(td) / "Lora"
            lora_dir.mkdir()
            (lora_dir / "old_name.safetensors").write_bytes(b"M")
            Image.new("RGB", (16, 16), (255, 0, 0)).save(lora_dir / "old_name.jpg", format="JPEG")
            write_json_atomic(
                lora_dir / "old_name.civitai.info",
                {"model": {"name": "Nice Lora", "type": "LORA"}, "name": "v1", "trainedWords": ["nice_tag"]},
            )
            (lora_dir / "orphan_leftover.json").write_text("{}", encoding="utf-8")

            fakes = _install_fake_webui_modules(lora_dir)
            hist_tmp = Path(td) / "history.json"
            quar_tmp = Path(td) / "quarantine"

            with mock.patch.dict(sys.modules, fakes):
                sys.modules.pop("scripts.model_organizer_ui", None)
                ui = importlib.import_module("scripts.model_organizer_ui")

                with mock.patch.object(ui, "HISTORY_PATH", hist_tmp), mock.patch.object(ui, "QUARANTINE_DIR", quar_tmp):
                    ui.on_ui_settings()
                    opts = fakes["modules.shared"].opts
                    for k in (
                        "mo_civitai_domain",
                        "mo_civitai_api_key",
                        "mo_proxy",
                        "mo_skip_nsfw_preview",
                        "mo_auto_rename_on_download",
                        "mo_auto_sync_triggers_on_download",
                        "mo_enable_card_buttons",
                        "mo_blur_nsfw_gallery",
                    ):
                        self.assertIn(k, opts.options)
                    self.assertEqual(len(fakes["modules.script_callbacks"]._app_started), 1)

                    tabs = ui.on_ui_tabs()
                    self.assertEqual(len(tabs), 1)
                    block, title, elem_id = tabs[0]
                    self.assertIsInstance(block, gr.Blocks)
                    self.assertEqual(title, "模型整理助手")
                    self.assertEqual(elem_id, "model_organizer")
                    tab_labels = [c.label for c in block.blocks.values() if isinstance(c, gr.Tab)]
                    self.assertEqual(tab_labels, ["1. 批量规范重命名", "2. 一键整理体检", "3. C站下载与自动入库"])

                    # Check table normalization compatibility
                    self.assertEqual(ui._normalize_table_rows({"data": [[True, "a"]]}), [[True, "a"]])
                    fake_df = types.SimpleNamespace(values=types.SimpleNamespace(tolist=lambda: [[False, "b"]]))
                    self.assertEqual(ui._normalize_table_rows(fake_df), [[False, "b"]])

                    # Preview normalizer -> Trigger sync -> Orphan quarantine -> Scan & Rename -> Undo
                    prev_out = ui.handle_maint_previews("lora", ui.ROOT_CHOICE, True, "keep_ext", dry_run=False)
                    self.assertIn("待处理/已处理：1", prev_out)
                    self.assertTrue((lora_dir / "old_name.preview.jpg").is_file())

                    trig_out = ui.handle_maint_triggers("lora", ui.ROOT_CHOICE, True, dry_run=False)
                    self.assertIn("新建 JSON：1", trig_out)
                    self.assertTrue((lora_dir / "old_name.json").is_file())

                    orph_out = ui.handle_maint_orphans("lora", ui.ROOT_CHOICE, True, move_to_quarantine=True)
                    self.assertIn("已安全隔离 1 个", orph_out)
                    self.assertFalse((lora_dir / "orphan_leftover.json").exists())

                    rows, meta, msg = ui.handle_scan_rename("lora", ui.ROOT_CHOICE, "", False, True, True)
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0][6], "Nice Lora")
                    self.assertIn("建议改名：1", msg)

                    ren_msg, undo_update = ui.handle_execute_rename(rows, meta, "lora", ui.ROOT_CHOICE, "")
                    self.assertIn("批量更名完成", ren_msg)
                    self.assertTrue((lora_dir / "Nice Lora.safetensors").is_file())
                    self.assertTrue((lora_dir / "Nice Lora.preview.jpg").is_file())
                    self.assertTrue((lora_dir / "Nice Lora.json").is_file())
                    self.assertTrue(undo_update.get("value"))

                    # Undo rename restores base stem across all renamed files
                    undo_msg, _ = ui.handle_undo_rename(undo_update["value"])
                    self.assertIn("成功撤销", undo_msg)
                    self.assertTrue((lora_dir / "old_name.safetensors").is_file())
                    self.assertTrue((lora_dir / "old_name.preview.jpg").is_file())
                    self.assertFalse((lora_dir / "Nice Lora.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
