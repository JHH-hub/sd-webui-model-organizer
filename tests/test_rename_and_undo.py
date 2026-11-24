import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_organizer.paths import logical_abspath, safe_resolve_subdir

from model_organizer.rename_engine import (
    RenameHistoryStore,
    append_versions_for_collisions,
    build_display_name,
    execute_atomic_moves,
    info_base_name,
    plan_renames,
    sanitize_filename,
    scan_rename_candidates,
)
from model_organizer.trigger_sync import write_json_atomic


class RenameAndUndoTests(unittest.TestCase):
    def test_display_name_and_sanitization(self):
        ckpt = {"model": {"name": "MiaoMiao Harem", "type": "Checkpoint"}, "name": "Anima_1.4"}
        lora = {"model": {"name": "Some Style", "type": "LORA"}, "name": "v2.0"}
        dup = {"model": {"name": "Model v3", "type": "Checkpoint"}, "name": "v3"}
        self.assertEqual(build_display_name(ckpt), "MiaoMiao Harem - Anima_1.4")
        self.assertEqual(build_display_name(lora), "Some Style")
        self.assertEqual(build_display_name(dup), "Model v3")
        self.assertEqual(info_base_name("abc.civitai.info"), "abc")
        self.assertEqual(sanitize_filename('a:b?c.'), "abc")

    def test_collision_resolution_scoped_by_parent_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sub = root / "sub"
            sub.mkdir()
            for folder, base, ver in (
                (root, "m1", "Mage"),
                (root, "m2", "Knight"),
                (sub, "m3", "v1"),
            ):
                (folder / f"{base}.safetensors").write_bytes(b"x")
                write_json_atomic(
                    folder / f"{base}.civitai.info",
                    {"model": {"name": "Costume Collection", "type": "LORA"}, "name": ver},
                )
            records = scan_rename_candidates(root, recursive=True)
            by_base = {r["base"]: r["suggested_name"] for r in records}
            self.assertEqual(by_base["m1"], "Costume Collection - Mage")
            self.assertEqual(by_base["m2"], "Costume Collection - Knight")
            self.assertEqual(by_base["m3"], "Costume Collection")

    def test_atomic_rename_and_undo_with_companions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "raw_123.safetensors").write_bytes(b"MODEL")
            (root / "raw_123.preview.png").write_bytes(b"IMG")
            (root / "raw_123.json").write_text('{"activation text": "foo"}', encoding="utf-8")
            write_json_atomic(
                root / "raw_123.civitai.info",
                {"model": {"name": "Clean:Style?", "type": "LORA"}, "name": "v1"},
            )

            records = scan_rename_candidates(root)
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["needs_rename"])
            self.assertEqual(records[0]["suggested_name"], "CleanStyle")
            self.assertEqual(records[0]["file_count"], 4)

            plan, errs = plan_renames([
                {
                    "parent_dir": r["parent_dir"],
                    "base": r["base"],
                    "new_name": r["suggested_name"],
                    "enabled": True,
                }
                for r in records
            ])
            self.assertEqual(errs, [])
            self.assertEqual(len(plan), 4)

            execute_atomic_moves(plan)
            self.assertFalse((root / "raw_123.safetensors").exists())
            self.assertTrue((root / "CleanStyle.safetensors").is_file())
            self.assertTrue((root / "CleanStyle.preview.png").is_file())
            self.assertTrue((root / "CleanStyle.json").is_file())
            self.assertTrue((root / "CleanStyle.civitai.info").is_file())

            hist = RenameHistoryStore(root / "rename_history.json")
            batch = hist.record(root, plan, "test batch")
            self.assertEqual(len(hist.list_batches()), 1)

            undone = hist.undo_batch(batch["batch_id"])
            self.assertEqual(undone["restored"], 4)
            self.assertTrue((root / "raw_123.safetensors").is_file())
            self.assertFalse((root / "CleanStyle.safetensors").exists())
            self.assertEqual(len(hist.list_batches()), 0)

    def test_plan_conflict_and_rollback_on_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.safetensors").write_bytes(b"1")
            (root / "b.safetensors").write_bytes(b"2")
            plan, errs = plan_renames([
                {"parent_dir": root, "base": "a", "new_name": "b", "enabled": True},
            ])
            self.assertEqual(plan, [])
            self.assertTrue(any("已存在" in e for e in errs))

            orig_rename = Path.rename
            calls = 0

            def flaky_rename(self_path, target):
                nonlocal calls
                calls += 1
                # Succeed stage 1 tmp rename and rollback, fail only on forward commit c.safetensors
                if Path(target).name == "c.safetensors":
                    raise OSError("disk full simulated")
                return orig_rename(self_path, target)

            with mock.patch.object(Path, "rename", flaky_rename):
                with self.assertRaises(RuntimeError):
                    execute_atomic_moves([(root / "a.safetensors", root / "c.safetensors")])

            # Must roll back to original path cleanly
            self.assertTrue((root / "a.safetensors").is_file())
            self.assertFalse((root / "c.safetensors").exists())
            leftovers = [p for p in root.iterdir() if p.name.startswith(".__mo_tmp_")]
            self.assertEqual(leftovers, [])

    def test_junction_safe_subdir_and_third_party_companions(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "Lora"
            ext_target = Path(td) / "ExternalPool"
            base.mkdir()
            ext_target.mkdir()

            with self.assertRaises(ValueError):
                safe_resolve_subdir(base, "../ExternalPool")

            junc = base / "LinkedPool"
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(junc), str(ext_target)], check=True, capture_output=True)
            else:
                junc.symlink_to(ext_target)

            try:
                resolved_sub = safe_resolve_subdir(base, "LinkedPool")
                self.assertEqual(resolved_sub, logical_abspath(junc))

                (junc / "old_j.safetensors").write_bytes(b"M")
                (junc / "old_j.safetensors.weilin-info.json").write_text("{}", encoding="utf-8")
                (junc / "old_j.source.json").write_text("{}", encoding="utf-8")
                write_json_atomic(
                    junc / "old_j.civitai.info",
                    {"model": {"name": "JunctionStyle", "type": "LORA"}, "name": "v1"},
                )

                records = scan_rename_candidates(base, recursive=True)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["rel_dir"], "LinkedPool")
                self.assertEqual(records[0]["file_count"], 4)

                plan, errs = plan_renames([
                    {"parent_dir": records[0]["parent_dir"], "base": "old_j", "new_name": "JunctionStyle", "enabled": True}
                ])
                self.assertEqual(errs, [])
                execute_atomic_moves(plan)
                self.assertTrue((junc / "JunctionStyle.safetensors").is_file())
                self.assertTrue((junc / "JunctionStyle.safetensors.weilin-info.json").is_file())
                self.assertTrue((junc / "JunctionStyle.source.json").is_file())
            finally:
                if os.name == "nt" and junc.exists():
                    subprocess.run(["cmd", "/c", "rmdir", str(junc)], capture_output=True)


if __name__ == "__main__":
    unittest.main()
