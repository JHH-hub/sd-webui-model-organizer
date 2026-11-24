import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from model_organizer.orphan_scanner import quarantine_orphans, scan_orphans
from model_organizer.preview_normalizer import apply_preview_normalizations
from model_organizer.trigger_sync import activation_text_from_info, apply_trigger_updates, write_json_atomic


class TriggersPreviewsOrphansTests(unittest.TestCase):
    def test_trigger_sync_protects_manual_edits(self):
        self.assertEqual(
            activation_text_from_info({"trainedWords": ["<lora:x:1>, hero_tag", "Hero_Tag", "pose"]}),
            "hero_tag, pose",
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sub = root / "chars"
            sub.mkdir()

            # 1. Missing .json -> create
            write_json_atomic(
                sub / "m1.civitai.info",
                {"trainedWords": ["m1_trig, <lora:m1:0.8>"], "baseModel": "Illustrious"},
            )
            # 2. Existing empty .json -> fill
            write_json_atomic(root / "m2.civitai.info", {"trainedWords": ["m2_trig"], "baseModel": "SDXL 1.0"})
            write_json_atomic(root / "m2.json", {"activation text": "", "preferred weight": 0.7})
            # 3. Existing manual .json -> protect
            write_json_atomic(root / "m3.civitai.info", {"trainedWords": ["auto_trig"]})
            write_json_atomic(root / "m3.json", {"activation text": "my_custom_prompt", "preferred weight": 0.9})

            dry = apply_trigger_updates(root, recursive=True, dry_run=True)
            self.assertEqual(dry["create"], 1)
            self.assertEqual(dry["fill"], 1)
            self.assertFalse((sub / "m1.json").exists())

            res = apply_trigger_updates(root, recursive=True, dry_run=False)
            self.assertEqual(res["updated"], 2)
            self.assertEqual(res["errors"], [])

            m1_json = json.loads((sub / "m1.json").read_text(encoding="utf-8"))
            self.assertEqual(m1_json["activation text"], "m1_trig")
            self.assertEqual(m1_json["sd version"], "Illustrious")

            m2_json = json.loads((root / "m2.json").read_text(encoding="utf-8"))
            self.assertEqual(m2_json["activation text"], "m2_trig")
            self.assertEqual(m2_json["preferred weight"], 0.7)

            m3_json = json.loads((root / "m3.json").read_text(encoding="utf-8"))
            self.assertEqual(m3_json["activation text"], "my_custom_prompt")

    def test_preview_normalizer_keep_ext_and_convert_png_and_no_prefix_collision(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "alpha.safetensors").write_bytes(b"1")
            (root / "alpha_extra.png").write_bytes(b"not_alpha_preview")
            Image.new("RGB", (32, 32), (10, 20, 30)).save(root / "alpha.webp", format="WEBP")

            (root / "beta.safetensors").write_bytes(b"2")
            (root / "beta.preview.jpg").write_bytes(b"already")

            dry = apply_preview_normalizations(root, mode="keep_ext", dry_run=True)
            self.assertEqual(dry["already_ok"], 1)
            self.assertEqual(dry["planned"], 1)

            res = apply_preview_normalizations(root, mode="keep_ext", dry_run=False)
            self.assertEqual(res["processed"], 1)
            self.assertTrue((root / "alpha.preview.webp").is_file())
            self.assertTrue((root / "alpha_extra.png").is_file())  # untouched!

            # Also test convert_png mode on a JPG image
            sub = root / "png_test"
            sub.mkdir()
            (sub / "gamma.safetensors").write_bytes(b"3")
            Image.new("RGB", (32, 32), (200, 100, 50)).save(sub / "gamma.jpg", format="JPEG")
            res2 = apply_preview_normalizations(sub, mode="convert_png", dry_run=False)
            self.assertEqual(res2["processed"], 1)
            png_path = sub / "gamma.preview.png"
            self.assertTrue(png_path.is_file())
            self.assertFalse((sub / "gamma.jpg").exists())
            self.assertEqual(png_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_orphan_scanner_and_quarantine(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Put LoRA here.txt").write_text("info", encoding="utf-8")
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            (root / "alive.safetensors").write_bytes(b"x")
            (root / "alive.civitai.info").write_text("{}", encoding="utf-8")
            (root / "alive.safetensors.weilin-info.json").write_text("{}", encoding="utf-8")
            (root / "alive.source.json").write_text("{}", encoding="utf-8")
            (root / "alive.来源.json").write_text("{}", encoding="utf-8")
            (root / "dead_model.civitai.info").write_text("{}", encoding="utf-8")
            (root / "dead_model.preview.png").write_bytes(b"img")
            (root / "dead_model.safetensors.weilin-info.json").write_text("{}", encoding="utf-8")

            orphans = scan_orphans(root, recursive=True)
            names = sorted(p.name for p in orphans)
            self.assertEqual(
                names,
                ["dead_model.civitai.info", "dead_model.preview.png", "dead_model.safetensors.weilin-info.json"],
            )

            qroot = Path(td) / "_quarantine"
            out = quarantine_orphans(orphans, root, qroot)
            self.assertEqual(out["moved"], 3)
            self.assertFalse((root / "dead_model.civitai.info").exists())
            self.assertTrue((root / "alive.civitai.info").is_file())
            self.assertTrue((root / "alive.safetensors.weilin-info.json").is_file())
            self.assertTrue((root / "adapter_config.json").is_file())
            moved_files = list(Path(out["quarantine_dir"]).rglob("*"))
            self.assertEqual(sorted(p.name for p in moved_files if p.is_file()), names)


if __name__ == "__main__":
    unittest.main()
