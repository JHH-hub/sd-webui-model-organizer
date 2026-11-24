import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_organizer.card_meta import extract_example_images, load_card_metadata, resolve_model_file
from model_organizer.civitai_client import CivitaiClient, CivitaiConfig
from model_organizer.trigger_sync import write_json_atomic
from model_organizer.update_checker import check_model_updates, collect_local_civitai_models


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self._json = json_data
        self.text = ""
        self.reason = "OK"

    def json(self):
        return self._json


class _FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        res = self.responses.get(url)
        if isinstance(res, Exception):
            raise res
        return res or _FakeResponse(status_code=404)


class CardMetaAndUpdatesTests(unittest.TestCase):
    def test_resolve_model_file_enforces_containment_and_supports_subdirs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lora_dir = root / "Lora"
            sub = lora_dir / "Anime"
            sub.mkdir(parents=True)
            m1 = sub / "frieren_v1.safetensors"
            m1.write_bytes(b"DATA")

            outside = root / "secret.safetensors"
            outside.write_bytes(b"SECRET")

            dirs = {"lora": lora_dir}
            # Exact path inside allowed root works (case differences handled cleanly)
            resolved = resolve_model_file(dirs, "lora", "frieren_v1", path_hint=str(m1))
            self.assertEqual(resolved, m1)

            # Traversal outside root is strictly rejected
            escape_hint = str(lora_dir / ".." / "secret.safetensors")
            self.assertIsNone(resolve_model_file(dirs, "lora", "secret", path_hint=escape_hint))
            self.assertIsNone(resolve_model_file(dirs, "lora", "secret", path_hint=str(outside)))

            # Lookup by name / relative name without path_hint works
            self.assertEqual(resolve_model_file(dirs, "lora", "frieren_v1"), m1)
            self.assertEqual(resolve_model_file(dirs, "lora", "Anime/frieren_v1"), m1)

    def test_load_card_metadata_trigger_priority_and_gallery_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            lora_dir = Path(td) / "Lora"
            lora_dir.mkdir()
            (lora_dir / "demo_lora.safetensors").write_bytes(b"X")
            write_json_atomic(
                lora_dir / "demo_lora.civitai.info",
                {
                    "id": 9901,
                    "modelId": 1234,
                    "name": "v2.0",
                    "baseModel": "Illustrious",
                    "trainedWords": ["<lora:demo:1>", "info_word_a", "info_word_b"],
                    "model": {"name": "Demo Style", "type": "LORA"},
                    "images": [
                        {"url": "https://image.civitai.com/demo/vid.mp4", "type": "video"},
                        {
                            "url": "https://image.civitai.com/demo/sfw.png",
                            "nsfwLevel": 1,
                            "width": 832,
                            "height": 1216,
                            "meta": {
                                "prompt": "1girl, solo",
                                "negativePrompt": "lowres",
                                "seed": 42,
                                "steps": 28,
                                "cfgScale": 6.5,
                                "sampler": "Euler a",
                                "Model": "NoobAI",
                            },
                        },
                        {
                            "url": "https://image.civitai.com/demo/nsfw.png",
                            "nsfw": True,
                            "nsfwLevel": 8,
                            "meta": {"prompt": "mature"},
                        },
                    ],
                },
            )
            write_json_atomic(
                lora_dir / "demo_lora.json",
                {"activation text": "custom_manual_trigger, info_word_a"},
            )

            meta = load_card_metadata({"lora": lora_dir}, "lora", "demo_lora", domain="civitai.red")
            self.assertTrue(meta["found"])
            self.assertEqual(meta["civitai_url"], "https://civitai.red/models/1234?modelVersionId=9901")
            self.assertEqual(meta["activation_text"], "custom_manual_trigger, info_word_a")
            self.assertEqual(meta["trained_words"], ["info_word_a", "info_word_b"])
            self.assertEqual(len(meta["images"]), 2)
            self.assertFalse(meta["images"][0]["nsfw"])
            self.assertEqual(meta["images"][0]["size"], "832x1216")
            self.assertEqual(meta["images"][0]["negative_prompt"], "lowres")
            self.assertEqual(meta["images"][0]["sampler"], "Euler a")
            self.assertTrue(meta["images"][1]["nsfw"])

            # Without .json, falls back to cleaned trainedWords
            (lora_dir / "demo_lora.json").unlink()
            meta2 = load_card_metadata({"lora": lora_dir}, "lora", "demo_lora", domain="civitai.com")
            self.assertEqual(meta2["activation_text"], "info_word_a, info_word_b")
            self.assertEqual(meta2["civitai_url"], "https://civitai.com/models/1234?modelVersionId=9901")
            self.assertEqual(len(extract_example_images({"images": []}, max_images=5)), 0)

    def test_check_model_updates_base_filter_dedup_and_error_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "Lora"
            root.mkdir()
            # Two local versions of modelId 100 (v1 + v2 already downloaded -> should compare against v2!)
            write_json_atomic(
                root / "char_v1.civitai.info",
                {
                    "id": 1001,
                    "modelId": 100,
                    "name": "v1",
                    "baseModel": "Illustrious",
                    "publishedAt": "2026-01-01T00:00:00Z",
                    "model": {"name": "Char Style"},
                },
            )
            write_json_atomic(
                root / "char_v2.civitai.info",
                {
                    "id": 1002,
                    "modelId": 100,
                    "name": "v2",
                    "baseModel": "Illustrious",
                    "publishedAt": "2026-02-01T00:00:00Z",
                    "model": {"name": "Char Style"},
                },
            )
            # Another model 200 where remote only has a newer Flux version
            write_json_atomic(
                root / "style_il.civitai.info",
                {
                    "id": 2001,
                    "modelId": 200,
                    "name": "v1-IL",
                    "baseModel": "Illustrious",
                    "publishedAt": "2026-01-10T00:00:00Z",
                    "model": {"name": "CrossBase Style"},
                },
            )
            # Broken/deleted model 300 -> error isolated
            write_json_atomic(
                root / "broken.civitai.info",
                {
                    "id": 3001,
                    "modelId": 300,
                    "name": "v1",
                    "baseModel": "Illustrious",
                    "publishedAt": "2026-01-05T00:00:00Z",
                    "model": {"name": "Deleted Model"},
                },
            )

            groups = collect_local_civitai_models(root, recursive=True)
            self.assertEqual([g["model_id"] for g in groups], ["100", "200", "300"])

            sess = _FakeSession({
                "https://civitai.red/api/v1/models/100": _FakeResponse(json_data={
                    "id": 100,
                    "name": "Char Style",
                    "modelVersions": [
                        {"id": 1003, "name": "v3", "baseModel": "Illustrious", "status": "Published", "publishedAt": "2026-03-01T00:00:00Z"},
                        {"id": 1002, "name": "v2", "baseModel": "Illustrious", "status": "Published", "publishedAt": "2026-02-01T00:00:00Z"},
                    ],
                }),
                "https://civitai.red/api/v1/models/200": _FakeResponse(json_data={
                    "id": 200,
                    "name": "CrossBase Style",
                    "modelVersions": [
                        {"id": 2002, "name": "v2-Flux", "baseModel": "Flux.1 D", "status": "Published", "publishedAt": "2026-04-01T00:00:00Z"},
                        {"id": 2001, "name": "v1-IL", "baseModel": "Illustrious", "status": "Published", "publishedAt": "2026-01-10T00:00:00Z"},
                    ],
                }),
                "https://civitai.red/api/v1/models/300": _FakeResponse(status_code=404),
            })
            client = CivitaiClient(CivitaiConfig(domain="civitai.red"), session=sess)

            # same_base_model_only=True suppresses Flux v2 update for 200, surfaces v3 for 100, records 404 error for 300
            res_same = check_model_updates(root, client, same_base_model_only=True, delay_sec=0)
            self.assertEqual(res_same["total_models"], 3)
            self.assertEqual(res_same["checked"], 2)
            self.assertEqual(res_same["updates_found"], 1)
            self.assertEqual(res_same["updates"][0]["latest_version_name"], "v3")
            self.assertIn("https://civitai.red/models/100?modelVersionId=1003", res_same["updates"][0]["url"])
            self.assertEqual(len(res_same["errors"]), 1)

            # same_base_model_only=False surfaces Flux update for 200 as well
            res_all = check_model_updates(root, client, same_base_model_only=False, delay_sec=0)
            self.assertEqual(res_all["updates_found"], 2)
            self.assertEqual({u["model_id"] for u in res_all["updates"]}, {"100", "200"})

    def test_fastapi_endpoints_via_on_app_started(self):
        with tempfile.TemporaryDirectory() as td:
            lora_dir = Path(td) / "Lora"
            lora_dir.mkdir()
            (lora_dir / "card_item.safetensors").write_bytes(b"M")
            write_json_atomic(
                lora_dir / "card_item.civitai.info",
                {
                    "id": 77,
                    "modelId": 88,
                    "name": "v1",
                    "trainedWords": ["trig_one"],
                    "model": {"name": "Card Item"},
                },
            )

            modules_pkg = types.ModuleType("modules")
            shared_mod = types.ModuleType("modules.shared")
            shared_mod.cmd_opts = types.SimpleNamespace(lora_dir=str(lora_dir), gradio_allowed_path=[])
            shared_mod.opts = types.SimpleNamespace(
                data={"mo_civitai_domain": "civitai.red", "mo_enable_card_buttons": True, "mo_blur_nsfw_gallery": False},
                add_option=lambda *a, **k: None,
            )
            cb_mod = types.ModuleType("modules.script_callbacks")
            cb_mod.on_ui_settings = lambda fn: None
            cb_mod.on_ui_tabs = lambda fn: None
            cb_mod.on_before_ui = lambda fn: None
            cb_mod.on_app_started = lambda fn: None
            modules_pkg.shared = shared_mod
            modules_pkg.script_callbacks = cb_mod

            with mock.patch.dict(
                sys.modules,
                {"modules": modules_pkg, "modules.shared": shared_mod, "modules.script_callbacks": cb_mod},
            ):
                sys.modules.pop("scripts.model_organizer_ui", None)
                ui = importlib.import_module("scripts.model_organizer_ui")
                ui.on_before_ui()
                allowed_pre = shared_mod.cmd_opts.gradio_allowed_path
                self.assertEqual(len(allowed_pre), 2)
                self.assertTrue(any(p.endswith("model_organizer_cards.js") for p in allowed_pre))
                self.assertTrue(any(p.endswith("style.css") for p in allowed_pre))
                self.assertFalse(any(p.rstrip("\\/").endswith(("sd-webui-model-organizer", "data")) for p in allowed_pre))

                fake_demo = types.SimpleNamespace(allowed_paths=[])
                app = FastAPI()
                ui.on_app_started(fake_demo, app)
                self.assertEqual(len(fake_demo.allowed_paths), 2)
                self.assertFalse(any(p.rstrip("\\/").endswith(("sd-webui-model-organizer", "data")) for p in fake_demo.allowed_paths))

                client = TestClient(app)
                cfg = client.get("/model_organizer/v1/config")
                self.assertEqual(cfg.status_code, 200)
                self.assertEqual(cfg.json()["domain"], "civitai.red")
                self.assertTrue(cfg.json()["enable_card_buttons"])
                self.assertFalse(cfg.json()["blur_nsfw_gallery"])

                meta = client.get("/model_organizer/v1/card_meta", params={"model_type": "lora", "name": "card_item"})
                self.assertEqual(meta.status_code, 200)
                self.assertTrue(meta.json()["found"])
                self.assertEqual(meta.json()["activation_text"], "trig_one")
                self.assertEqual(meta.json()["civitai_url"], "https://civitai.red/models/88?modelVersionId=77")
