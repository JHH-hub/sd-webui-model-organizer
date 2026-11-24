import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_organizer.civitai_client import (
    CivitaiClient,
    CivitaiConfig,
    backfill_missing_info,
    compute_sha256,
    parse_civitai_url,
    rewrite_civitai_host,
)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", headers=None, reason="OK"):
        self.status_code = status_code
        self._json = json_data
        self._content = content
        self.headers = headers or {}
        self.reason = reason
        self.ok = 200 <= status_code < 400
        self.text = json.dumps(json_data) if json_data is not None else content.decode("utf-8", errors="ignore")

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1024):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def close(self):
        pass


class CivitaiClientTests(unittest.TestCase):
    def test_url_parsing_and_domain_rewrite(self):
        self.assertEqual(parse_civitai_url("2104817"), ("2104817", None))
        self.assertEqual(
            parse_civitai_url("https://civitai.red/models/2091887/my-slug?modelVersionId=9988"),
            ("2091887", "9988"),
        )
        with self.assertRaises(ValueError):
            parse_civitai_url("https://example.com/not-a-model")

        self.assertEqual(
            rewrite_civitai_host("https://civitai.com/api/download/models/123?type=Model", "civitai.red"),
            "https://civitai.red/api/download/models/123?type=Model",
        )
        # Image CDN host must not be rewritten
        self.assertEqual(
            rewrite_civitai_host("https://image.civitai.com/x/y/z.jpeg", "civitai.red"),
            "https://image.civitai.com/x/y/z.jpeg",
        )

    def test_resumable_part_download_and_auto_organize(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Pre-seed partial file for HTTP 206 resume check
            (root / "raw_model_file.safetensors.part").write_bytes(b"HEAD_")

            model_data = {"id": 101, "name": "Test:Lora", "type": "LORA", "nsfw": False}
            version_data = {
                "id": 202,
                "name": "v1.0",
                "baseModel": "Illustrious",
                "trainedWords": ["test_trigger, <lora:raw:1>"],
                "files": [
                    {
                        "name": "raw_model_file.safetensors",
                        "primary": True,
                        "type": "Model",
                        "downloadUrl": "https://civitai.com/api/download/models/202",
                    }
                ],
                "images": [
                    {"url": "https://image.civitai.com/a/nsfw.jpeg", "nsfw": True, "type": "image"},
                    {"url": "https://image.civitai.com/a/sfw.webp", "nsfw": False, "type": "image"},
                ],
            }

            captured_urls = []

            def fake_get(url, headers=None, proxies=None, stream=False, timeout=30):
                captured_urls.append((url, dict(headers or {}), proxies))
                if "download/models/202" in url:
                    self.assertEqual(headers.get("Range"), "bytes=5-")
                    return FakeResponse(206, content=b"TAIL_OK", headers={"Content-Length": "7"})
                if "sfw.webp" in url:
                    return FakeResponse(200, content=b"RIFF....WEBP")
                raise AssertionError(f"Unexpected URL: {url}")

            sess = mock.MagicMock()
            sess.get.side_effect = fake_get
            client = CivitaiClient(
                CivitaiConfig(domain="civitai.red", api_key="secret_k", proxy="socks5h://127.0.0.1:7890", skip_nsfw_preview=True),
                session=sess,
            )

            res = client.download_model_version(
                model_data, version_data, root, auto_rename=True, auto_sync_triggers=True
            )
            self.assertEqual(captured_urls[0][0], "https://civitai.red/api/download/models/202")
            self.assertEqual(captured_urls[0][2], {"http": "socks5h://127.0.0.1:7890", "https": "socks5h://127.0.0.1:7890"})
            self.assertEqual(res["renamed_to"], "TestLora")
            self.assertEqual(res["model_path"].read_bytes(), b"HEAD_TAIL_OK")
            self.assertTrue((root / "TestLora.preview.webp").is_file())
            self.assertEqual(res["preview_path"].read_bytes(), b"RIFF....WEBP")
            self.assertEqual(res["activation_text"], "test_trigger")
            self.assertTrue((root / "TestLora.json").is_file())
            self.assertTrue((root / "TestLora.civitai.info").is_file())

    def test_backfill_missing_info_skips_vae_and_caches_404(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            m_hit = root / "hit_model.safetensors"
            m_miss = root / "miss_model.safetensors"
            m_vae = root / "sdxl.vae.safetensors"
            m_hit.write_bytes(b"HIT_BYTES")
            m_miss.write_bytes(b"MISS_BYTES")
            m_vae.write_bytes(b"VAE_BYTES")

            hit_sha = compute_sha256(m_hit)
            miss_sha = compute_sha256(m_miss)
            cache_file = root / "_not_found.json"

            queried_urls = []

            def fake_get(url, headers=None, proxies=None, stream=False, timeout=30):
                queried_urls.append(url)
                if url.endswith(hit_sha):
                    return FakeResponse(
                        200,
                        json_data={
                            "id": 900,
                            "modelId": 500,
                            "name": "v2",
                            "baseModel": "Illustrious",
                            "trainedWords": ["hit_word"],
                            "model": {"name": "HitModel", "type": "LORA"},
                            "images": [],
                        },
                    )
                if url.endswith(miss_sha):
                    return FakeResponse(404, json_data={"error": "Not found"})
                raise AssertionError(f"Unexpected URL: {url}")

            sess = mock.MagicMock()
            sess.get.side_effect = fake_get
            client = CivitaiClient(CivitaiConfig(domain="civitai.red"), session=sess)

            res1 = backfill_missing_info(
                root, client, recursive=True, limit=10, download_preview=False, sync_triggers=True,
                not_found_cache_path=cache_file, delay_sec=0,
            )
            self.assertEqual(res1["matched"], 1)
            self.assertEqual(res1["not_found"], 1)
            self.assertEqual(res1["triggers_synced"], 1)
            self.assertTrue((root / "hit_model.civitai.info").is_file())
            self.assertTrue((root / "hit_model.json").is_file())
            self.assertFalse((root / "sdxl.vae.civitai.info").exists())

            # Second run must skip cached miss hash cleanly (0 HTTP calls)
            queried_urls.clear()
            res2 = backfill_missing_info(
                root, client, recursive=True, limit=10, download_preview=False, sync_triggers=True,
                not_found_cache_path=cache_file, delay_sec=0,
            )
            self.assertEqual(res2["queried"], 0)
            self.assertEqual(res2["skipped_cached"], 1)
            self.assertEqual(queried_urls, [])


if __name__ == "__main__":
    unittest.main()
