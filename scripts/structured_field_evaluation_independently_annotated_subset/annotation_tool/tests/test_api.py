import sys
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

from app import create_app  # noqa: E402


def hazard():
    return {
        "id": "h-1",
        "bbox": {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3},
        "object_name": "barrier",
        "hazard_category": "GROUND",
        "distance_steps": 3,
        "direction_bin": "ahead",
        "notes": "",
        "evaluation_role": "",
        "role_notes": "",
    }


class ApiTests(unittest.TestCase):
    def setUp(self):
        runtime = TOOL_DIR / "tests" / "_runtime" / uuid.uuid4().hex
        self.image_dir = runtime / "set"
        self.image_dir.mkdir(parents=True)
        Image.new("RGB", (120, 90), "white").save(self.image_dir / "sample.png")
        app = create_app(
            image_dir=self.image_dir,
            data_path=runtime / "annotations.json",
            backup_dir=runtime / "backups",
            static_dir=TOOL_DIR / "static",
            enable_background_backup=False,
        )
        self.client = TestClient(app)

    def test_lists_images_and_serves_file(self):
        response = self.client.get("/api/images")
        image = self.client.get("/api/image/sample.png")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["images"][0]["path"], "sample.png")
        self.assertEqual(response.json()["images"][0]["evaluation_review_status"], "pending")
        self.assertEqual(response.json()["summary"]["role_pending"], 1)
        self.assertEqual(response.json()["summary"]["role_completed"], 0)
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.headers["content-type"], "image/png")

    def test_favicon_request_does_not_log_as_missing(self):
        response = self.client.get("/favicon.ico")

        self.assertEqual(response.status_code, 204)

    def test_frontend_files_disable_cache_and_contain_required_elements(self):
        index = self.client.get("/")
        script = self.client.get("/static/app.js")
        required_ids = [
            "object-name",
            "distance-steps",
            "hazard-notes",
            "image-notes",
            "overlay",
            "canvas-viewport",
            "role-progress-text",
            "role-filter",
            "next-role-pending-btn",
            "complete-role-review-btn",
            "role-notes",
        ]

        self.assertEqual(index.headers["cache-control"], "no-store")
        self.assertEqual(script.headers["cache-control"], "no-store")
        for element_id in required_ids:
            self.assertIn(f'id="{element_id}"', index.text)
        self.assertIn("requiredElement", script.text)
        for role in ("required", "optional", "ignore"):
            self.assertIn(f'name="evaluation-role" value="{role}"', index.text)

    def test_text_inputs_are_excluded_before_space_shortcut(self):
        script = self.client.get("/static/app.js").text
        input_guard = 'if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;'
        space_shortcut = 'if (e.code === "Space") { state.spaceDown = true; e.preventDefault(); }'

        self.assertLess(script.index(input_guard), script.index(space_shortcut))

    def test_saves_and_reads_multiple_hazards(self):
        payload = {
            "status": "completed",
            "image_notes": "",
            "hazards": [hazard(), {**hazard(), "id": "h-2", "object_name": "bicycle"}],
        }
        saved = self.client.put("/api/annotations/sample.png", json=payload)
        fetched = self.client.get("/api/annotations/sample.png")

        self.assertEqual(saved.status_code, 200)
        self.assertEqual(len(fetched.json()["hazards"]), 2)

    def test_invalid_completed_annotation_returns_422(self):
        response = self.client.put(
            "/api/annotations/sample.png",
            json={"status": "completed", "image_notes": "", "hazards": []},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("at least one hazard", response.json()["detail"])

    def test_backup_and_export_endpoints(self):
        self.client.put(
            "/api/annotations/sample.png",
            json={"status": "draft", "image_notes": "", "hazards": [hazard()]},
        )
        backup = self.client.post("/api/backup")
        export = self.client.get("/api/export")

        self.assertEqual(backup.status_code, 200)
        self.assertTrue(backup.json()["created"])
        self.assertEqual(export.status_code, 200)
        self.assertIn("attachment", export.headers["content-disposition"])


if __name__ == "__main__":
    unittest.main()
