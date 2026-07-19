import json
import sys
import unittest
import uuid
from pathlib import Path

from PIL import Image


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

from core import AnnotationStore, ValidationError  # noqa: E402


def make_image(path: Path, size=(320, 240)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def valid_hazard(hazard_id="hazard-1", evaluation_role=""):
    return {
        "id": hazard_id,
        "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
        "object_name": "open manhole",
        "hazard_category": "PIT",
        "distance_steps": 4,
        "direction_bin": "left_ahead",
        "notes": "",
        "evaluation_role": evaluation_role,
        "role_notes": "",
    }


class AnnotationStoreTests(unittest.TestCase):
    def setUp(self):
        runtime_dir = TOOL_DIR / "tests" / "_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.root = runtime_dir / uuid.uuid4().hex
        self.root.mkdir()
        self.image_dir = self.root / "set"
        self.data_path = self.root / "annotations.json"
        self.backup_dir = self.root / "backups"
        make_image(self.image_dir / "a.png", (100, 80))
        make_image(self.image_dir / "nested" / "b.jpg", (200, 150))
        (self.image_dir / "ignored.txt").write_text("no", encoding="utf-8")
        self.store = AnnotationStore(
            image_dir=self.image_dir,
            data_path=self.data_path,
            backup_dir=self.backup_dir,
            max_backups=3,
        )

    def tearDown(self):
        pass

    def test_scans_supported_images_recursively_with_dimensions(self):
        images = self.store.list_images()

        self.assertEqual([item["path"] for item in images], ["a.png", "nested/b.jpg"])
        self.assertEqual(images[0]["width"], 100)
        self.assertEqual(images[1]["height"], 150)
        self.assertEqual(images[0]["status"], "unannotated")

    def test_rejects_completed_record_with_missing_required_field(self):
        with self.assertRaisesRegex(ValidationError, "object_name"):
            self.store.save_annotation(
                "a.png",
                {
                    "status": "completed",
                    "image_notes": "",
                    "hazards": [{**valid_hazard(), "object_name": ""}],
                },
            )

    def test_completed_record_can_contain_multiple_hazards(self):
        saved = self.store.save_annotation(
            "a.png",
            {
                "status": "completed",
                "image_notes": "two hazards",
                "hazards": [valid_hazard(), valid_hazard("hazard-2")],
            },
        )

        self.assertEqual(saved["status"], "completed")
        self.assertEqual(len(saved["hazards"]), 2)
        persisted = json.loads(self.data_path.read_text(encoding="utf-8"))
        self.assertEqual(len(persisted["annotations"]["a.png"]["hazards"]), 2)

    def test_invalid_completed_update_is_downgraded_to_draft(self):
        saved = self.store.save_annotation(
            "a.png",
            {
                "status": "completed",
                "image_notes": "",
                "hazards": [{**valid_hazard(), "distance_steps": None}],
            },
            downgrade_invalid_completed=True,
        )

        self.assertEqual(saved["status"], "draft")

    def test_rejects_image_path_outside_set(self):
        make_image(self.root / "outside.png")

        with self.assertRaisesRegex(ValueError, "outside"):
            self.store.resolve_image("../outside.png")

    def test_backup_rotation_keeps_latest_files(self):
        self.store.save_annotation(
            "a.png",
            {"status": "draft", "image_notes": "", "hazards": [valid_hazard()]},
        )
        for index in range(5):
            self.store.create_backup(label=f"manual-{index}")

        backups = sorted(self.backup_dir.glob("*.json"))
        self.assertEqual(len(backups), 3)
        self.assertTrue(any("manual-4" in path.name for path in backups))

    def test_state_persists_last_opened_image(self):
        self.store.set_last_opened("nested/b.jpg")
        reloaded = AnnotationStore(self.image_dir, self.data_path, self.backup_dir)

        self.assertEqual(reloaded.get_state()["last_opened_image"], "nested/b.jpg")

    def test_old_records_receive_pending_role_defaults_without_rewriting_file(self):
        old_record = {
            "schema_version": 1,
            "updated_at": "old",
            "state": {"last_opened_image": None},
            "annotations": {
                "a.png": {
                    "image_path": "a.png",
                    "image_width": 100,
                    "image_height": 80,
                    "status": "completed",
                    "image_notes": "",
                    "hazards": [
                        {
                            key: value
                            for key, value in valid_hazard().items()
                            if key not in {"evaluation_role", "role_notes"}
                        }
                    ],
                    "updated_at": "old",
                }
            },
        }
        self.data_path.write_text(json.dumps(old_record), encoding="utf-8")
        before = self.data_path.read_text(encoding="utf-8")

        reloaded = AnnotationStore(self.image_dir, self.data_path, self.backup_dir)
        annotation = reloaded.get_annotation("a.png")

        self.assertEqual(annotation["evaluation_review_status"], "pending")
        self.assertEqual(annotation["hazards"][0]["evaluation_role"], "")
        self.assertEqual(annotation["hazards"][0]["role_notes"], "")
        self.assertEqual(self.data_path.read_text(encoding="utf-8"), before)

    def test_new_annotation_defaults_to_pending_role_review(self):
        annotation = self.store.get_annotation("nested/b.jpg")

        self.assertEqual(annotation["evaluation_review_status"], "pending")

    def test_completed_role_review_requires_every_hazard_role(self):
        with self.assertRaisesRegex(ValidationError, "evaluation_role"):
            self.store.save_annotation(
                "a.png",
                {
                    "status": "completed",
                    "evaluation_review_status": "completed",
                    "image_notes": "",
                    "hazards": [valid_hazard()],
                },
            )

    def test_role_review_does_not_change_completed_base_status(self):
        saved = self.store.save_annotation(
            "a.png",
            {
                "status": "completed",
                "evaluation_review_status": "completed",
                "image_notes": "",
                "hazards": [valid_hazard(evaluation_role="required")],
            },
        )

        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["evaluation_review_status"], "completed")

    def test_completed_source_save_inherits_roles_to_matching_mirror(self):
        make_image(self.image_dir / "a_m.png", (100, 80))
        source_hazard = valid_hazard(evaluation_role="required")
        mirror_hazard = {
            **valid_hazard("mirror-hazard"),
            "bbox": {"x": 0.6, "y": 0.2, "width": 0.3, "height": 0.4},
            "direction_bin": "right_ahead",
        }
        self.store.save_annotation(
            "a_m.png",
            {
                "status": "completed",
                "evaluation_review_status": "pending",
                "image_notes": "",
                "hazards": [mirror_hazard],
            },
        )

        self.store.save_annotation(
            "a.png",
            {
                "status": "completed",
                "evaluation_review_status": "completed",
                "image_notes": "",
                "hazards": [source_hazard],
            },
        )
        mirror = self.store.get_annotation("a_m.png")

        self.assertEqual(mirror["status"], "completed")
        self.assertEqual(mirror["evaluation_review_status"], "completed")
        self.assertEqual(mirror["hazards"][0]["evaluation_role"], "required")

    def test_incompatible_mirror_stays_pending(self):
        make_image(self.image_dir / "a_m.png", (100, 80))
        self.store.save_annotation(
            "a_m.png",
            {
                "status": "completed",
                "evaluation_review_status": "pending",
                "image_notes": "",
                "hazards": [valid_hazard("mirror-hazard")],
            },
        )

        self.store.save_annotation(
            "a.png",
            {
                "status": "completed",
                "evaluation_review_status": "completed",
                "image_notes": "",
                "hazards": [valid_hazard(evaluation_role="required")],
            },
        )

        self.assertEqual(
            self.store.get_annotation("a_m.png")["evaluation_review_status"],
            "pending",
        )


if __name__ == "__main__":
    unittest.main()
