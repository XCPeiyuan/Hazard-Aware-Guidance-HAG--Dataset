from __future__ import annotations

import json
import os
import re
import shutil
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
HAZARD_CATEGORIES = {"GROUND", "PIT", "OVERHEAD"}
DIRECTION_BINS = {
    "right",
    "right_ahead",
    "slightly_right_ahead",
    "ahead",
    "slightly_left_ahead",
    "left_ahead",
    "left",
}
MIRRORED_DIRECTION_BINS = {
    "right": "left",
    "right_ahead": "left_ahead",
    "slightly_right_ahead": "slightly_left_ahead",
    "ahead": "ahead",
    "slightly_left_ahead": "slightly_right_ahead",
    "left_ahead": "right_ahead",
    "left": "right",
}
EVALUATION_ROLES = {"required", "optional", "ignore"}
EVALUATION_REVIEW_STATUSES = {"pending", "completed"}
ENGLISH_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 '\-(),./]*$")


class ValidationError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnnotationStore:
    def __init__(
        self,
        image_dir: Path,
        data_path: Path,
        backup_dir: Path,
        max_backups: int = 100,
    ):
        self.image_dir = Path(image_dir).resolve()
        self.data_path = Path(data_path).resolve()
        self.backup_dir = Path(backup_dir).resolve()
        self.max_backups = max_backups
        self._lock = threading.RLock()
        self._data = self._load()

    def _empty_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "updated_at": utc_now(),
            "state": {"last_opened_image": None},
            "annotations": {},
        }

    def _load(self) -> dict[str, Any]:
        if not self.data_path.exists():
            return self._empty_data()
        try:
            loaded = json.loads(self.data_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read annotation data: {exc}") from exc
        loaded.setdefault("state", {"last_opened_image": None})
        loaded.setdefault("annotations", {})
        return loaded

    def _with_role_defaults(self, record: dict[str, Any]) -> dict[str, Any]:
        clean = deepcopy(record)
        clean.setdefault("evaluation_review_status", "pending")
        for hazard in clean.get("hazards", []):
            hazard.setdefault("evaluation_role", "")
            hazard.setdefault("role_notes", "")
        return clean

    def _atomic_write(self) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.data_path.with_suffix(self.data_path.suffix + ".tmp")
        text = json.dumps(self._data, ensure_ascii=False, indent=2) + "\n"
        temporary.write_text(text, encoding="utf-8")
        try:
            os.replace(temporary, self.data_path)
        except PermissionError:
            # Some managed Windows workspaces block rename/replace operations.
            self.data_path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
            try:
                temporary.unlink(missing_ok=True)
            except PermissionError:
                pass

    def resolve_image(self, relative_path: str) -> Path:
        normalized = PurePosixPath(relative_path.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError("Image path is outside the set directory")
        resolved = (self.image_dir / Path(*normalized.parts)).resolve()
        try:
            resolved.relative_to(self.image_dir)
        except ValueError as exc:
            raise ValueError("Image path is outside the set directory") from exc
        if not resolved.is_file() or resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise FileNotFoundError(relative_path)
        return resolved

    def list_images(self) -> list[dict[str, Any]]:
        annotations = self._data["annotations"]
        images = []
        if not self.image_dir.exists():
            return images
        paths = sorted(
            path
            for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for path in paths:
            relative = path.relative_to(self.image_dir).as_posix()
            with Image.open(path) as image:
                width, height = image.size
            annotation = annotations.get(relative)
            role_status = (
                self._with_role_defaults(annotation)["evaluation_review_status"]
                if annotation
                else "pending"
            )
            images.append(
                {
                    "path": relative,
                    "name": path.name,
                    "width": width,
                    "height": height,
                    "status": annotation["status"] if annotation else "unannotated",
                    "hazard_count": len(annotation.get("hazards", [])) if annotation else 0,
                    "evaluation_review_status": role_status,
                    "updated_at": annotation.get("updated_at") if annotation else None,
                }
            )
        return images

    def get_annotation(self, image_path: str) -> dict[str, Any]:
        image = self.resolve_image(image_path)
        relative = image.relative_to(self.image_dir).as_posix()
        existing = self._data["annotations"].get(relative)
        if existing:
            return self._with_role_defaults(existing)
        with Image.open(image) as opened:
            width, height = opened.size
        return {
            "image_path": relative,
            "image_width": width,
            "image_height": height,
            "status": "draft",
            "evaluation_review_status": "pending",
            "image_notes": "",
            "hazards": [],
            "updated_at": None,
        }

    def _validate_bbox(self, bbox: Any, prefix: str) -> None:
        if not isinstance(bbox, dict):
            raise ValidationError(f"{prefix}.bbox is required")
        for key in ("x", "y", "width", "height"):
            value = bbox.get(key)
            if not isinstance(value, (int, float)):
                raise ValidationError(f"{prefix}.bbox.{key} must be a number")
            if key in {"width", "height"} and not 0 < value <= 1:
                raise ValidationError(f"{prefix}.bbox.{key} must be greater than 0")
            if key in {"x", "y"} and not 0 <= value <= 1:
                raise ValidationError(f"{prefix}.bbox.{key} must be between 0 and 1")
        if bbox["x"] + bbox["width"] > 1.000001 or bbox["y"] + bbox["height"] > 1.000001:
            raise ValidationError(f"{prefix}.bbox must stay inside the image")

    def _validate_completed(self, record: dict[str, Any]) -> None:
        hazards = record.get("hazards")
        if not isinstance(hazards, list) or not hazards:
            raise ValidationError("completed images require at least one hazard")
        identifiers = set()
        for index, hazard in enumerate(hazards):
            prefix = f"hazards[{index}]"
            if not isinstance(hazard, dict):
                raise ValidationError(f"{prefix} must be an object")
            hazard_id = hazard.get("id")
            if not isinstance(hazard_id, str) or not hazard_id.strip():
                raise ValidationError(f"{prefix}.id is required")
            if hazard_id in identifiers:
                raise ValidationError(f"{prefix}.id must be unique")
            identifiers.add(hazard_id)
            self._validate_bbox(hazard.get("bbox"), prefix)
            name = hazard.get("object_name")
            if not isinstance(name, str) or not name.strip():
                raise ValidationError(f"{prefix}.object_name is required")
            if not ENGLISH_NAME_PATTERN.fullmatch(name.strip()):
                raise ValidationError(f"{prefix}.object_name must use English text")
            if hazard.get("hazard_category") not in HAZARD_CATEGORIES:
                raise ValidationError(f"{prefix}.hazard_category is required")
            distance = hazard.get("distance_steps")
            if not isinstance(distance, int) or isinstance(distance, bool) or distance < 1:
                raise ValidationError(f"{prefix}.distance_steps must be an integer >= 1")
            if hazard.get("direction_bin") not in DIRECTION_BINS:
                raise ValidationError(f"{prefix}.direction_bin is required")

    def _validate_evaluation_review(self, record: dict[str, Any]) -> None:
        if record.get("evaluation_review_status") != "completed":
            return
        for index, hazard in enumerate(record.get("hazards", [])):
            if hazard.get("evaluation_role") not in EVALUATION_ROLES:
                raise ValidationError(f"hazards[{index}].evaluation_role is required")

    def _mirror_path(self, source_path: str) -> str | None:
        source = PurePosixPath(source_path)
        if source.stem.endswith("_m"):
            return None
        return str(source.with_name(f"{source.stem}_m{source.suffix}"))

    def _hazards_are_mirrors(self, source: dict[str, Any], mirror: dict[str, Any]) -> bool:
        source_hazards = source.get("hazards", [])
        mirror_hazards = mirror.get("hazards", [])
        if len(source_hazards) != len(mirror_hazards):
            return False
        for left, right in zip(source_hazards, mirror_hazards):
            left_box = left.get("bbox") or {}
            right_box = right.get("bbox") or {}
            expected = {
                "x": 1 - left_box.get("x", 0) - left_box.get("width", 0),
                "y": left_box.get("y"),
                "width": left_box.get("width"),
                "height": left_box.get("height"),
            }
            if any(
                not isinstance(right_box.get(key), (int, float))
                or not isinstance(value, (int, float))
                or abs(right_box[key] - value) > 0.000001
                for key, value in expected.items()
            ):
                return False
            if right.get("direction_bin") != MIRRORED_DIRECTION_BINS.get(left.get("direction_bin")):
                return False
        return True

    def _inherit_mirror_roles(self, source_path: str, source: dict[str, Any]) -> None:
        mirror_path = self._mirror_path(source_path)
        if not mirror_path or source.get("evaluation_review_status") != "completed":
            return
        mirror = self._data["annotations"].get(mirror_path)
        if not mirror or not self._hazards_are_mirrors(source, mirror):
            return
        mirror = self._with_role_defaults(mirror)
        for source_hazard, mirror_hazard in zip(source["hazards"], mirror["hazards"]):
            mirror_hazard["evaluation_role"] = source_hazard["evaluation_role"]
            mirror_hazard["role_notes"] = source_hazard.get("role_notes", "")
        mirror["evaluation_review_status"] = "completed"
        mirror["updated_at"] = source["updated_at"]
        self._data["annotations"][mirror_path] = mirror

    def save_annotation(
        self,
        image_path: str,
        record: dict[str, Any],
        downgrade_invalid_completed: bool = False,
    ) -> dict[str, Any]:
        image = self.resolve_image(image_path)
        relative = image.relative_to(self.image_dir).as_posix()
        if not isinstance(record, dict):
            raise ValidationError("annotation must be an object")
        status = record.get("status", "draft")
        if status not in {"draft", "completed"}:
            raise ValidationError("status must be draft or completed")
        review_status = record.get("evaluation_review_status", "pending")
        if review_status not in EVALUATION_REVIEW_STATUSES:
            raise ValidationError("evaluation_review_status must be pending or completed")
        hazards = deepcopy(record.get("hazards", []))
        for hazard in hazards:
            if isinstance(hazard, dict):
                hazard["evaluation_role"] = str(hazard.get("evaluation_role", ""))
                hazard["role_notes"] = str(hazard.get("role_notes", ""))
        clean = {
            "image_path": relative,
            "image_width": None,
            "image_height": None,
            "status": status,
            "evaluation_review_status": review_status,
            "image_notes": str(record.get("image_notes", "")),
            "hazards": hazards,
            "updated_at": utc_now(),
        }
        with Image.open(image) as opened:
            clean["image_width"], clean["image_height"] = opened.size
        if status == "completed":
            try:
                self._validate_completed(clean)
            except ValidationError:
                if not downgrade_invalid_completed:
                    raise
                clean["status"] = "draft"
        self._validate_evaluation_review(clean)
        with self._lock:
            self._data["annotations"][relative] = clean
            self._inherit_mirror_roles(relative, clean)
            self._data["state"]["last_opened_image"] = relative
            self._data["updated_at"] = clean["updated_at"]
            self._atomic_write()
        return deepcopy(clean)

    def set_last_opened(self, image_path: str) -> None:
        resolved = self.resolve_image(image_path)
        relative = resolved.relative_to(self.image_dir).as_posix()
        with self._lock:
            self._data["state"]["last_opened_image"] = relative
            self._data["updated_at"] = utc_now()
            self._atomic_write()

    def get_state(self) -> dict[str, Any]:
        return deepcopy(self._data["state"])

    def export_data(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def object_name_suggestions(self) -> list[str]:
        names = {
            hazard.get("object_name", "").strip()
            for record in self._data["annotations"].values()
            for hazard in record.get("hazards", [])
            if hazard.get("object_name", "").strip()
        }
        return sorted(names, key=str.casefold)

    def create_backup(self, label: str = "auto") -> Path | None:
        with self._lock:
            if not self.data_path.exists():
                return None
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-") or "backup"
            target = self.backup_dir / f"annotations-{stamp}-{safe_label}.json"
            shutil.copy2(self.data_path, target)
            backups = sorted(self.backup_dir.glob("annotations-*.json"))
            for old in backups[:-self.max_backups]:
                old.unlink(missing_ok=True)
            return target
