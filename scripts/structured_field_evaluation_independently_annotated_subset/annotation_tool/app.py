from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from core import AnnotationStore, ValidationError


TOOL_DIR = Path(__file__).resolve().parent
STRUCTURED_FIELD_DIR = TOOL_DIR.parent


def create_app(
    image_dir: Path = STRUCTURED_FIELD_DIR / "set",
    data_path: Path = STRUCTURED_FIELD_DIR / "annotations.json",
    backup_dir: Path = STRUCTURED_FIELD_DIR / "backups",
    static_dir: Path = TOOL_DIR / "static",
    enable_background_backup: bool = True,
) -> FastAPI:
    store = AnnotationStore(image_dir, data_path, backup_dir)
    static_dir = Path(static_dir)
    async def backup_loop() -> None:
        while True:
            await asyncio.sleep(600)
            updated_at = store.export_data().get("updated_at")
            if data_path.exists() and updated_at != app.state.last_backup_update:
                store.create_backup("auto")
                app.state.last_backup_update = updated_at

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if data_path.exists():
            store.create_backup("startup")
            app.state.last_backup_update = store.export_data().get("updated_at")
        if enable_background_backup:
            app.state.backup_task = asyncio.create_task(backup_loop())
        try:
            yield
        finally:
            task = getattr(app.state, "backup_task", None)
            if task:
                task.cancel()

    app = FastAPI(title="Structured-field Evaluation on an Independently Annotated Subset Multi-Hazard Annotation Tool", lifespan=lifespan)
    app.state.store = store
    app.state.last_backup_update = None

    def get_image_path(encoded_path: str) -> str:
        return unquote(encoded_path)

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        target = static_dir / "index.html"
        if not target.exists():
            raise HTTPException(status_code=500, detail="Frontend files are missing")
        return FileResponse(target, headers={"Cache-Control": "no-store"})

    @app.get("/static/{asset_path:path}")
    def static_asset(asset_path: str) -> FileResponse:
        requested = (static_dir / asset_path).resolve()
        try:
            requested.relative_to(static_dir.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Asset not found") from exc
        if not requested.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(requested, headers={"Cache-Control": "no-store"})

    @app.get("/favicon.ico", status_code=204)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/images")
    def list_images() -> dict[str, Any]:
        images = store.list_images()
        completed = sum(item["status"] == "completed" for item in images)
        draft = sum(item["status"] == "draft" for item in images)
        role_completed = sum(item["evaluation_review_status"] == "completed" for item in images)
        return {
            "images": images,
            "summary": {
                "total": len(images),
                "completed": completed,
                "draft": draft,
                "unannotated": len(images) - completed - draft,
                "role_completed": role_completed,
                "role_pending": len(images) - role_completed,
            },
            "state": store.get_state(),
            "object_name_suggestions": store.object_name_suggestions(),
        }

    @app.get("/api/image/{image_path:path}")
    def image_file(image_path: str) -> FileResponse:
        try:
            target = store.resolve_image(get_image_path(image_path))
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc
        return FileResponse(target)

    @app.get("/api/annotations/{image_path:path}")
    def get_annotation(image_path: str) -> dict[str, Any]:
        try:
            return store.get_annotation(get_image_path(image_path))
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc

    @app.put("/api/annotations/{image_path:path}")
    async def put_annotation(image_path: str, request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
            return store.save_annotation(
                get_image_path(image_path),
                payload,
                downgrade_invalid_completed=bool(payload.get("downgrade_invalid_completed")),
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    @app.post("/api/state/last-opened/{image_path:path}")
    def set_last_opened(image_path: str) -> dict[str, bool]:
        try:
            store.set_last_opened(get_image_path(image_path))
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc
        return {"saved": True}

    @app.post("/api/backup")
    def backup() -> dict[str, Any]:
        target = store.create_backup("manual")
        return {"created": target is not None, "filename": target.name if target else None}

    @app.get("/api/export")
    def export() -> Response:
        content = json.dumps(store.export_data(), ensure_ascii=False, indent=2) + "\n"
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="p0-05-annotations.json"'},
        )

    return app


app = create_app()
