from __future__ import annotations

import importlib.util
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
STRUCTURED_FIELD_DIR = TOOL_DIR.parent


def require_environment() -> None:
    missing = [
        name
        for name in ("fastapi", "uvicorn", "PIL", "pydantic")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError("缺少依赖: " + ", ".join(missing))
    if not (STRUCTURED_FIELD_DIR / "set").is_dir():
        raise RuntimeError(f"图片目录不存在: {STRUCTURED_FIELD_DIR / 'set'}")
    if not (TOOL_DIR / "static" / "index.html").is_file():
        raise RuntimeError("网页文件缺失，请检查 annotation_tool/static")


def choose_port(start: int = 8765, attempts: int = 30) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("未找到可用端口")


def main() -> int:
    try:
        require_environment()
        if "--check" in sys.argv[1:]:
            print("Environment check passed")
            return 0
        port = choose_port()
    except RuntimeError as exc:
        print(f"\n启动失败: {exc}\n")
        return 1

    # A unique query prevents an older cached index from loading with newer assets.
    url = f"http://127.0.0.1:{port}/?launch={time.time_ns()}"
    print("=" * 60)
    print("Structured-field Evaluation on an Independently Annotated Subset 多障碍物标注工具")
    print(f"地址: {url}")
    print("关闭窗口或按 Ctrl+C 可停止服务。")
    print("=" * 60)
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=port, app_dir=str(TOOL_DIR), log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
