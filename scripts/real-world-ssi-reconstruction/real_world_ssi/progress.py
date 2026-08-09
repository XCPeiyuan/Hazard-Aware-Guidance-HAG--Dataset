"""批处理进度事件、GPU 遥测、控制台面板与可审计日志。"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import os
import subprocess
import sys
import threading
import time
from typing import Callable, Literal, Protocol, Sequence, TextIO


class ProgressStage(str, Enum):
    READ_IMAGE = "read_image"
    MASK = "mask"
    DEPTH = "depth"
    SKY_QUALITY = "sky_quality"
    MINIMAX = "minimax"
    EXPORT = "export"
    FINALIZE = "finalize"
    INPUT_FILTER = "input_filter"


ProgressEventKind = Literal[
    "batch_start",
    "sample_start",
    "stage_start",
    "stage_end",
    "sample_end",
    "batch_end",
]


@dataclass(frozen=True)
class ProgressEvent:
    kind: ProgressEventKind
    timestamp: float
    index: int
    total: int
    sample_id: str = ""
    object_count: int = 0
    stage: ProgressStage | None = None
    status: str | None = None
    cached: bool = False
    elapsed_seconds: float = 0.0

    @classmethod
    def sample_started(
        cls,
        *,
        timestamp: float,
        index: int,
        total: int,
        sample_id: str,
        object_count: int,
    ) -> ProgressEvent:
        return cls(
            kind="sample_start",
            timestamp=timestamp,
            index=index,
            total=total,
            sample_id=sample_id,
            object_count=object_count,
            stage=ProgressStage.READ_IMAGE,
        )


class ProgressReporter(Protocol):
    def emit(self, event: ProgressEvent) -> None: ...

    def close(self) -> None: ...


class NullProgressReporter:
    def emit(self, event: ProgressEvent) -> None:
        del event

    def close(self) -> None:
        return None


def format_bar(completed: int, total: int, *, width: int = 20) -> str:
    if width <= 0:
        return ""
    if total <= 0:
        return "-" * width
    bounded = min(max(completed, 0), total)
    filled = int(bounded / total * width)
    if bounded == total:
        filled = width
    return "#" * filled + "-" * (width - filled)


def format_duration(seconds: float) -> str:
    value = int(max(seconds, 0.0))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def rolling_eta_seconds(
    durations: Sequence[float], *, remaining: int
) -> float | None:
    window = tuple(durations)[-50:]
    if len(window) < 3:
        return None
    return sum(window) / len(window) * max(remaining, 0)


@dataclass(frozen=True)
class GpuSnapshot:
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    device_name: str


class NvidiaSmiTelemetry:
    COMMAND = (
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,name",
        "--format=csv,noheader,nounits",
    )

    @staticmethod
    def parse(raw: str) -> GpuSnapshot:
        line = next(line.strip() for line in raw.splitlines() if line.strip())
        utilization, used, total, name = (part.strip() for part in line.split(",", 3))
        return GpuSnapshot(float(utilization), float(used), float(total), name)

    def sample(self) -> GpuSnapshot | None:
        try:
            result = subprocess.run(
                self.COMMAND,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=2,
                check=False,
            )
            if result.returncode != 0:
                return None
            return self.parse(result.stdout)
        except (OSError, subprocess.SubprocessError, StopIteration, ValueError):
            return None


def _enable_virtual_terminal(stream: TextIO) -> bool:
    if os.name != "nt":
        return True
    try:
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_uint()
        kernel32 = ctypes.windll.kernel32
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        enabled = mode.value | 0x0004
        return bool(kernel32.SetConsoleMode(handle, enabled))
    except (AttributeError, OSError, ValueError):
        return False


_STAGE_LABELS = {
    ProgressStage.READ_IMAGE: "Read image",
    ProgressStage.MASK: "Mask extraction",
    ProgressStage.DEPTH: "Depth estimation",
    ProgressStage.SKY_QUALITY: "Sky and quality checks",
    ProgressStage.MINIMAX: "MiniMax classification",
    ProgressStage.EXPORT: "SSI export",
    ProgressStage.FINALIZE: "Finalize reports",
    ProgressStage.INPUT_FILTER: "Input filter",
}

_STAGE_POSITIONS = {
    ProgressStage.READ_IMAGE: 1,
    ProgressStage.MASK: 2,
    ProgressStage.DEPTH: 3,
    ProgressStage.SKY_QUALITY: 4,
    ProgressStage.MINIMAX: 5,
    ProgressStage.EXPORT: 6,
    ProgressStage.FINALIZE: 7,
}


class ConsoleProgressReporter:
    def __init__(
        self,
        output_root: Path,
        *,
        stream: TextIO = sys.stdout,
        interactive: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = datetime.now,
        telemetry: NvidiaSmiTelemetry | None = None,
        refresh_seconds: float = 1.0,
        telemetry_seconds: float = 2.0,
        start_heartbeat: bool = True,
    ) -> None:
        self._stream = stream
        detected_interactive = stream.isatty() if interactive is None else interactive
        if interactive is None and detected_interactive:
            detected_interactive = _enable_virtual_terminal(stream)
        self._interactive = detected_interactive
        self._clock = clock
        self._wall_clock = wall_clock
        self._telemetry = telemetry or NvidiaSmiTelemetry()
        self._refresh_seconds = max(refresh_seconds, 0.01)
        self._telemetry_seconds = max(telemetry_seconds, 0.0)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._closed = False
        self._active = False
        self._index = 0
        self._total = 0
        self._sample_id = ""
        self._object_count = 0
        self._stage: ProgressStage | None = None
        self._stage_started_at = 0.0
        self._sample_started_at = 0.0
        self._run_started_at = self._clock()
        self._completed = 0
        self._counts: Counter[str] = Counter()
        self._durations: deque[float] = deque(maxlen=50)
        self._recent: deque[tuple[str, str, float]] = deque(maxlen=3)
        self._gpu: GpuSnapshot | None = None
        self._last_gpu_sample_at: float | None = None
        self._log = None
        try:
            reports = Path(output_root) / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            self._log = (reports / "run_progress.log").open(
                "a", encoding="utf-8", newline="\n"
            )
            self._log.write(
                f"\nRUN {self._wall_clock().isoformat(timespec='seconds')}\n"
            )
            self._log.flush()
        except OSError as exc:
            self._safe_write(f"WARNING: progress log unavailable: {exc}\n")
        if self._interactive:
            self._safe_write("\x1b[?25l")
        self._thread: threading.Thread | None = None
        if start_heartbeat:
            self._thread = threading.Thread(
                target=self._heartbeat,
                name="ssi-progress-heartbeat",
                daemon=True,
            )
            self._thread.start()

    def emit(self, event: ProgressEvent) -> None:
        try:
            with self._lock:
                if self._closed:
                    return
                self._apply_event(event)
                self._write_log(event)
                self._render_locked(event)
        except Exception:
            return

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._active = False
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._interactive:
                self._safe_write("\x1b[?25h\n")
            if self._log is not None:
                try:
                    self._log.flush()
                    self._log.close()
                except OSError:
                    pass
                self._log = None

    def _apply_event(self, event: ProgressEvent) -> None:
        if event.total:
            self._total = event.total
        if event.index:
            self._index = event.index
        if event.sample_id:
            self._sample_id = event.sample_id
        self._object_count = event.object_count
        if event.kind == "sample_start":
            self._active = True
            self._sample_started_at = event.timestamp
            self._stage_started_at = event.timestamp
            self._stage = event.stage or ProgressStage.READ_IMAGE
        elif event.kind == "stage_start":
            self._stage = event.stage
            self._stage_started_at = event.timestamp
        elif event.kind == "sample_end":
            self._active = False
            self._stage = event.stage or ProgressStage.FINALIZE
            self._completed += 1
            status = event.status or "unknown"
            self._counts[status] += 1
            duration = max(event.elapsed_seconds, 0.0)
            self._durations.append(duration)
            self._recent.append((event.sample_id, status, duration))
        elif event.kind == "batch_end":
            self._active = False

    def _heartbeat(self) -> None:
        while not self._stop.wait(self._refresh_seconds):
            try:
                with self._lock:
                    if self._active and not self._closed:
                        self._render_locked(None)
            except Exception:
                continue

    def _sample_gpu_locked(self) -> GpuSnapshot | None:
        now = self._clock()
        if (
            self._last_gpu_sample_at is None
            or now - self._last_gpu_sample_at >= self._telemetry_seconds
        ):
            self._last_gpu_sample_at = now
            try:
                self._gpu = self._telemetry.sample()
            except Exception:
                self._gpu = None
        return self._gpu

    def _render_locked(self, event: ProgressEvent | None) -> None:
        gpu = self._sample_gpu_locked()
        if self._interactive:
            self._safe_write(self._dashboard_text(gpu))
        elif event is not None:
            stage = _STAGE_LABELS.get(event.stage, "Batch")
            status = f" -> {event.status}" if event.status else ""
            cached = " cached" if event.cached else ""
            self._safe_write(
                f"[{self._wall_clock().strftime('%H:%M:%S')}] "
                f"{event.index}/{event.total} {event.sample_id} "
                f"{stage}{cached}{status} | {self._gpu_text(gpu)}\n"
            )

    def _dashboard_text(self, gpu: GpuSnapshot | None) -> str:
        now = self._clock()
        total = self._total
        percent = self._completed / total * 100 if total else 0.0
        stage_label = _STAGE_LABELS.get(self._stage, "Waiting")
        stage_number = _STAGE_POSITIONS.get(self._stage)
        stage_prefix = f"[{stage_number}/7] " if stage_number is not None else ""
        stage_elapsed = max(now - self._stage_started_at, 0.0) if self._active else 0.0
        image_elapsed = max(now - self._sample_started_at, 0.0) if self._active else 0.0
        eta = rolling_eta_seconds(
            self._durations, remaining=max(total - self._completed, 0)
        )
        eta_text = format_duration(eta) if eta is not None else "estimating"
        average = sum(self._durations) / len(self._durations) if self._durations else 0.0
        speed = 3600.0 / average if average > 0 else 0.0
        success = sum(
            self._counts[name]
            for name in ("safe", "hazard", "warning", "category_pending")
        )
        device = gpu.device_name if gpu is not None else "CUDA telemetry unavailable"
        recent = "\n".join(
            f" {sample_id:<12} {status:<16} {duration:>6.1f}s"
            for sample_id, status, duration in self._recent
        ) or " (none yet)"
        return (
            "\x1b[2J\x1b[H"
            f" VIDVIP FULL -> SSI                         GPU: {device}\n\n"
            f" Overall [{format_bar(self._completed, total)}] "
            f"{self._completed:,} / {total:,}   {percent:.2f}%\n"
            f" Current {self._sample_id or '-'}   objects: {self._object_count}\n"
            f" Stage {stage_prefix}{stage_label}   elapsed {format_duration(stage_elapsed)} "
            f"| image {format_duration(image_elapsed)}\n\n"
            f" Completed {self._completed:,}       Average speed {speed:.1f} images/hour\n"
            f" Success {success:,}         Elapsed {format_duration(now - self._run_started_at)}\n"
            f" Safe {self._counts['safe']:,} | Hazard {self._counts['hazard']:,} | "
            f"Warning {self._counts['warning']:,} | Pending {self._counts['category_pending']:,}\n"
            f" Filtered {self._counts['filtered']:,}        ETA {eta_text}\n"
            f" Rejected {self._counts['rejected']:,}\n"
            f" Failed {self._counts['failed']:,}          {self._gpu_text(gpu)}\n\n"
            f" Recent results\n{recent}\n"
        )

    @staticmethod
    def _gpu_text(gpu: GpuSnapshot | None) -> str:
        if gpu is None:
            return "GPU N/A"
        return (
            f"GPU {gpu.utilization_percent:.0f}% | "
            f"VRAM {gpu.memory_used_mib / 1024:.1f}/"
            f"{gpu.memory_total_mib / 1024:.1f} GiB"
        )

    def _write_log(self, event: ProgressEvent) -> None:
        if self._log is None:
            return
        try:
            stage = event.stage.value if event.stage is not None else "batch"
            status = event.status or ""
            self._log.write(
                f"{self._wall_clock().isoformat(timespec='seconds')} "
                f"{event.kind} {event.index}/{event.total} {event.sample_id} "
                f"stage={stage} status={status} cached={event.cached} "
                f"elapsed={event.elapsed_seconds:.3f}\n"
            )
            self._log.flush()
        except OSError:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None

    def _safe_write(self, value: str) -> None:
        try:
            self._stream.write(value)
            self._stream.flush()
        except (OSError, ValueError):
            pass
