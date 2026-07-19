import sys
from pathlib import Path

import _pytest.tmpdir
from _pytest.pathlib import rm_rf


DISTANCE_ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(DISTANCE_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(DISTANCE_ANALYSIS_ROOT))


_ORIGINAL_CLEANUP_DEAD_SYMLINKS = _pytest.tmpdir.cleanup_dead_symlinks
_ORIGINAL_MAKE_NUMBERED_DIR = _pytest.tmpdir.make_numbered_dir


def _permission_tolerant_cleanup_dead_symlinks(root: Path) -> None:
    try:
        _ORIGINAL_CLEANUP_DEAD_SYMLINKS(root)
    except PermissionError:
        # The managed Windows workspace can deny directory enumeration after
        # pytest creates basetemp. This affects cleanup only, not test results.
        return


def _workspace_readable_make_numbered_dir(
    root: Path, prefix: str, mode: int = 0o700
) -> Path:
    return _ORIGINAL_MAKE_NUMBERED_DIR(root, prefix, mode=0o755)


def _workspace_readable_getbasetemp(
    factory: _pytest.tmpdir.TempPathFactory,
) -> Path:
    if factory._basetemp is not None:
        return factory._basetemp
    if factory._given_basetemp is None:
        return _ORIGINAL_GETBASETEMP(factory)
    basetemp = factory._given_basetemp
    if basetemp.exists():
        rm_rf(basetemp)
    basetemp.mkdir(mode=0o755)
    factory._basetemp = basetemp.resolve()
    return factory._basetemp


_ORIGINAL_GETBASETEMP = _pytest.tmpdir.TempPathFactory.getbasetemp


def pytest_configure() -> None:
    _pytest.tmpdir.cleanup_dead_symlinks = _permission_tolerant_cleanup_dead_symlinks
    _pytest.tmpdir.make_numbered_dir = _workspace_readable_make_numbered_dir
    _pytest.tmpdir.TempPathFactory.getbasetemp = _workspace_readable_getbasetemp
