import unittest
import subprocess
from pathlib import Path


STRUCTURED_FIELD_DIR = Path(__file__).resolve().parents[2]
BAT_PATH = STRUCTURED_FIELD_DIR / "启动Structured-field Evaluation on an Independently Annotated Subset标注工具.bat"


class LauncherTests(unittest.TestCase):
    def test_windows_batch_file_uses_crlf_only(self):
        content = BAT_PATH.read_bytes()

        self.assertNotIn(b"\n", content.replace(b"\r\n", b""))

    def test_batch_passes_launch_script_to_python(self):
        content = BAT_PATH.read_text(encoding="ascii")

        self.assertIn('"%PYTHON%" "%TOOL%\\launch.py"', content)

    def test_launcher_bypasses_old_cached_index(self):
        content = (STRUCTURED_FIELD_DIR / "annotation_tool" / "launch.py").read_text(encoding="utf-8")

        self.assertIn("/?launch={time.time_ns()}", content)

    def test_batch_runs_from_path_containing_parentheses(self):
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", str(BAT_PATH), "--check"],
            cwd=STRUCTURED_FIELD_DIR,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Environment check passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
