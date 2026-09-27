"""Run with the installed project interpreter: python tests/check_scaffold.py."""

import subprocess
import sys
import tempfile


with tempfile.TemporaryDirectory() as directory:
    help_result = subprocess.run(
        [sys.executable, "-m", "carpet_ad.cli", "--help"],
        cwd=directory, capture_output=True, text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "robustness" in help_result.stdout
    unavailable = subprocess.run(
        [sys.executable, "-m", "carpet_ad.cli", "robustness"],
        cwd=directory, capture_output=True, text=True,
    )
    assert unavailable.returncode == 2, unavailable.stdout
print("PASS: installed CLI works outside the repository; missing required arguments fail.")
