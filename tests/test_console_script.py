import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "command",
    (
        (str(Path(sys.executable).parent / "ynab"),),
        (sys.executable, "-m", "ynab_agent"),
    ),
    ids=("console-script", "python-module"),
)
def test_installed_entrypoints_show_help(command):
    result = subprocess.run(
        [*command, "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "YNAB Agent" in result.stdout
    assert "yeet-the-red" in result.stdout
    assert "wealth" in result.stdout
