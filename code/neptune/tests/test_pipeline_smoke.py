import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_whole_pipeline_on_synthetic(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_pipeline.py"), "--work", str(tmp_path),
                        "--threads", "2", "--skip-baselines"], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "SMOKE OK" in r.stdout
    reps = list((tmp_path / "out" / "reports" / "smoke_claim").rglob("h1_report.md"))
    assert len(reps) == 1 and "Verdict" in reps[0].read_text()
