import os
import subprocess
import sys

import pytest

from modeltuner import safety


def test_lock_is_exclusive_and_released(tmp_path):
    path = tmp_path / "RUN.lock"
    with safety.RunLock(path):
        assert path.exists()
        # a lock held by another live process refuses
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            path.write_text(f"{other.pid} x", encoding="utf-8")
            with pytest.raises(safety.SafetyError):
                with safety.RunLock(path):
                    pass
        finally:
            other.kill()
            other.wait()
        path.write_text(f"{os.getpid()} x", encoding="utf-8")
    assert not path.exists()


def test_a_stale_lock_is_taken_over(tmp_path):
    path = tmp_path / "RUN.lock"
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    path.write_text(f"{p.pid} old", encoding="utf-8")      # its process has exited
    with safety.RunLock(path) as lock:
        assert lock.held
        assert path.read_text(encoding="utf-8").split()[0] == str(os.getpid())
