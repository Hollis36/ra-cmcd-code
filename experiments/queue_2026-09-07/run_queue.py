"""Sequential experiment queue for stages E2, B3 (export), A1, B2, D1; the B3 bootstrap (CPU) is
launched detached after its export phase so it overlaps with training. Every stage is a subprocess;
a failing stage is logged and the queue moves on. Progress: queue_progress.log; stdout/stderr per
stage under logs/. Resumable: rerun and finished work is skipped by the stage scripts themselves.
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
LOG = HERE / "queue_progress.log"
STAGES = [
    ("E2", [PY, "-u", str(HERE / "e2_strata_recall.py")]),
    ("B3-export", [PY, "-u", str(HERE / "b3_subterm_bootstrap.py"), "--phase", "export"]),
    ("S1", [PY, "-u", str(HERE / "s1_students.py")]),
    ("A1", [PY, "-u", str(HERE / "a1_students.py")]),
    ("B2", [PY, "-u", str(HERE / "b2_students.py")]),
    ("D1", [PY, "-u", str(HERE / "d1_pools.py")]),
]


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] QUEUE {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    (HERE / "logs").mkdir(exist_ok=True)
    only = set(sys.argv[1:])
    for name, cmd in STAGES:
        if only and name not in only:
            continue
        log(f"stage {name} start")
        with (HERE / "logs" / f"stage_{name}.log").open("a", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=r"<YOLO11_ROOT>", stdout=f, stderr=subprocess.STDOUT, creationflags=0x08000000).returncode
        log(f"stage {name} {'done' if rc == 0 else f'FAILED rc={rc}'}")
        if name == "B3-export" and rc == 0:
            out = (HERE / "logs" / "stage_B3-bootstrap.log").open("a", encoding="utf-8")
            subprocess.Popen([PY, "-u", str(HERE / "b3_subterm_bootstrap.py"), "--phase", "bootstrap"], cwd=r"<YOLO11_ROOT>", stdout=out,
                             stderr=subprocess.STDOUT, creationflags=0x00000008 | 0x00000200, close_fds=True)
            log("stage B3-bootstrap launched detached (CPU)")
    log("queue complete")


if __name__ == "__main__":
    main()
