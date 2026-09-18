"""S1 follow-up chain: high-IoU AP breakdown for the corrected soft_v2 seeds, then the B3 paired bootstrap
with the full-vs-soft_v2 contrast. Runs detached; B3 lines go to queue_progress.log (monitored)."""
import subprocess, sys, datetime as dt
from pathlib import Path
HERE = Path(__file__).resolve().parent
LOG = HERE / "logs" / "s1_post.log"
PY = sys.executable
def log(m):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] S1-post {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f: f.write(line + "\n")
    with (HERE / "queue_progress.log").open("a", encoding="utf-8") as f: f.write(line + "\n")
steps = [("highiou", [PY, "-u", r"<PAPER_ROOT>/experiments/n2_soft_teacher_highiou/eval_highiou.py"]),
         ("b3", [PY, "-u", str(HERE / "b3_subterm_bootstrap.py"), "--phase", "all"])]
for name, cmd in steps:
    log(f"{name} start")
    with (HERE / "logs" / f"s1_post_{name}.log").open("a", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=r"<YOLO11_ROOT>", stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"{name} done rc={rc}")
    if rc != 0:
        log("chain aborted"); sys.exit(rc)
log("chain complete")
