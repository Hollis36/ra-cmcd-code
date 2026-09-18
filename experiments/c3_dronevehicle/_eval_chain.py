import subprocess, sys
from pathlib import Path
here = Path(r'<PAPER_ROOT>/experiments/c3_dronevehicle')
for script in ('phase2_eval.py', 'phase2_bootstrap_global.py'):
    with (here / ('chain_%s.log' % script)).open('a', encoding='utf-8') as f:
        rc = subprocess.run([sys.executable, '-u', str(here / script)], cwd=str(here), stdout=f,
                            stderr=subprocess.STDOUT, creationflags=0x08000000).returncode
    print(script, 'rc', rc, flush=True)
    if rc != 0:
        break
