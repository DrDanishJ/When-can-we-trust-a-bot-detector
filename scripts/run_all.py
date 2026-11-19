"""Run steps 1-12 in order with the same --config / --set arguments.

    python scripts/run_all.py --config configs/smoke.yaml       # quick check on the samples
    python scripts/run_all.py                                    # full run (hours)
    python scripts/run_all.py --from 4                           # resume from step 4
    python scripts/run_all.py --to 8                             # stop after step 8
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_build_features.py", "02_train_eval.py", "03_explain.py", "04_shortcut_audit.py",
         "05_conformal.py", "06_reliability.py", "07_shift_llm.py", "08_trust_gate.py",
         "09_conformal_checks.py", "10_shortcut_audit_ext.py", "11_shift_ext.py", "12_trust_gate_ext.py"]


def main():
    argv = sys.argv[1:]
    start, stop = 1, len(STEPS)
    for flag in ("--from", "--to"):
        if flag in argv:
            i = argv.index(flag)
            val = int(argv[i + 1])
            argv = argv[:i] + argv[i + 2:]
            start, stop = (val, stop) if flag == "--from" else (start, val)
    for k, step in enumerate(STEPS, 1):
        if k < start or k > stop:
            continue
        print(f"\n===== step {k}: {step} =====", flush=True)
        r = subprocess.run([sys.executable, str(HERE / step), *argv])
        if r.returncode != 0:
            sys.exit(f"step {k} ({step}) failed with exit code {r.returncode}")


if __name__ == "__main__":
    main()
