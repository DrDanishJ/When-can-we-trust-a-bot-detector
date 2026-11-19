"""Step 0 (once) - download the LSABRE rewritten TwiBot-20 set into data/raw/lsabre.

    python scripts/00_get_lsabre.py

Needs git. The repository is ~700 MB; only its dataset/ folder (24 CSV files) is used.
"""
import _bootstrap  # noqa: F401

import shutil
import subprocess
from pathlib import Path

from ibd.config import cli

URL = "https://github.com/runi-cyber-ai/LSABRE"


def main():
    cfg, _ = cli(__doc__)
    target = Path(cfg["paths"]["lsabre"])
    if target.exists() and any(target.glob("*.csv")):
        print("already present:", target)
        return
    repo = target.parent if target.name == "dataset" else target
    repo.parent.mkdir(parents=True, exist_ok=True)
    if repo.exists():
        shutil.rmtree(repo)
    subprocess.run(["git", "clone", "--depth", "1", URL, str(repo)], check=True)
    print("done:", sorted(p.name for p in target.glob("*.csv"))[:3], "...")


if __name__ == "__main__":
    main()
