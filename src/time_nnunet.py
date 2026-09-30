#!/usr/bin/env python3
"""Wall-clock nnU-Net inference per case (default mirroring TTA), from the difference between 10-case and
1-case runs so that process start-up and model loading are excluded."""
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, OUT, save_json  # noqa: E402
from nnunet_t2 import DS, RAW, SCRIPTS, TRAINER, env  # noqa: E402

src = sorted((RAW / DS / "imagesTs").glob("*.nii.gz"))
tmp = CACHE / "nn_timing"
res = {}
for n in (1, 10):
    i_dir, o_dir = tmp / f"in{n}", tmp / f"out{n}"
    shutil.rmtree(tmp, ignore_errors=True) if n == 1 else None
    i_dir.mkdir(parents=True, exist_ok=True)
    o_dir.mkdir(parents=True, exist_ok=True)
    for f in src[:n]:
        shutil.copy2(f, i_dir / f.name)
    t0 = time.perf_counter()
    subprocess.run([str(SCRIPTS / "nnUNetv2_predict.exe"), "-i", str(i_dir), "-o", str(o_dir), "-d", "502", "-c",
                    "3d_fullres", "-f", "0", "-tr", TRAINER, "-chk", "checkpoint_final.pth", "-npp", "1", "-nps", "1"],
                   env=env(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res[n] = time.perf_counter() - t0
per_case = (res[10] - res[1]) / 9
save_json(OUT / "timing_nnunet.json", {"total_1_case_s": res[1], "total_10_cases_s": res[10],
                                       "per_case_s": per_case, "tta": "default mirroring"})
print(res, "per case", per_case)
