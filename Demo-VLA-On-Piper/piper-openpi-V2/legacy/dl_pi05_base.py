"""Download the pi05_base JAX checkpoint (~12.4 GB) from the public
openpi-assets bucket to the local cache. Anonymous access."""
import os
import shutil
import time

import gcsfs

SRC = "openpi-assets/checkpoints/pi05_base"
DST = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi05_base")

os.makedirs(os.path.dirname(DST), exist_ok=True)   # parent only (NOT DST -> avoids nesting)
if os.path.exists(DST):
    shutil.rmtree(DST)

fs = gcsfs.GCSFileSystem(token="anon")
print(f"downloading {SRC} -> {DST}", flush=True)
t = time.time()
fs.get(SRC, DST, recursive=True)

# flatten if it nested into DST/pi05_base
inner = os.path.join(DST, "pi05_base")
if os.path.isdir(inner):
    for x in os.listdir(inner):
        shutil.move(os.path.join(inner, x), os.path.join(DST, x))
    os.rmdir(inner)

print("layout:", sorted(os.listdir(DST)), flush=True)
sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fns in os.walk(DST) for f in fns)
print(f"DONE {sz/1e9:.2f} GB in {(time.time()-t)/60:.1f} min", flush=True)
