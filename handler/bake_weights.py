"""Build-time: bake the SeedVR2 weights into the image.

Runs during the build so the worker ships with its weights and needs **no network volume** at
runtime. That is not a build-speed decision: **RunPod network volumes exist in only a few
datacentres, which often do not overlap with GPU availability**, so baking is what lets the
endpoint run wherever there is a free card. On a worker whose whole design is about landing on a
card with enough memory, being able to use any datacentre's free cards is the point.

**Downloaded directly from HuggingFace rather than through SeedVR2's own downloader**, which
imports the model registry and with it the DiT/VAE torch classes and their optional attention
backends — a chain that can fail at build time for reasons unrelated to the files. Downloading
plain files avoids it. The repo id and filenames mirror `src/utils/model_registry.py` at the
pinned commit, and the destination is what the runtime pipeline reads.

**One model per image**, and which one is `--build-arg SEEDVR2_MODEL=…`. The weights are too
large to bake two, so a different checkpoint is a different build of this repo. Anything not
baked would download at runtime on first use and, with no volume, re-download on every cold
start — a per-job cost, billed, on a path nobody chose.
"""

import os
import sys

from huggingface_hub import hf_hub_download

REPO = "numz/SeedVR2_comfyUI"

#: Shared across every DiT variant, so it is baked whichever checkpoint is selected.
VAE_FILE = "ema_vae_fp16.safetensors"

model_dir = os.environ["SEEDVR2_MODEL_DIR"]
dit_file = os.environ.get("SEEDVR2_MODEL", "seedvr2_ema_7b_fp16.safetensors")

os.makedirs(model_dir, exist_ok=True)

total = 0.0
for filename in (dit_file, VAE_FILE):
    path = hf_hub_download(repo_id=REPO, filename=filename, local_dir=model_dir)
    size_gb = os.path.getsize(path) / (1024 ** 3)
    total += size_gb
    print("baked {} -> {} ({:.2f} GB)".format(filename, path, size_gb), flush=True)

# Reported because image size is a cold-start cost and a container-disk cost, and CF is waiting
# on both figures (handoff §9). A number printed by the build is one nobody has to estimate.
print("baked {:.2f} GB into {} -> {}".format(total, model_dir, sorted(os.listdir(model_dir))),
      flush=True)

if not os.path.isfile(os.path.join(model_dir, dit_file)):
    print("the requested checkpoint is not present after the download", file=sys.stderr)
    raise SystemExit(1)
