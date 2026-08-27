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

import hashlib
import os
import sys

from huggingface_hub import hf_hub_download

REPO = "numz/SeedVR2_comfyUI"

#: **The weights are pinned by commit, not by branch.** Without this the build took whatever
#: `main` held on the day, and an upstream content change would have been invisible: same
#: filename, same size class, same green build, different numbers out of every coefficient in
#: the calibration. A commit sha cannot be force-pushed to different contents, so this is the
#: strong half of the pin; the hashes below are what make it checkable from inside the build.
REVISION = "09ced71023636e9bc8cdf9cdecfb2625d1e691e8"

#: filename -> (bytes, sha256).
#:
#: **This is a COPY, and its home is `registry-v1.json`'s `calibration_key` in the private
#: project repository** — that is where the calibration records which weights it measured. The
#: copy exists because the build cannot read that file: the docker context is `./handler` in
#: THIS repository (see the workflow's `context:`), and the project repository is not checked
#: out on the runner at all. Anything that changes there has to be brought here by hand until
#: something generates this block.
#:
#: The pair sums to 15.81 GiB, which is what `deployment.md` records as the baked weight size —
#: so these identify the files in the live image rather than an intention.
EXPECTED = {
    "seedvr2_ema_7b_fp16.safetensors": (
        16479334424,
        "7b8241aa957606ab6cfb66edabc96d43234f9819c5392b44d2492d9f0b0bbe4a",
    ),
    "ema_vae_fp16.safetensors": (
        501324814,
        "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1",
    ),
}


def sha256_of(path):
    """Streamed, because the DiT is 15.3 GiB and the runner has no room to hold it twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

#: Shared across every DiT variant, so it is baked whichever checkpoint is selected.
VAE_FILE = "ema_vae_fp16.safetensors"

model_dir = os.environ["SEEDVR2_MODEL_DIR"]
dit_file = os.environ.get("SEEDVR2_MODEL", "seedvr2_ema_7b_fp16.safetensors")

os.makedirs(model_dir, exist_ok=True)

total = 0.0
# **De-duplicated, because `--build-arg SEEDVR2_MODEL=ema_vae_fp16.safetensors` makes these one
# file.** `dit_file` is whatever the build-arg names and `VAE_FILE` is a constant, so the two can
# be the same string — and the tuple then iterated it twice. `hf_hub_download` short-circuits on
# the second pass, so nothing was fetched twice and nothing failed: **only `total` was wrong, too
# large by one file, in the number CF reads for cold-start and container-disk cost.** It failed in
# the direction that does not announce itself.
#
# Written out rather than `dict.fromkeys((dit_file, VAE_FILE))`, which is shorter and hides why a
# collision is possible at all. The reader needs to see the cause, not just the guard.
files = (dit_file,) if dit_file == VAE_FILE else (dit_file, VAE_FILE)
for filename in files:
    path = hf_hub_download(repo_id=REPO, filename=filename, local_dir=model_dir,
                           revision=REVISION)
    size = os.path.getsize(path)
    size_gb = size / (1024 ** 3)
    total += size_gb

    # **The assertion is what gives the recorded hash a job.** A number written down and never
    # compared against is the shape of a fact that rots unnoticed; this is the comparison. Size
    # first because it is free and a truncated download is the common failure, then the hash.
    expected = EXPECTED.get(filename)
    if expected is None:
        print("WARNING: {} has no recorded size or hash — baked UNVERIFIED. The calibration key "
              "names the 7B model and the shared VAE; a different --build-arg SEEDVR2_MODEL is "
              "a different checkpoint and nothing here can vouch for it.".format(filename),
              flush=True)
    else:
        want_size, want_sha = expected
        if size != want_size:
            sys.exit("{}: expected {} bytes, got {}. The pin resolved to different content than "
                     "the calibration measured.".format(filename, want_size, size))
        got_sha = sha256_of(path)
        if got_sha != want_sha:
            sys.exit("{}: sha256 mismatch.\n  expected {}\n  got      {}\nThe pin resolved to "
                     "different content than the calibration measured.".format(
                         filename, want_sha, got_sha))
        print("verified {} {} bytes sha256 {}".format(filename, size, got_sha), flush=True)

    print("baked {} -> {} ({:.2f} GB)".format(filename, path, size_gb), flush=True)

# Reported because image size is a cold-start cost and a container-disk cost, and CF is waiting
# on both figures (handoff §9). A number printed by the build is one nobody has to estimate.
print("baked {:.2f} GB into {} -> {}".format(total, model_dir, sorted(os.listdir(model_dir))),
      flush=True)

if not os.path.isfile(os.path.join(model_dir, dit_file)):
    print("the requested checkpoint is not present after the download", file=sys.stderr)
    raise SystemExit(1)
