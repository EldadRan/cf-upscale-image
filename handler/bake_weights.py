"""Build-time: bake the SeedVR2 weights into the image.

Runs during the build so the worker ships with its weights and needs **no network volume** at
runtime. That is not a build-speed decision: **RunPod network volumes exist in only a few
datacentres, which often do not overlap with GPU availability**, so baking is what lets the
endpoint run wherever there is a free card. On a worker whose whole design is about landing on a
card with enough memory, being able to use any datacentre's free cards is the point.

**Downloaded directly from HuggingFace rather than through SeedVR2's own downloader**, which adds
its own validation and cache on top of the fetch. The filenames mirror
`src/utils/model_registry.py` at the pinned commit, and the destination is what the runtime
pipeline reads.

**The repository each file comes from is READ from that registry** (`get_model_repo`, X1b, CF
2026-11-22), not written down a second time here: the 7B fp8 mixed checkpoint lives in
`AInVFX/`, the fp16 one and the VAE in `numz/`, and a hard-coded repo would be a second copy of a
fact the vendored tree already holds. Importing the registry pulls the DiT/VAE torch classes, which
used to be the reason to avoid it — but the Dockerfile's `import inference_cli` check above the
bake already imports that chain and stops the build if it cannot, so the risk is paid before this
script runs.

**One model per image**, and which one is `--build-arg SEEDVR2_MODEL=…`. The weights are too
large to bake two, so a different checkpoint is a different build of this repo. Anything not
baked would download at runtime on first use and, with no volume, re-download on every cold
start — a per-job cost, billed, on a path nobody chose.
"""

import hashlib
import os
import sys

from huggingface_hub import hf_hub_download

#: **The weights are pinned by commit, not by branch — one commit per repository.** Without this
#: the build took whatever `main` held on the day, and an upstream content change would have been
#: invisible: same filename, same size class, same green build, different numbers out of every
#: coefficient in the calibration. A commit sha cannot be force-pushed to different contents, so
#: this is the strong half of the pin; the hashes below are what make it checkable from inside the
#: build. **A repository with no revision here is refused**, never fetched from `main`.
#:
#: `AInVFX/`'s commit was read from HuggingFace on 2026-11-22: its tree at that commit lists the
#: fp8 mixed file at the size and sha256 `EXPECTED` records.
REVISIONS = {
    "numz/SeedVR2_comfyUI": "09ced71023636e9bc8cdf9cdecfb2625d1e691e8",
    "AInVFX/SeedVR2_comfyUI": "ac66d6d98fa49975d893b58c55bff7677191c862",
}

#: filename -> (bytes, sha256).
#:
#: **This is a COPY, and its home is `registry-v1.json`'s `calibration_key` in the private
#: project repository** — that is where the calibration records which weights it measured. The
#: copy exists because the build cannot read that file: the docker context is `./handler` in
#: THIS repository (see the workflow's `context:`), and the project repository is not checked
#: out on the runner at all. Anything that changes there has to be brought here by hand until
#: something generates this block.
#:
#: The fp16 pair sums to 15.81 GiB, which is what `deployment.md` records as the baked weight
#: size — so those identify the files in the live image rather than an intention.
#:
#: **The fp8 mixed entry is NOT a calibration copy** (X1b, CF 2026-11-22). It is the vendored
#: registry's 7B fp8 (`model_registry.py`, sha256 there) with HuggingFace's byte count, for the
#: Vast stills build; nothing has been calibrated on it. No other checkpoint is added.
EXPECTED = {
    "seedvr2_ema_7b_fp16.safetensors": (
        16479334424,
        "7b8241aa957606ab6cfb66edabc96d43234f9819c5392b44d2492d9f0b0bbe4a",
    ),
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors": (
        8466296338,
        "3d68b5ec0b295ae28092e355c8cad870edd00b817b26587d0cb8f9dd2df19bb2",
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

# **REFUSED HERE, BEFORE ANYTHING IS FETCHED (CF's ruling, 2026-08-28).** Naming the shared VAE as
# the DiT is the one build-arg value that makes this script ship an image with no model in it and
# report success. The whole chain is below and every link works as written:
#
#   `files` collapses to a single entry, so only the VAE is ever downloaded;
#   the size and hash assertions pass, because the VAE really is the VAE;
#   the total is right, the listing is right, and every printed line is true;
#   and the final guard asks `isfile(model_dir/dit_file)` -- which is the VAE's own path, present
#   because it was just fetched. **The guard is satisfied BY the file whose presence is the
#   evidence that the DiT is missing.**
#
# **CF ruled the fix here rather than at that guard**, and the reason is that the guard is not
# wrong. `dit_file` names a checkpoint; asking whether the named checkpoint arrived is exactly
# the right question, and it answers correctly for every input except one that is a category
# error before it is a bug. Hardening it would encode "the DiT must not be the VAE" as a
# post-download detection, which treats a nonsensical request as a configuration to be caught
# late instead of one to be rejected on sight. A build-arg naming the VAE as the model is never
# a thing anyone meant.
if dit_file == VAE_FILE:
    # **`dit_file`, not `VAE_FILE`, even though this branch proves them equal.** Reporting the
    # constant back instead of what the caller actually passed is correct only for as long as the
    # comparison stays exact string equality — and the obvious next change here is to loosen it.
    # A message that echoes a constant is a number belonging to something else.
    sys.exit(
        "SEEDVR2_MODEL={0} names the shared VAE as the DiT checkpoint, which is not a "
        "configuration this image can have.\n"
        "  The VAE is baked unconditionally alongside whichever DiT is selected, so this would "
        "download one file, verify it correctly, and produce an image with no model in it -- and "
        "every check downstream would pass, because the checkpoint's own presence test would be "
        "satisfied by the VAE sitting at that path.\n"
        "  Pass a DiT checkpoint name, or omit the build-arg for the default.".format(dit_file))

# The vendored registry is a package under the SeedVR2 tree, imported as `src.utils...` the way
# the Dockerfile's own checks import it.
sys.path.insert(0, os.environ["SEEDVR2_DIR"])
from src.utils.model_registry import get_model_repo  # noqa: E402

os.makedirs(model_dir, exist_ok=True)

total = 0.0
# **Two files, always, because the collision that made this a branch is now refused above.**
#
# This line used to read `(dit_file,) if dit_file == VAE_FILE else (dit_file, VAE_FILE)`, and the
# history is kept because the collision it de-duplicated is real and the reason it mattered is not
# obvious: `dit_file` is whatever the build-arg names and `VAE_FILE` is a constant, so the two
# could be the same string, and the tuple then iterated it twice. `hf_hub_download` short-circuits
# on the second pass, so nothing was fetched twice and nothing failed — **only `total` was wrong,
# too large by one file, in the number CF reads for cold-start and container-disk cost.**
#
# **De-duplicating was the wrong layer, and that is the whole finding.** It made the collision
# survivable and therefore silent, which is what let it go on to produce a model-less image that
# reported success. The refusal above removes the collision instead of accommodating it, so this
# branch has nothing left to decide.
files = (dit_file, VAE_FILE)
for filename in files:
    repo = get_model_repo(filename)
    if repo not in REVISIONS:
        sys.exit("{}: the vendored registry places it in {}, which has no pinned revision here. "
                 "Pin one in REVISIONS rather than fetching whatever its main holds.".format(
                     filename, repo))
    print("fetching {} from {} @ {}".format(filename, repo, REVISIONS[repo]), flush=True)
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=model_dir,
                           revision=REVISIONS[repo])
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
