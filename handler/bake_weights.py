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
import shutil
import sys
import urllib.request
import zipfile

from huggingface_hub import hf_hub_download

REPO = "numz/SeedVR2_comfyUI"

#: **RIFE, and it is baked on EVERY variant including the weightless one.** `BAKE_WEIGHTS=0`
#: means "no SeedVR2 checkpoint" — route C is the image that has no upscaler, and it is precisely
#: the image that must interpolate. Fetched rather than vendored because it pins the same way
#: SeedVR2's does, and baked rather than downloaded at runtime for `bake_weights`'s own reason:
#: a fresh worker on a lazily-streaming host paid 6m06s and 12m29s for weights it had to read.
#:
#: **The archive carries the model CODE as well as the weights**, which is stronger than what
#: SeedVR2 gets: `RIFE_HDv3.Model` — the class the pipeline constructs — is inside the same zip
#: as `flownet.pkl`, so the definition is pinned by the hash of its own weights. There is no
#: second pin to keep in step.
#:
#: **`flownet.pkl` is a pickle and `torch.load` on a pickle executes code**, unlike SeedVR2's
#: safetensors. The hash below makes it the SAME pickle on every build; it does not make it an
#: inert one. Recorded rather than mitigated (CF, 2026-08-23), so nobody reads "hash-asserted"
#: as "safe to load from anywhere".
RIFE_REPO = "hzwer/RIFE"
RIFE_REVISION = "01fdc7e97404120c243c3ea7b427046e5dc7643e"
RIFE_ARCHIVE = "RIFEv4.26_0921.zip"
RIFE_ARCHIVE_BYTES = 22869906
RIFE_ARCHIVE_SHA256 = "1fa9b9cda3d9b8c3e301359e2595960902f97bf926c08598b0e9957a3f3f760e"

#: The four files the pipeline needs, by their name inside the archive's single directory, with
#: the size and sha256 of each. Everything else in the zip — a `.DS_Store`, a `__pycache__` of
#: another Python's bytecode, and Finder's `__MACOSX` shadows — is dropped rather than shipped.
#: **The archive is NOT self-sufficient, which is the thing to know about it.** `RIFE_HDv3.py`
#: opens with `from model.warplayer import warp` and `from model.loss import *` — a package that
#: lives in the Practical-RIFE *repository* and not in the weights zip. The reference script hides
#: this by taking a whole checkout as `--rife-dir`; an image that baked only the archive would
#: import-error on the first interpolation, having verified four hashes on the way.
#:
#: Two files, both pure Python, both tiny, pinned by commit rather than by tag. Raw file bytes at
#: a commit sha are stable in a way a repository tarball is not, so the same assertion the weights
#: get applies here. `loss.py` needs torchvision, which the base image already carries — its
#: `VGGPerceptualLoss` would fetch VGG19 weights, but nothing constructs it: `RIFE_HDv3.Model`
#: has that line commented out upstream, so no second download hides in this import.
RIFE_SOURCE_COMMIT = "17d8c7a1005b37f4c97bfee04e316aaec7fdc536"
RIFE_SOURCE_URL = ("https://raw.githubusercontent.com/hzwer/Practical-RIFE/{}/model/{}")
RIFE_SOURCE_FILES = {
    "warplayer.py": (
        1058, "eed94da2f2e8056fa0ceabed88b87fedf25ec849494991a956b9f2cbad33632c"),
    "loss.py": (
        4641, "9e4679cd685a37add8d8bb4a963b9822df0e1d344b82d01f975fc3426c8fc77a"),
}

RIFE_MEMBERS = {
    "flownet.pkl": (
        24636301, "45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f"),
    "RIFE_HDv3.py": (
        3101, "81bbd0648e499de79e44768d284005d9d57d0f6eb7c30adae407f22675055730"),
    "IFNet_HDv3.py": (
        6433, "655b4c772b037967b86c2dd31c8fa3b5323b79dd9a0e0088708d89149bbc8a32"),
    "refine.py": (
        3510, "0c5698b4a05b9f6ab551740575c1c35e248e5b1829bab6445186081ebe15f032"),
}

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

def verify(path, label, want_size, want_sha):
    """Size then sha256, exiting on either. Size first because it is free and a truncated
    download is the common failure; the hash then says the bytes are the ones measured."""
    size = os.path.getsize(path)
    if size != want_size:
        sys.exit("{}: expected {} bytes, got {}. The pin resolved to different content than the "
                 "calibration measured.".format(label, want_size, size))
    got_sha = sha256_of(path)
    if got_sha != want_sha:
        sys.exit("{}: sha256 mismatch.\n  expected {}\n  got      {}\nThe pin resolved to "
                 "different content than the calibration measured.".format(
                     label, want_sha, got_sha))
    print("verified {} {} bytes sha256 {}".format(label, size, got_sha), flush=True)


def bake_seedvr2():
    """The SeedVR2 checkpoint and the shared VAE. **Skipped on the route-C image.**"""
    model_dir = os.environ["SEEDVR2_MODEL_DIR"]
    dit_file = os.environ.get("SEEDVR2_MODEL", "seedvr2_ema_7b_fp16.safetensors")
    os.makedirs(model_dir, exist_ok=True)

    total = 0.0
    for filename in (dit_file, VAE_FILE):
        path = hf_hub_download(repo_id=REPO, filename=filename, local_dir=model_dir,
                               revision=REVISION)
        size = os.path.getsize(path)
        size_gb = size / (1024 ** 3)
        total += size_gb

        # **The assertion is what gives the recorded hash a job.** A number written down and
        # never compared against is the shape of a fact that rots unnoticed; this is the
        # comparison. Size first because it is free and a truncated download is the common
        # failure, then the hash.
        expected = EXPECTED.get(filename)
        if expected is None:
            print("WARNING: {} has no recorded size or hash — baked UNVERIFIED. The calibration "
                  "key names the 7B model and the shared VAE; a different --build-arg "
                  "SEEDVR2_MODEL is a different checkpoint and nothing here can vouch for it."
                  .format(filename), flush=True)
        else:
            verify(path, filename, *expected)

        print("baked {} -> {} ({:.2f} GB)".format(filename, path, size_gb), flush=True)

    # Reported because image size is a cold-start cost and a container-disk cost, and CF is
    # waiting on both figures (handoff §9). A number printed by the build is one nobody has to
    # estimate.
    print("baked {:.2f} GB into {} -> {}".format(total, model_dir, sorted(os.listdir(model_dir))),
          flush=True)

    if not os.path.isfile(os.path.join(model_dir, dit_file)):
        print("the requested checkpoint is not present after the download", file=sys.stderr)
        raise SystemExit(1)


def bake_rife():
    """Practical-RIFE's `train_log` — the model code and its weights, from one pinned archive.

    **Baked on every variant, including the one with no SeedVR2.** Route C is the image without
    an upscaler and it is exactly the image that has to interpolate, so this is not conditional
    on `BAKE_WEIGHTS`.

    The archive is verified whole before anything is extracted, and each extracted file is
    verified again after. Two checks rather than one because they answer different questions: the
    first says the pin resolved to the bytes the calibration measured, the second says the
    extraction produced the files those bytes contain — a truncated write and a wrong download
    are different failures and only the first is visible upstream.
    """
    rife_dir = os.environ["RIFE_MODEL_DIR"]
    train_log = os.path.join(rife_dir, "train_log")
    os.makedirs(train_log, exist_ok=True)

    archive = hf_hub_download(repo_id=RIFE_REPO, filename=RIFE_ARCHIVE, local_dir=rife_dir,
                              revision=RIFE_REVISION)
    verify(archive, RIFE_ARCHIVE, RIFE_ARCHIVE_BYTES, RIFE_ARCHIVE_SHA256)

    with zipfile.ZipFile(archive) as bundle:
        # **Named members, not `extractall`.** The zip was built on a Mac and carries a
        # `.DS_Store`, a `__pycache__` of another Python's bytecode, and a `__MACOSX` shadow of
        # every entry. `extractall` would ship all of it, and stale `.pyc` files beside their
        # sources are a way to run code nobody can see. It is also the answer to zip-slip: a
        # member is looked up by the name we asked for, so a crafted path cannot escape.
        members = {os.path.basename(name): name
                   for name in bundle.namelist() if not name.startswith("__MACOSX/")}
        for wanted, (want_size, want_sha) in sorted(RIFE_MEMBERS.items()):
            inside = members.get(wanted)
            if inside is None:
                sys.exit("{} is not in {} — the archive's shape changed under a pinned hash, "
                         "which should be impossible".format(wanted, RIFE_ARCHIVE))
            destination = os.path.join(train_log, wanted)
            with bundle.open(inside) as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst)
            verify(destination, wanted, want_size, want_sha)
            print("baked {} -> {} ({} bytes)".format(
                wanted, destination, os.path.getsize(destination)), flush=True)

    os.remove(archive)

    # **The `model` package the archive does not carry.** Fetched from the pinned commit and
    # verified the same way, into a sibling of `train_log` so the two names the vendored code
    # imports — `train_log.*` and `model.*` — are both reachable from one directory on the path.
    model_pkg = os.path.join(rife_dir, "model")
    os.makedirs(model_pkg, exist_ok=True)
    for name, (want_size, want_sha) in sorted(RIFE_SOURCE_FILES.items()):
        destination = os.path.join(model_pkg, name)
        url = RIFE_SOURCE_URL.format(RIFE_SOURCE_COMMIT, name)
        with urllib.request.urlopen(url, timeout=120) as response, \
                open(destination, "wb") as handle:
            shutil.copyfileobj(response, handle)
        verify(destination, "model/" + name, want_size, want_sha)
        print("baked model/{} -> {}".format(name, destination), flush=True)

    print("baked RIFE into {} -> train_log {} · model {}".format(
        rife_dir, sorted(os.listdir(train_log)), sorted(os.listdir(model_pkg))), flush=True)


# **RIFE always, SeedVR2 only when asked for.** `BAKE_WEIGHTS=0` builds the route-C image, which
# has no upscaler by design and still needs its interpolator — so the flag gates one and not the
# other. The Dockerfile invokes this unconditionally and the decision is made here, where the two
# fetches and their assertions already live, rather than in shell.
if os.environ.get("BAKE_WEIGHTS") == "0":
    print("BAKE_WEIGHTS=0 — no SeedVR2 checkpoint; RIFE is baked regardless (contract 6b)",
          flush=True)
else:
    bake_seedvr2()

bake_rife()
