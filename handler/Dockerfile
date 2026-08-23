# CF SeedVR2 upscale worker.
#
# **Ordered by how often each step changes**, which is what makes the build cache hit: base image,
# system packages, pinned third-party source, Python dependencies, the dependency import check,
# **weights**, then this repo's code last. A handler edit rebuilds one small layer.
#
# **Any step inserted below the bake invalidates it and costs the whole ~17 GB download.** A
# `pip install` added in the wrong place is the mistake to watch for.
#
# Weights are baked rather than mounted because **RunPod network volumes exist in only a few
# datacentres, which often do not overlap with GPU availability** — baking is what lets the
# endpoint run wherever there is a free card. That is the reason; build speed is not.

# Python 3.11, stated in the tag rather than deduced from an Ubuntu release (docs/decisions.md
# 2.1). torch 2.8.0 rather than the image worker's 2.4.0: the diffusers==0.33.1 pin exists only
# for 2.4.0's infer_schema and goes with it.
#
# **This used to claim CUDA 12.8.1 "widens the card pool this worker's headroom comes from". It
# does the opposite**, and it was never measured — the base image's `cuda>=12.8` requirement is a
# *floor on the host driver*, so a newer CUDA admits fewer machines, not more. Measured when most
# of the fleet refused to start the container while the sibling worker's `cuda12.4.1` build ran on
# the same cards. The `NVIDIA_REQUIRE_CUDA` override below is what recovers the pool.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# The pinned SeedVR2 commit. The handler drives this exact source's entry points, so a moving
# checkout would move the worker's accepted values underneath it.
ARG SEEDVR2_COMMIT=4490bd1f482e026674543386bb2a4d176da245b9

# **One baked model per image, and which one is a build parameter.** The weights are too large to
# bake two, so a different checkpoint is a different build of this repo rather than a runtime
# choice — and `cf-upscale-worker` on a 3B variant is `--build-arg`, not a fork. CF records the
# value as `routing.provider_model`, per endpoint.
ARG SEEDVR2_MODEL=seedvr2_ema_7b_fp16.safetensors

# **The route-C image is this image without the weights** (contract §6b). Route C never loads
# SeedVR2, so a build that omits the checkpoint is not a stripped worker — it is the correct one
# for that route, and it does not pay a cold start for ~15 GiB it will never read.
#
# **It gates SeedVR2 and NOT RIFE** (CF, 2026-08-23). Route C is the image with no upscaler and
# it is precisely the image that must interpolate, so the interpolator is baked on every variant
# and the flag decides only whether a checkpoint rides with it. The shell no longer branches: the
# script is always invoked and reads this value itself, because the two fetches and their hash
# assertions already live there and a second decision point in shell is a second place to be
# wrong about which model is which.
#
# **`0` or `1`, and anything else fails the build.** An earlier draft baked on a mistyped value
# — fail-safe in isolation, but two ENV flags below derive from this one arg, and a value that is
# neither cannot set both honestly. A typo that stops a build costs minutes; a typo that ships an
# image whose flags disagree about its own bytes costs a campaign. Default 1, and the default
# path is the shell that ran before this arg existed: with the weights baked the RUN's contents
# are what they were.
# **The layer's DIGEST necessarily changes**
# — the instruction text is part of a layer's identity, so `if` here is a new layer even when it
# does exactly what the old one did. Contents identical, digest not; the two are different claims
# and only the first was ever available.
ARG BAKE_WEIGHTS=1

# A pinned static ffmpeg, not the distribution's. Ubuntu 22.04 ships 4.4, this repo is developed
# against 9.x, and the gap is not cosmetic: `-vsync` was removed in 9.x while `-fps_mode` arrived
# in 5.0, so a command written for one is invalid on the other (docs/decisions.md 3.3). Pinning
# one build for the container makes the encode path a known quantity instead of a distribution
# detail. It also guarantees libwebp, which every poster and every crop needs.
#
# **The release tag is a MONTH-END snapshot and must stay one.** BtbN keeps fourteen rolling
# dailies and retains month-end builds for years; the previous pin was a daily and started
# 404ing the day it aged out, which fails this RUN two minutes into a twenty-one minute build.
# A newer-looking daily tag is a fourteen-day fuse and, since the dailies have moved past this
# ffmpeg commit, also a toolchain change. Moving the tag while holding the asset name keeps the
# ffmpeg commit, the libavcodec version and the x265 build; it does not keep the bytes, which
# differ between releases. What holds the build steady is the capability assertion below.
ARG FFMPEG_RELEASE=autobuild-2026-07-31-14-10
ARG FFMPEG_ASSET=ffmpeg-n8.1.2-34-g9b6c8969e0-linux64-gpl-8.1

# **The base image's `cuda>=12.8` label is what kept this off most of RunPod's fleet.**
#
# `nvidia-container-cli` checks `NVIDIA_REQUIRE_CUDA` against the *host driver* before `runc`
# creates the process, so a host on 12.4 refuses the container with
# `unsatisfied condition: cuda>=12.8` and nothing of ours ever runs. Measured against the sibling
# worker on the same account and the same A40s: its `cuda12.4.1` base asks for `cuda>=12.4` and
# runs; this one asked for 12.8 and was refused. **The comment below claiming CUDA 12.8.1 widens
# the card pool is exactly backwards, and is corrected there.**
#
# 12.0 rather than removing the check: CUDA guarantees *minor version compatibility*, so an
# application built against 12.8 runs on any 12.x driver — that is the guarantee PyTorch itself
# relies on. Lowering the bound admits precisely 12.0–12.7, which is the range that guarantee
# covers. **Not `NVIDIA_DISABLE_REQUIRE=1`**, the usual suggestion: that removes the check
# altogether and would let an 11.x host start the container, turning a clean refusal into a
# confusing failure after the worker has been billed.
#
# The residual risk is that a host which still cannot run this now fails at model load rather than
# at container start. `hardware.read()` records the driver version on every job so that failure is
# one lookup rather than a hunt.
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.0"

# **Measured: 26.40 GiB of a 96 GB card held by the allocator and unusable.** A 2K-to-4K clip at
# the `resident` rung OOMed with 62.34 GiB actually allocated, 26.40 GiB reserved-but-unallocated,
# and 5.58 GiB free — failing on a single 19.78 GiB request inside the VAE decode. More than a
# quarter of the card was fragmentation, and PyTorch names this setting in the error text itself.
#
# `expandable_segments` lets the allocator grow a segment rather than pick from fixed-size blocks,
# which is exactly the shape of this workload: a few enormous, short-lived tensors of varying size
# per chunk. It is the largest single memory lever found so far, and it costs nothing to set.
#
# **Set here rather than in the handler** because the CUDA caching allocator reads it once, when
# it initialises — which happens on the first CUDA call, well before any of this repo's code could
# reliably set it. An environment variable in the image cannot be too late.
#
# Untested as of 2026-08-13: measured only that the fragmentation exists, not that this fixes it.
# The run that answers it is `fhd_source.mp4 --target 2160 --rung resident`, which OOMed here.
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ENV SEEDVR2_DIR=/app/SeedVR2 \
    SEEDVR2_MODEL_DIR=/app/models/SEEDVR2 \
    SEEDVR2_MODEL=${SEEDVR2_MODEL} \
    RIFE_MODEL_DIR=/app/models/RIFE \
    PYTHONUNBUFFERED=1

# ── system packages ─────────────────────────────────────────────────────────────────────────
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git libgl1 libglib2.0-0 xz-utils \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL --retry 12 --retry-delay 15 --retry-all-errors --retry-max-time 600 -o /tmp/ffmpeg.tar.xz \
      "https://github.com/BtbN/FFmpeg-Builds/releases/download/${FFMPEG_RELEASE}/${FFMPEG_ASSET}.tar.xz" \
 && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp \
 && install -m755 "/tmp/${FFMPEG_ASSET}/bin/ffmpeg" /usr/local/bin/ffmpeg \
 && install -m755 "/tmp/${FFMPEG_ASSET}/bin/ffprobe" /usr/local/bin/ffprobe \
 && rm -rf /tmp/ffmpeg.tar.xz "/tmp/${FFMPEG_ASSET}"

# **Assert the toolchain rather than commenting it.** Each of these is a fact this repo's code
# depends on, and each has a silent failure mode: a Python that is not 3.11 means the venv these
# dependencies were resolved against is not the one they run on; a missing libwebp means every
# poster and crop fails on the first job rather than at build time; and `use_metadata_tags` is
# what makes the identity tags exist at all — without it ffmpeg succeeds, the file plays, and the
# tags are silently absent (docs/decisions.md 3.3).
#
# **Written to files rather than piped into `grep -q`.** `grep -q` exits the instant it matches,
# which closes the pipe under ffmpeg and kills it with SIGPIPE — exit 141, which `-o pipefail`
# then propagates and `set -e` turns into a failed build. It is a race against ffmpeg finishing
# its write, so it passes on a fast machine and fails on a slow one; it passed in CI and failed
# here on the same commit. Grepping a file cannot race.
RUN set -eux; \
    python3 -c "import sys; assert sys.version_info[:2] == (3, 11), 'expected Python 3.11, got %s' % sys.version"; \
    ffmpeg -hide_banner -encoders > /tmp/encoders.txt; \
    ffmpeg -hide_banner -h muxer=mp4 > /tmp/mp4muxer.txt; \
    grep -q libwebp /tmp/encoders.txt; \
    grep -q libx264 /tmp/encoders.txt; \
    grep -q use_metadata_tags /tmp/mp4muxer.txt; \
    ffmpeg -version > /tmp/ffver.txt; \
    echo "toolchain: $(python3 -V), $(head -1 /tmp/ffver.txt)"; \
    rm -f /tmp/encoders.txt /tmp/mp4muxer.txt /tmp/ffver.txt

# ── pinned third-party source ───────────────────────────────────────────────────────────────
# **Vendored, not cloned.** The tree is committed at `vendor/SeedVR2`, pristine at the commit
# `SEEDVR2_COMMIT` names, so this build reaches no third-party host for its source. It used to
# `git clone` upstream and check that commit out, which made every build — including a rollback
# rebuild of an image that already shipped — depend on a repository nobody here controls staying
# up, keeping its history, and keeping that commit reachable.
#
# `SEEDVR2_COMMIT` is no longer a checkout instruction. It is the provenance stamp naming which
# upstream commit this copy was taken from, reported in every run record, and the string
# `vendor_patch.py` fails against if the tree under it is not the one it expects.
COPY vendor/SeedVR2 $SEEDVR2_DIR

# **The one edit to the pinned source, and the only one.** Everything else this worker changes
# about SeedVR2 is a runtime rebinding of a module-level name (`build_args`, `phasewatch`,
# `_read_frames_from_cap`, `wavelet_reconstruction`) — none of which touch the tree. The bfloat16
# handoff at the end of `_process_frames_core` is three lines inside a function body with no name
# to rebind, and it doubles the largest host allocation in the process; see `vendor_patch.py` for
# what and why.
#
# Above the dependency install rather than beside the code copy, because the import check below
# must exercise the patched source — an image whose patch silently failed would pass every test
# that does not send it 8K. `vendor_patch.py` raises if the pinned commit has moved under it, so
# a stale patch stops the build instead of shipping.
#
# Absolute paths: `WORKDIR /app` is not set until the dependency block below, and a relative
# `COPY` here would land the script wherever the base image left the working directory.
COPY vendor_patch.py /app/vendor_patch.py
RUN python3 /app/vendor_patch.py "$SEEDVR2_DIR"

# ── Python dependencies ─────────────────────────────────────────────────────────────────────
# SeedVR2's own requirements list `torch` and `torchvision` unpinned, which the base image
# already satisfies — so pip leaves the CUDA build in place rather than replacing it with a
# PyPI default. That is load-bearing and worth knowing before adding a version constraint here.
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r "$SEEDVR2_DIR/requirements.txt" -r requirements.txt

# **The dependency import check, above the bake.** `import inference_cli` pulls the real model
# chain — torch, diffusers, the attention backends — so a bad combination breaks the build in
# seconds rather than after a multi-GB download. This is the fail-loud-early rule in its most
# concrete form, and it is the check that makes moving to torch 2.8.0 cheap to try.
RUN cd "$SEEDVR2_DIR" && python3 -c "import inference_cli; print('inference_cli import OK')"

# **The two seams the release rides on, asserted where a failure is free.** Both are runtime
# rebindings of vendored names, and a vendored bump that renamed either would leave the worker
# installing nothing — the first symptom being a phase-4 crash on an 8K job with three phases of
# GPU already spent, or a host OOM that reads as a hardware problem. Checked here, a bump that
# moves them costs ten minutes of CI instead.
RUN cd "$SEEDVR2_DIR" && python3 -c "\
import inference_cli, src.utils.color_fix as cf, src.core.generation_phases as gp; \
assert callable(getattr(cf, 'wavelet_reconstruction', None)), 'color_fix.wavelet_reconstruction is gone'; \
assert getattr(gp, 'wavelet_reconstruction', None) is cf.wavelet_reconstruction, 'generation_phases no longer imports it'; \
assert 'cf-upscale-worker (release item 3)' in open('inference_cli.py').read(), 'vendor_patch did not take'; \
assert 'cf-upscale-worker (F-2026-08-19-31)' in open('src/utils/downloads.py').read(), 'vendor_patch skip-validation did not take'; \
assert 'cf-upscale-worker (F-2026-08-21-54)' in open('src/core/generation_phases.py').read(), 'vendor_patch dit-less phase entry did not take'; \
assert open('src/optimization/memory_manager.py').read().count('cf-upscale-worker (F-2026-08-21-54)') == 2, 'vendor_patch dit-less/vae-less cleanup did not take twice'; \
print('release seams OK: colour pyramid patchable, bfloat16 handoff removed, weight re-hash skipped, dit-less entry reloads, dit-less cleanup survives')"

# ── weights ─────────────────────────────────────────────────────────────────────────────────
# Their own layer, so editing the handler below does not re-download them. Nothing may be
# inserted between here and the code copy.
# `df` on both sides of the bake, in the same layer so it does not disturb the ordering above.
# This is the only place the real cost of the weights is visible: every earlier disk figure in
# docs/decisions.md 3.5 is about the base, and no build had reached this line when the size
# projections there were written.
COPY bake_weights.py RIFE-LICENSES.txt ./
RUN set -eux; \
    case "$BAKE_WEIGHTS" in 0|1) ;; \
      *) echo "BAKE_WEIGHTS must be 0 or 1, got '$BAKE_WEIGHTS'" >&2; exit 2 ;; \
    esac; \
    echo "--- disk before the bake ---"; df -h /; \
    mkdir -p "$SEEDVR2_MODEL_DIR" "$RIFE_MODEL_DIR"; \
    BAKE_WEIGHTS="$BAKE_WEIGHTS" python3 bake_weights.py; \
    install -m644 RIFE-LICENSES.txt "$RIFE_MODEL_DIR/LICENSE.txt"; \
    echo "--- disk after the bake ---"; df -h /; \
    du -sh "$SEEDVR2_MODEL_DIR" "$RIFE_MODEL_DIR"

# **The weights are in the image, so nothing re-proves it at runtime** (F-2026-08-19-31). The
# vendored `download_weight()` validates an existing model file against a size+mtime cache and,
# on a miss, reads all 16.4 GiB of it through SHA256 — and our cache is never seeded, because the
# bake above goes through `hf_hub_download` on purpose. Every fresh worker therefore paid that
# read on its first job: seconds on a materialized NVMe host, **6m06s and 12m29s measured** on
# hosts that stream layers lazily, in silence, before any frame was touched. `vendor_patch.py`
# teaches the vendored gate to honour this flag; see its docstring for why the obvious fix —
# seeding the mtime cache — is the wrong one on exactly the filesystems where this bug lives.
#
# **Set here, not above.** This line is a statement of fact about the layer beneath it. If the
# bake is ever made conditional, this must move with it: an image that claims baked weights it
# does not have would skip the validation of a file that really is missing — which the patch's
# surviving `os.path.exists` guard then catches, but the honest place to be right is here.
#
# **Made conditional 2026-08-22 with the bake itself, exactly as the paragraph above required.**
# `CF_WEIGHTS_BAKED` rides here rather than in the ENV block near the top of the file for two
# reasons: this is where the fact becomes true, and an ENV up there would put `BAKE_WEIGHTS` above
# the system packages, the ffmpeg fetch, the vendor copy and the pip install — so building the
# route-C variant would invalidate all of them and pay a cold build for a one-byte difference.
ENV SEEDVR2_WEIGHTS_BAKED=${BAKE_WEIGHTS} \
    CF_WEIGHTS_BAKED=${BAKE_WEIGHTS}

# ── build identity ──────────────────────────────────────────────────────────────────────────
# **An image that cannot say what it is makes every measurement taken on it an assertion.** The
# 2026-08-17 calibration campaign banked ten runs' worth of VRAM peaks and recorded
# `"image": null` on all ten, because the handler read `IMAGE_TAG` and nothing ever set it; the
# claim that those ten share one build rests on nobody having re-created the endpoint. The
# registry key that prices this worker is keyed on exactly that.
#
# **Below the bake, deliberately.** These change on every single build, and a layer above the
# weights would re-download 17 GB to record a commit hash. Here they cost one small layer.
#
# The digest is not among them and cannot be: it is computed when the image is pushed, after the
# last layer is sealed. `IMAGE_REF` is the next best thing and is enough — CI pushes an immutable
# `:sha-<commit>` tag alongside `:latest`, so the reference resolves to one digest forever.
ARG IMAGE_REF=""
ARG BUILD_COMMIT=""
ARG BUILD_UTC=""
# **`WORKER_VERSION` was never set, so every build block said `0.1.0-dev`**
# (F-2026-08-18-10) — the same class of gap as the missing `IMAGE_TAG` that release item 5
# closed, and found the same way: by reading a bundle back and noticing a constant where a
# reading belonged. It is a hand-set version rather than a derived one because that is what it
# is for: `BUILD_COMMIT` already answers "which code", truthfully and without anyone having to
# remember to bump anything, while this answers "which release of it" and only a person can say.
ARG WORKER_VERSION="0.2.0"
ENV IMAGE_TAG=${IMAGE_REF} \
    BUILD_COMMIT=${BUILD_COMMIT} \
    BUILD_UTC=${BUILD_UTC} \
    WORKER_VERSION=${WORKER_VERSION} \
    SEEDVR2_COMMIT=${SEEDVR2_COMMIT}

# ── this repo's code, last ──────────────────────────────────────────────────────────────────
# **`*.py` is not everything the handler needs.** `calibration.json` is measured data the
# estimator plans from, and when it was added to the repo but not to this line the image shipped
# without it — silently. Nothing failed: `load_calibration` treats an absent file as a normal
# state meaning "take the floor", which is correct behaviour and indistinguishable from having no
# measurements at all. A video job planned to the floor at 15.7 s/frame with a shipped table that
# said `resident` would do it in 3.4, and the only symptom was a rung nobody expected.
#
# Named explicitly rather than `COPY *`: a wildcard would also sweep in whatever lands in this
# directory later. `tests/run_local.py` asserts every non-`.py` file here is covered by a line
# below, so adding a data file and forgetting this is a red suite rather than a quiet floor.
COPY *.py calibration.json ./

# The handler must import with the heavy chain still lazy — if this ever pulls in torch, the
# rung-1 suite stops running in CI and nobody finds out until a GPU-less runner fails.
RUN python3 -c "import handler; import sys; assert 'torch' not in sys.modules, 'handler imported torch eagerly'; print('handler import OK, torch still lazy')"

CMD ["python3", "-u", "handler.py"]
