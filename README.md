# cf-upscale-image

A RunPod serverless GPU worker that upscales stills and video with SeedVR2, driven in-process
rather than through ComfyUI. Bytes move through S3-compatible object storage in both directions:
a job names a source URL and an output destination, and no image data travels in the job envelope.

A single image is a degenerate case of the video path — one frame in, one frame out — so there is
one code path and one image for both.

## What is here

```
handler/                        the worker, and the Docker build context
handler/vendor/SeedVR2/         the upstream SeedVR2 source, vendored pristine
.github/workflows/              the build-and-publish workflow
```

`handler/Dockerfile` is the whole build. It starts from `runpod/pytorch:2.8.0-py3.11-cuda12.8.1`,
copies in the vendored SeedVR2 source, applies `vendor_patch.py`, installs a pinned ffmpeg and
bakes the model weights into the image, so a cold worker reaches for nothing over the network
before its first frame.

## The vendored source

`handler/vendor/SeedVR2/` is [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
committed here **pristine**, exactly as upstream published it at the commit `SEEDVR2_COMMIT`
names. It is licensed Apache-2.0 and upstream's `LICENSE` is kept alongside it.

It is vendored rather than cloned at build time so that building this image depends on nothing
outside this repository — including a rebuild of a version that already shipped, which would
otherwise need a third-party repository to still exist, still have its history, and still have
that commit reachable.

`SEEDVR2_COMMIT` is therefore not a checkout instruction. It records which upstream commit this
copy was taken from, and the worker reports it in every run record.

**`vendor_patch.py` is the only modification, and it is applied at build rather than baked into
the vendored tree.** That is deliberate: keeping the tree pristine and the change in a script
means the modification is stated rather than buried, the tree stays diffable against upstream in
one command, and the patch's exact-match-or-fail guard doubles as a tamper check on this
repository's own copy — if the text it expects is not there, the build stops instead of quietly
producing a different image.

## Building

**One image per model and base, each named for both (X2, CF 2026-11-22): `cf-upscale-<model>-<base>`.**
Each has its own workflow, dispatched by hand with no inputs, and its own package:

| workflow | package | base | checkpoint |
|---|---|---|---|
| `publish-7b-fp16-runpod.yml` | `cf-upscale-7b-fp16-runpod` | `runpod/pytorch` | 7B fp16 — the RunPod worker |
| `publish-7b-fp8-vast-mini.yml` | `cf-upscale-7b-fp8-vast-mini` | Vast `vastai/pytorch` mini | 7B fp8 mixed — Vast stills |

Both call one reusable build, `.github/workflows/build.yml`, which refuses a package name that
does not describe the pair it is given.

**The RunPod image bakes its weights; the Vast image does not (X4).** It builds with
`BAKE_WEIGHTS=0` and `handler/weights.py` fetches the checkpoint and VAE from HuggingFace at start
— pinned revision, size and sha256 from `bake_weights.py`, resumable ranges, hashed once per host,
the fetch's bytes and seconds in the run record under `load_strip.weights`. The older `cf-upscale-image` package is left as it was;
the endpoints pinned to it stay pinned until they are re-pinned.

The workflows have **no `push` trigger, deliberately** — not every commit is meant to become an
image. Builds are dispatched on purpose, from the Actions tab, and a pull request never publishes.

`publish` is gated on `toolchain-gate` (`needs: toolchain-gate`), which installs the pinned ffmpeg
and asserts that `libwebp`, `libx264` and `use_metadata_tags` are present before a build starts.
A bad pin costs seconds that way rather than a full build — and the assertion is only meaningful
because the gate installs the exact binary the image will carry, not a distribution's.

A dispatched build publishes two tags into its own package on GHCR, and caches from its own
`:latest`:

```
ghcr.io/<owner>/cf-upscale-<model>-<base>:latest
ghcr.io/<owner>/cf-upscale-<model>-<base>:sha-<commit>
```

**Endpoints pin the `sha-` tag, never `latest`.** `latest` moves, and a worker that pulled it
cannot say afterwards which build it ran. The image stamps its own `BUILD_COMMIT`, `IMAGE_REF`
and `BUILD_UTC` at build time and reports them in every run record, so a measurement can always
be traced to the bytes that produced it.

Expect a build to take tens of minutes. The 7B fp16 RunPod image is roughly 24 GiB compressed and
the 7B fp8 Vast image, which carries no weights, is the base and code alone.

To build the RunPod image locally (the Dockerfile's no-argument build):

```
docker build handler/
```

## Tests

**The contract suite is not in this repository, and that is deliberate rather than missing.** It
lives with the test harness and the operating scripts it exercises, and it is run before a build
is dispatched rather than by this workflow — its result is quoted in the request that asks for
the build.

It needs no GPU: it exercises the planner, the validator, the schedule arithmetic and the encode
path, and stops where the model would be called. What it does need is the same pinned ffmpeg this
workflow installs, because a suite green on a different encoder proves less than it looks.

So what CI enforces is narrower than a green suite, and worth stating plainly: **`toolchain-gate`
checks the toolchain, not the worker.** A dispatched build proves the image assembles and that
its ffmpeg has the capabilities the encode path needs. Whether the worker behaves is established
before the dispatch, not by it.

## First thing after a clone: install the hooks

```
git config core.hooksPath .githooks
```

**One command, once per checkout, and nothing works without it.** `core.hooksPath` is per-clone
config — the hooks are tracked in `.githooks/`, but git does not look there until it is told to.
**Until you run it this repository has no hooks at all**, and a push whose architecture graphs are
stale will succeed silently.

`.githooks/README.md` says what the hook does, what it deliberately does not do, and the one cost
of pointing `core.hooksPath` away from `.git/hooks/`.
