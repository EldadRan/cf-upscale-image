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

The workflow has **no `push` trigger, deliberately** — not every commit is meant to become an
image. Builds are dispatched on purpose, from the Actions tab, and pull requests run the tests
without publishing.

`publish` is gated on `contract-tests` (`needs: contract-tests`), so a red suite publishes
nothing. The suite also pins the ffmpeg build and asserts the codecs the worker relies on before
any GPU-free test runs, because a bad pin should cost seconds rather than a full build.

A dispatched build publishes two tags to GHCR:

```
ghcr.io/<owner>/cf-upscale-image:latest
ghcr.io/<owner>/cf-upscale-image:sha-<commit>
```

**Endpoints pin the `sha-` tag, never `latest`.** `latest` moves, and a worker that pulled it
cannot say afterwards which build it ran. The image stamps its own `BUILD_COMMIT`, `IMAGE_REF`
and `BUILD_UTC` at build time and reports them in every run record, so a measurement can always
be traced to the bytes that produced it.

Expect a build to take tens of minutes and produce an image of roughly 24 GiB; the weights are
most of it.

To build the same thing locally:

```
docker build handler/
```

## Running the tests

The contract suite runs without a GPU: it exercises the planner, the validator, the schedule
arithmetic and the encode path, and stops where the model would be called. That is what makes it
a gate worth having on every build — it is fast enough to run before one, and it fails for the
reasons a job would fail.

CI runs it as the `contract-tests` job. It needs Python 3.11, the pinned ffmpeg the workflow
installs, and `pip install boto3 requests numpy pillow`.
