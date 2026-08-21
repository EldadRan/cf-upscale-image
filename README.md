# cf-upscale-image

A RunPod serverless GPU worker that upscales stills and video with SeedVR2, driven in-process
rather than through ComfyUI. Bytes move through S3-compatible object storage in both directions:
a job names a source URL and an output destination, and no image data travels in the job envelope.

A single image is a degenerate case of the video path — one frame in, one frame out — so there is
one code path and one image for both.

## What is here

```
handler/                        the worker, and the Docker build context
.github/workflows/              the build-and-publish workflow
```

`handler/Dockerfile` is the whole build. It starts from `runpod/pytorch:2.8.0-py3.11-cuda12.8.1`,
fetches the SeedVR2 source pinned by `SEEDVR2_COMMIT`, applies `vendor_patch.py`, installs a
pinned ffmpeg and bakes the model weights into the image, so a cold worker reaches for nothing
over the network before its first frame.

`vendor_patch.py` is the only modification made to the vendored source. It states each patch at
length and fails the build if the upstream text it expects has moved, so a silently rebased
dependency stops the build rather than producing a subtly different image.

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
