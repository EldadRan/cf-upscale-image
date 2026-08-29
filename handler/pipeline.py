"""Driving SeedVR2, and owning the loop the vendored CLI would otherwise own.

`process_single_file` is not used, and `docs/decisions.md` 0.3 has the argument. In short: its
video branch is a thin loop around `_stream_video_chunks` that adds only the writing, and the
writing is the part that cannot satisfy the contract. So this module reimplements that loop —
about thirty lines — and keeps everything else.

What still comes from the pinned source, unchanged:

  **`parse_arguments()` with a synthetic `sys.argv`.** Every default, every `choices` list and
  every bit of argparse validation comes from the vendored code, so the worker's accepted values
  cannot drift from what that code actually accepts. This is the single most valuable thing the
  existing image worker established and it survives intact.

  **`_stream_video_chunks()`**, whose docstring states *"Caller is responsible for VideoCapture
  lifecycle and result handling"* — it is designed to be driven this way.

**The frame-count discipline is the reason this module exists in its shape.** The vendored
`process_single_file` budgets frames from `cv2.CAP_PROP_FRAME_COUNT`, a container probe, while
the read is a decode; where the probe under-counts, frames are silently dropped. Here the budget
is explicit, the decode is counted as it happens, and decoded-in is asserted against written-out
before anything is returned.
"""

import gc
import os
import sys

import colorfix
from errors import INTERNAL, INVALID_SOURCE, WorkerError
from phasewatch import PhaseWatch

SEEDVR2_DIR = os.environ.get("SEEDVR2_DIR", "/app/SeedVR2")

# SeedVR2 is one-step: there is no sampling trajectory for a seed to vary, so a caller-settable
# one would be a parameter that changes nothing. Pinned rather than accepted — a parameter's
# presence is not a capability.
SEED = 42

#: `--batch_size` must follow 4n+1 (1, 5, 9, 13, 17, 21…), stated in the vendored help text and
#: enforced nowhere. A value off that lattice is the kind of thing that fails deep inside the
#: model with an unhelpful shape error, so the estimator picks from here and this is the list.
BATCH_SIZES = tuple(4 * n + 1 for n in range(0, 32))


def snap_batch_size(requested):
    """Largest valid 4n+1 batch size not exceeding `requested`, minimum 1."""
    return max(b for b in BATCH_SIZES if b <= max(1, requested))


def _model_dir():
    """Baked weights by default; an env override so weights can be bind-mounted for local
    iteration, and `/runpod-volume` if one happens to be mounted. The indirection is what lets
    the handler be worked on without an image build, and the baked path is the default so a
    deployed image needs no environment set to be correct."""
    override = os.environ.get("SEEDVR2_MODEL_DIR", "").strip()
    if override:
        return override
    if os.path.isdir("/runpod-volume"):
        return "/runpod-volume/models/SEEDVR2"
    return os.path.join(SEEDVR2_DIR, "models", "SEEDVR2")


def load_cli():
    """Import the vendored CLI lazily. Pulls in torch and cv2, so it is a GPU-box import and
    must stay out of module scope: the rung-1 contract suite runs in CI with neither."""
    if SEEDVR2_DIR not in sys.path:
        sys.path.insert(0, SEEDVR2_DIR)
    import inference_cli  # noqa: PLC0415 — deliberate lazy heavy import

    return inference_cli


def _vendored_module(cli, dotted):
    """A module inside the vendored tree, by the name `inference_cli` itself imports it under.

    Reached through `sys.modules` rather than a fresh `import_module` so the object patched is
    the one the vendored code already holds a reference to. Importing it again by path would
    produce a second module object, and patching that one would change nothing.
    """
    import importlib  # noqa: PLC0415 — only needed on the GPU path

    return sys.modules.get(dotted) or importlib.import_module(dotted)


def _color_fix_module(cli):
    return _vendored_module(cli, "src.utils.color_fix")


def _phases_module(cli):
    return _vendored_module(cli, "src.core.generation_phases")


def build_args(cli, plan, source_path, model_filename, color_correction, debug=False):
    """The vendored `argparse.Namespace`, built through the vendored parser.

    Every value here is one the pinned source declares. Passing them through `parse_arguments()`
    rather than constructing a Namespace by hand is what makes that true: an option this code
    spells wrongly fails immediately and loudly, rather than being silently absent from a
    Namespace the model then reads a default from.
    """
    argv = [
        "inference_cli.py", source_path,
        "--output", os.devnull,          # never used: this module owns the writing
        "--output_format", "mp4",
        "--dit_model", model_filename,
        "--model_dir", _model_dir(),
        "--resolution", str(plan["target_short_edge_px"]),
        # No longest-edge cap. A cap refuses work that would have succeeded, which is the failure
        # this model row has already produced once; fit is decided per job by the estimator.
        "--max_resolution", "0",
        "--seed", str(SEED),
        "--color_correction", color_correction,
        "--batch_size", str(plan["batch_size"]),
        "--chunk_size", str(plan["chunk_size"]),
        "--temporal_overlap", str(plan["temporal_overlap"]),
        "--blocks_to_swap", str(plan["blocks_to_swap"]),
        "--dit_offload_device", plan["dit_offload_device"],
        "--vae_offload_device", plan["vae_offload_device"],
        "--tensor_offload_device", plan["tensor_offload_device"],
    ]
    if plan["swap_io_components"]:
        argv.append("--swap_io_components")
    # **The overlap is passed explicitly rather than left to the vendored default.** 128 px is
    # 8% of a 1536 tile and 33% of a 384 one, so the same default is five different policies
    # across the tile sizes this worker uses -- at the smallest it puts most of the frame inside
    # a cross-fade. Passing it makes that a decision rather than an inheritance.
    if plan["vae_encode_tiled"]:
        argv += ["--vae_encode_tiled",
                 "--vae_encode_tile_size", str(plan["vae_encode_tile_size"]),
                 "--vae_encode_tile_overlap", str(plan["vae_encode_tile_overlap"])]
    if plan["vae_decode_tiled"]:
        argv += ["--vae_decode_tiled",
                 "--vae_decode_tile_size", str(plan["vae_decode_tile_size"]),
                 "--vae_decode_tile_overlap", str(plan["vae_decode_tile_overlap"])]
    if debug:
        argv.append("--debug")

    # **`--cache_dit` / `--cache_vae`, and the reasoning that used to omit them was right about
    # the wrong scope** (F-2026-08-20-45). It said: this worker is not looking for reuse, anything
    # a *subsequent job* has no certain use for is freed, and the cold start that buys back is not
    # worth the OOM it risks. Every clause of that is still true between jobs.
    #
    # Within one job it was exactly backwards. `_process_frames_core` runs once per chunk, and
    # with the caches off it builds a fresh runner every time — so chunk 2 materialised 16.4 GiB
    # of weights *beside chunk 1's copy, which had not been freed*. Measured on the 222-frame
    # run: chunk-2 postprocess-entry anon 23.9 GiB, chunk-2 DiT anon 38.58. The second copy is
    # the whole 14.7 GiB, and it is why a plan that priced one copy could not fit.
    #
    # The flags are the vendored gate: `_process_frames_core` reads
    # `cache_dit = args.cache_dit if runner_cache is not None else False`, so the cache dict alone
    # does nothing. Both are needed, and `release_runner_cache` below puts the between-jobs
    # posture back at the end of every job so nothing about residency changes — that is Build D's
    # ruling to make, not this fix's.
    argv += ["--cache_dit", "--cache_vae"]

    saved = sys.argv
    sys.argv = argv
    try:
        return cli.parse_arguments()
    finally:
        sys.argv = saved


class _StillCapture:
    """The slice of `cv2.VideoCapture` that `_read_frames_from_cap` actually uses, over one frame.

    **Exists because `VideoCapture` cannot return four channels.** It decodes to BGR whatever the
    source held, so an RGBA PNG loses its alpha at the read — before the model, which would
    otherwise have upscaled it (`generation_phases.py` sets `is_rgba` from the tensor's last
    dimension and routes alpha through edge-guided upscaling). Measured on a real 2160×2160 RGBA
    file: `imread(IMREAD_UNCHANGED)` gives `(2160, 2160, 4)`, `VideoCapture` gives
    `(2160, 2160, 3)`.

    Deliberately narrow: `read`, `release`, `isOpened`, `get`. A fuller fake would stand in for
    more of `VideoCapture` than this worker drives, and the extra surface is what goes stale.
    """

    def __init__(self, frame, fps=1.0):
        self._frame = frame
        self._fps = fps
        self._served = False

    def read(self):
        if self._served:
            return False, None
        self._served = True
        return True, self._frame

    def isOpened(self):
        return self._frame is not None

    def release(self):
        self._frame = None

    def get(self, prop):
        height, width = self._frame.shape[:2]
        return {"fps": self._fps, "width": float(width), "height": float(height)}.get(
            _PROP_NAMES.get(prop), 0.0)


#: Only the three properties `open_source` reads. Resolved lazily against the vendored cv2 so
#: there is one cv2 in the process, as everywhere else here.
_PROP_NAMES = {}


def to_unit_scale(array, np):
    """Pixels in [0, 1], whatever depth they arrived at.

    **This is what broke every 16-bit source.** `cv2.imread(..., IMREAD_UNCHANGED)` preserves the
    file's bit depth, so a 16-bit PNG arrives as `uint16` — and dividing that by 255 hands the
    model values up to 257 where it expects at most 1. Everything clamps to white, on a path where
    the vendored code's own log reports correct dimensions throughout, so nothing looks wrong.

    The vendored reader divides by 255 unconditionally too, and is safe only because
    `cv2.VideoCapture` always decodes to 8 bits. Reading a still with `IMREAD_UNCHANGED` is what
    made the difference reachable, so the normalisation has to follow the dtype rather than assume
    one.

    **PIL cannot be used to check this**: it reports a 16-bit RGBA PNG as `uint8` because it
    downconverts on open. Two supplied files were 16-bit and PIL called both 8-bit, which is how
    this hid behind four wrong diagnoses.
    """
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    if array.dtype in (np.float32, np.float64):
        return array.astype(np.float32)
    return array.astype(np.float32) / 255.0


def to_uint8(array, np):
    """8-bit, scaled rather than truncated.

    `uint16(65535).astype(uint8)` is 255 by luck and `uint16(32768).astype(uint8)` is 0 — a cast
    wraps rather than converts, so a half-transparent alpha would become fully opaque.
    """
    if array.dtype == np.uint16:
        return (array.astype(np.float32) / 257.0).round().astype(np.uint8)
    return array.astype(np.uint8)


def resize_alpha(alpha, width, height):
    """Alpha at the output's size, from the source's.

    **Lanczos, and only for alpha.** The vendored model does not diffuse alpha either — it
    upscales it by edge-guided interpolation and recombines — so doing the interpolation here
    costs a guided filter's worth of edge quality and buys correctness. A soft alpha edge is a
    visible flaw on a cutout; a white rectangle is not a picture at all.

    Kept as a module-level function taking plain arrays so it can be tested without a GPU, which
    is the only way this path gets exercised outside a paid job.
    """
    import numpy as np
    from PIL import Image

    array = np.asarray(alpha)
    if array.ndim == 3:
        array = array[..., 0]
    image = Image.fromarray(to_uint8(array, np), mode="L")
    return np.asarray(image.resize((width, height), Image.LANCZOS))


def _fit_to(cli, frames, size):
    """Every frame at exactly `size`, as (width, height).

    **`INTER_AREA` on the way down, and it is not the same choice as everywhere else.** This module
    uses nearest-neighbour for crop evidence, because smooth interpolation there would invent
    plausible pixels and flatter the model. Here the opposite applies: this is a real delivery
    resize of at most the aspect difference, and area averaging is what a downscale wants — Lanczos
    would ring on the high-frequency detail the model just produced.

    Alpha rides along as a fourth channel rather than being handled separately: `cv2.resize` treats
    it as data, and at these ratios there is no edge decision to get wrong.
    """
    cv2, np = cli.cv2, cli.np
    width, height = int(size[0]), int(size[1])
    fitted = np.empty((frames.shape[0], height, width, frames.shape[3]), dtype=frames.dtype)
    for index, frame in enumerate(frames):
        fitted[index] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return fitted


def _assert_channels(frames, expected):
    """Refuse a chunk whose channel count is not the one promised.

    A module-level function on a plain array for the same reason `resize_alpha` is one: it is the
    only way this path gets exercised without a paid GPU job.
    """
    if frames.shape[-1] != expected:
        raise WorkerError(
            INTERNAL,
            "alpha was routed through the model, but it returned {} channel(s) where {} were "
            "expected. The vendored RGBA branch did not run; the master would have been written "
            "with every pixel shifted by one channel.".format(frames.shape[-1], expected))


def _read_frames_preserving_alpha(cli, alpha_out, strip=True):
    """A drop-in for the vendored `_read_frames_from_cap` that keeps a fourth channel.

    `strip` decides who upscales the alpha. True takes it out here and hands the model three
    channels, so the alpha is resized by `resize_alpha` and reattached; False hands over four, so
    the model's `is_rgba` triggers and `edge_guided_alpha_upscale` does it along the RGB's own
    edges. See `decisions.md` 4.9 — the second is expected to be better on a cutout and is
    measured at nothing, which is why it is a flag and not a change.

    The vendored one hard-codes `COLOR_BGR2RGB`, which **raises** on a four-channel frame rather
    than dropping the alpha quietly — so a shim capture cannot simply be handed to it. This is the
    same function with the vendored *image* path's two-branch conversion (`inference_cli.py` lines
    330 and 333) in place of the single fixed one. Everything else — the early break, the `None`
    for no frames, the float32 [0,1] scaling, the `[T, H, W, C]` stack — is preserved, because
    `_stream_video_chunks` depends on all of it.
    """
    cv2 = cli.cv2
    np = cli.np
    torch = cli.torch

    def read_frames(cap, max_frames):
        frames = []
        for _ in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
                # **The alpha is taken out here and never shown to the model — and the reason
                # given for that was wrong.** Kept for now because it is measured correct
                # (+1.000 through the padding path) and changing it needs its own measurement,
                # not because the vendored path is broken. See `decisions.md` 4.9.
                #
                # What this comment used to claim: that alpha is captured from the *padded* video
                # while the RGB is cropped back, so the two disagree whenever padding is needed,
                # and that an RGBA source needing padding returns a white rectangle at -0.01
                # correlation. Both halves are wrong. The padding at that site is **temporal**
                # (4n+1 frames), not spatial, and a temporal mismatch would fail at the
                # `torch.cat` rather than produce a picture. The white rectangle was the 16-bit
                # divide-by-255 (3.9) — this was one of that bug's four wrong diagnoses, and the
                # -0.01 measurement was confounded by the control being 8-bit.
                #
                # What the vendored path actually does is better than this one for a cutout:
                # `edge_guided_alpha_upscale` upscales alpha guided by the RGB's own edges, and
                # detects binary masks separately from gradient alphas. This resizes alpha with
                # Lanczos and knows nothing about where the edges went.
                if strip:
                    alpha_out.append(frame[..., 3].copy())
                    frame = frame[..., :3]
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(to_unit_scale(frame, np))
        if not frames:
            return None
        return torch.from_numpy(np.stack(frames)).to(torch.float32)

    return read_frames


def open_source(cli, source_path, keep_alpha=False):
    """Open the source and read its shape. **Returns no frame count** — see the module docstring.

    `cv2` arrives through the vendored CLI's own import so there is one cv2 in the process.

    `keep_alpha` opens a still through `imread(IMREAD_UNCHANGED)` instead, which is the only way
    a fourth channel reaches the model at all.
    """
    if keep_alpha:
        return _open_still_with_alpha(cli, source_path)
    capture = cli.cv2.VideoCapture(source_path)
    if not capture.isOpened():
        raise WorkerError(INVALID_SOURCE, "the decoder could not open the source")
    fps = capture.get(cli.cv2.CAP_PROP_FPS) or 0.0
    width = int(capture.get(cli.cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cli.cv2.CAP_PROP_FRAME_HEIGHT))
    if not width or not height:
        raise WorkerError(INVALID_SOURCE, "the decoder reports no dimensions for the source")
    return capture, {"fps": fps, "width": width, "height": height}


def _open_still_with_alpha(cli, source_path):
    """One frame, four channels, read the way the vendored image path reads it."""
    frame = cli.cv2.imread(source_path, cli.cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise WorkerError(INVALID_SOURCE, "the decoder could not open the source")
    if frame.ndim != 3 or frame.shape[2] != 4:
        # The probe said this source has alpha and the decoder disagrees. Refusing beats
        # continuing on three channels, which would look like a success that quietly lost data.
        raise WorkerError(
            INVALID_SOURCE,
            "source probed as carrying alpha but decoded to {} channel(s)".format(
                1 if frame.ndim == 2 else frame.shape[2]),
        )
    height, width = frame.shape[:2]
    _PROP_NAMES.update({cli.cv2.CAP_PROP_FPS: "fps",
                        cli.cv2.CAP_PROP_FRAME_WIDTH: "width",
                        cli.cv2.CAP_PROP_FRAME_HEIGHT: "height"})
    return _StillCapture(frame), {"fps": 1.0, "width": width, "height": height}


def run(cli, capture, args, plan, frame_budget, writer, on_chunk=None, keep_alpha=False,
        alpha_through_model=False, exact_size=None, ratchet=None, on_batch=None,
        schedule=None, on_tile=None):
    """Stream chunks through the model and into the writer. Returns frames written.

    `frame_budget` is what the caller decided to read, not what a container claimed. Passing a
    generous budget is safe: `_read_frames_from_cap` stops when the decode is exhausted, so the
    budget bounds the read from above and the decode determines it from below.

    `on_chunk(frames_done)` is called every `_PROGRESS_EVERY` frames written and again at each
    chunk boundary, which is where progress and the ETA come from — measured in-process, because
    reading a library's log back through RunPod's platform costs a collection lag observed at
    2 h 17 m between adjacent statements. **Per frames written rather than per chunk**, because
    a chunk holding the whole clip is the designed normal and the per-chunk cadence reported
    `0/N` for the entire run (F-2026-08-18-11).
    """
    # **A chunk size of zero is the OOM**, not a default. Upstream reads 0 as "load every frame
    # at once", which is what makes a long clip fail on a card that would have handled it in
    # pieces. The estimator always chooses a positive value; this refuses rather than falling
    # back, because a silent fallback here is the exact defect this worker exists to fix.
    chunk_size = plan["chunk_size"]
    if not chunk_size or chunk_size < 1:
        raise WorkerError(
            INTERNAL,
            "the plan set chunk_size={!r}; upstream reads that as 'load every frame at once', "
            "which is the failure mode this worker exists to avoid".format(chunk_size),
        )
    # **Swapped for the duration of the stream, then put back.** The vendored reader hard-codes a
    # three-channel conversion that raises on a four-channel frame, so an alpha job cannot reach
    # `_stream_video_chunks` without this. Patched rather than forked for the same reason
    # `build_args` patches `sys.argv`: the vendored source stays the source of truth, and the
    # coupling is one named function rather than a copy that drifts from it.
    original_reader = None
    carried_alpha = []
    if keep_alpha:
        original_reader = cli._read_frames_from_cap
        cli._read_frames_from_cap = _read_frames_preserving_alpha(
            cli, carried_alpha, strip=not alpha_through_model)
    # **Swapped for the same reason and in the same way**: the colour-correction pyramid indexes
    # its padded convolution in 32 bits, which caps a 4K pass at 84 frames and an 8K pass at 21.
    # Above that the vendored code does not degrade -- it dies in phase 4 with the three
    # expensive phases already paid for. `colorfix.install` splits the pyramid and nothing else,
    # and only when the batch would actually cross the bound, so every window measured below it
    # takes the byte-for-byte path it was measured on.
    colour_restore = colorfix.install(cli, _color_fix_module(cli), _phases_module(cli),
                                      debug=cli.debug)
    # **Installed here rather than around a chunk.** Each of the four phases runs once per chunk,
    # and the figure worth keeping is the maximum across all of them; a watch scoped to one chunk
    # would report the last chunk's peak, which is the one number nobody wants. Scoped here it
    # also spans a ratchet, so a run that stepped down still reports which phase set the ceiling.
    # `on_batch(phase, index, total)` is the job's only heartbeat once the chunk holds the whole
    # clip. `on_chunk` used to fire exactly once in that case, at the end; it now advances with
    # the frames actually written (F-2026-08-18-11).
    #: **One model, for the life of this job** (F-2026-08-20-45), and on rung 2 for the life of
    #: each chunk's DiT (amendment 9). Local to this call, so it dies when the job does; the
    #: vendored global cache it also populates is evicted by `release_runner_cache`, which the
    #: caller runs in a `finally`.
    runner_cache = {}
    # **`on_tile` is the deadline checkpoint's only input** (`api.md` §4d). It rides the same tap
    # as `on_batch` and for the same reason: the tap wraps the vendored logger and observes before
    # delegating, so a tile announcement reaches us whether or not the vendored debug is on. A
    # refusal raised from it travels the seam `_is_a_refusal` opened.
    watch = PhaseWatch(cli, on_batch=_scheduling_eviction(on_batch, plan, runner_cache,
                                                          debug=cli.debug, schedule=schedule),
                       on_tile=on_tile)
    #: **Stamped with the attempt the handler is on.** `run.last_phases` is a function attribute
    #: and nothing clears it, so on a warm worker it survives between jobs — and a run that
    #: raises before reaching here (a still, a refusal, a failure before the stream) would find
    #: the PREVIOUS run's watch waiting and record its phase times as its own. The stamp is what
    #: lets `handler._phase_watch` tell this attempt's watch from a leftover.
    #:
    #: Read off `run` rather than taken as a parameter because `run`'s signature is the vendored
    #: call's shape; the handler publishes the token the same way it reads the results back.
    watch.attempt_token = getattr(run, "attempt_token", None)
    run.last_phases = watch
    #: **Reachable by the end-of-job release**, which runs in the handler's `finally` and has no
    #: sight of this scope. Published rather than passed because the release must also work on
    #: the paths that never reached `run` — a refusal before the stream, or a crash inside it.
    run.last_runner_cache = runner_cache
    try:
        # `carried_alpha` is what `_stream` reattaches. None when the model carried the channel
        # itself, because reattaching a Lanczos alpha over the model's would be doing the work
        # twice and keeping the worse of the two.
        with watch:
            written = _stream(cli, capture, args, plan, frame_budget, writer, on_chunk, chunk_size,
                              carried_alpha if (keep_alpha and not alpha_through_model) else None,
                              runner_cache=runner_cache, schedule=schedule,
                              expect_channels=4 if (keep_alpha and alpha_through_model) else None,
                              exact_size=exact_size, ratchet=ratchet)
        run.last_pixel_stats = getattr(_stream, "last_pixel_stats", None)
        run.last_output_size = getattr(_stream, "last_output_size", None)
        # **The capture the stream finished on, which is not always the one it started with.** A
        # ratchet re-opens the source, so `assert_source_exhausted` asking the caller's original
        # capture whether anything is left would be asking a decoder that was abandoned at the
        # frame the OOM happened on — and it would answer "yes, plenty", failing a job that
        # delivered every frame.
        run.last_capture = getattr(_stream, "last_capture", capture)
        run.last_ratchet = list(getattr(_stream, "last_ratchet", []))
        return written
    finally:
        if original_reader is not None:
            cli._read_frames_from_cap = original_reader
        colorfix.uninstall(colour_restore)


def _stream(cli, capture, args, plan, frame_budget, writer, on_chunk, chunk_size,
            carried_alpha=None, expect_channels=None, exact_size=None, ratchet=None,
            runner_cache=None, schedule=None):
    """The loop itself, split out so the reader swap in `run` has a scope to wrap.

    **An out-of-memory here is recoverable without losing what is already written**, and that is
    the whole reason this function has an outer loop.

    The failure it replaces: an OOM used to unwind to `_upscale_once`, which restarts the clip
    from frame zero at a more conservative rung. On a 44-chunk job an OOM in the last chunk
    therefore cost the entire run *and then ran it again* — four and a half hours to redo four and
    a half hours. This module's own note called it "the case nobody has seen"; the 929-frame run
    of 2026-08-14 made it concrete.

    **What makes recovery possible is that an OOM does not kill the container.** It is a Python
    exception raised inside this process. ffmpeg is a separate process holding a pipe that already
    contains every frame written so far, and it is untouched. So this is not a checkpoint scheme:
    keep the same writer, re-open the decoder after the last frame written, carry on with a
    smaller temporal window.

    **And the point of it is quality, not resilience.** The estimator plans the window from the
    *maximum* of measured peaks because a miss used to cost the whole job. When a miss costs one
    chunk instead, that conservatism stops paying for itself and the window — the dominant quality
    lever on video — can be planned closer to the edge.

    Three properties it has to preserve, each of which is a way it could look like it worked:

      **No frame written twice and none skipped.** The resume point is `frames_written`: what
      actually reached the writer. A chunk that OOMed after writing half its frames therefore
      resumes correctly with nobody tracking where inside it the failure landed.

      **The model keeps its temporal context.** A fresh generator's first chunk has no overlap
      with the last one written, so the decode restarts `lead_in` frames *earlier* and the first
      `lead_in` frames it produces are discarded. Without that, the ratchet introduces a cut the
      source does not have — spending quality to save it.

      **The alpha stays aligned.** `carried_alpha` is appended by the patched reader, one entry per
      frame *read*, and indexed by output position. Re-reading appends duplicates, so it is
      truncated to the resume point and indexed by an explicit cursor: the `frames_written`
      arithmetic that was right with one generator is wrong with two.
    """
    import numpy as np  # noqa: PLC0415 — kept off module scope like every other use here

    frames_written = 0
    #: Running statistics of what the model produced, carried into the manifest. **Not a verdict.**
    #: A worker cannot tell a broken output from a photograph of a white wall, and refusing on
    #: "looks blank" would reject legitimate work. Recording the figure lets a human or a later
    #: analysis notice, which is the honest division of labour.
    #: **A tally, not a running sum.** The values being summarised are `uint8`, so 256 counts are
    #: a complete and exact description of every frame written -- mean and spread fall out of it
    #: with no pass over the pixels at all. What this replaces is the reason it exists:
    #: `(as_bytes.astype("float64") ** 2).sum()` widened the whole chunk to 8 bytes a channel and
    #: then squared it, two temporaries of 8x the output for a number that never needed more than
    #: 256 integers. At 4K that is 38.2 GB each; at 8K, 153 GB each, on a host that was already
    #: being killed by its cgroup with the card half empty.
    levels = np.zeros(256, dtype=np.int64)
    _stream.last_output_size = None
    _stream.last_ratchet = []
    _stream.last_capture = capture
    #: Output frames still to be discarded — the lead-in re-read for temporal context. Zero on the
    #: first pass, which is every ordinary job, and costs nothing there.
    skip = 0
    #: **The caller's dict, or one of our own if nobody passed it.** `run` creates it so that the
    #: batch hook it hangs on `PhaseWatch` and the cache the chunk loop fills are the *same
    #: object* — a rung-2 eviction reaching into a different dict would drop nothing and report
    #: so, which is the honest half of a failure that should not exist.
    if runner_cache is None:
        runner_cache = {}
    #: Where the current generator began in the source, and where its next frame lands in
    #: `carried_alpha`. Distinct from `frames_written` because a discarded frame advances these
    #: and not that.
    start_at = 0
    alpha_cursor = 0

    while True:
        try:
            for result in cli._stream_video_chunks(
                cap=capture,
                frames_to_process=max(0, frame_budget - start_at),
                chunk_size=chunk_size,
                overlap=plan["temporal_overlap"],
                args=args,
                device_id="0",
                debug=cli.debug,
                # **One runner, built once, reused by every chunk** (F-2026-08-20-45). The
                # comment here used to read "nothing is held between chunks that a later chunk
                # does not certainly need", which is true of everything except the one thing it
                # was switching off: the model. The vendored docstring calls this parameter
                # "Optional model cache dict for reuse across chunks" and we were passing None.
                #
                # Per job, not per worker: this dict is created in `run` and dies with the call,
                # and `release_runner_cache` evicts the vendored global cache in the same
                # `finally`. Chunk N finds the model already home; job N+1 does not.
                runner_cache=runner_cache,
                log_progress=False,
            ):
                # [T, H, W, C] in [0, 1] — the model's native output, now in the dtype it
                # actually computed in. Converted here rather than anywhere later, and released
                # immediately: a decoded chunk of 8K frames is the largest thing in the process
                # after the model itself.
                #
                # **A slice at a time, and the finite check rides along.** Both used to run over
                # the whole chunk, and between them they allocated four buffers the size of the
                # output — a bool mask and the three float32 stages of `clamp`/`*255`/`round`.
                # At 8K that is 115 GB of host memory to produce a uint8 array of 25, on a
                # container whose cgroup was already killing it. The arithmetic is unchanged:
                # each slice is widened to float32 exactly as the whole tensor was, so every
                # value that comes out is the value that came out before.
                as_bytes = _to_bytes(cli, result)

                # **The lead-in, dropped before anything else looks at it.** These frames exist to
                # give the model back the temporal context a restart took away; they were already
                # written by the generator that OOMed, and writing them again would duplicate
                # frames in the master while every count still agreed with itself.
                if skip:
                    dropped = min(skip, as_bytes.shape[0])
                    skip -= dropped
                    alpha_cursor += dropped
                    as_bytes = as_bytes[dropped:]
                    if as_bytes.shape[0] == 0:
                        del as_bytes, result
                        continue

                # **The model's output size is adopted, never predicted.**
                # `estimator.output_dimensions` computes what the caller asked for; the model
                # returns what its own internal rounding produced, and the two disagree at large
                # scales — measured 8208x4320 against a predicted 8210x4320 on a 973x512 source at
                # a 4320 target. A still caught it as a byte-count refusal; a video would not have,
                # because rawvideo carries no shape and ffmpeg would have skewed every frame
                # against the declared `-s` while exiting 0.
                #
                # No rounding rule is inferred here. Three observed widths would fit several, and
                # fitting a rule to points you chose is how the white rectangle survived six
                # diagnoses (`docs/decisions.md` 3.9). The frame in hand knows its own shape; that
                # is the authority.
                # **Fitted to the caller's canvas before anything downstream sees a size.** The
                # estimator asks the model for a short edge that *covers* the request, so this
                # only ever shrinks — and it must happen here rather than after the master is
                # written, because the writer, the identity tag, the manifest and the derives all
                # take the frame's shape as given.
                if exact_size is not None and \
                        (as_bytes.shape[2], as_bytes.shape[1]) != tuple(exact_size):
                    as_bytes = _fit_to(cli, as_bytes, exact_size)

                if frames_written == 0:
                    _stream.last_output_size = (as_bytes.shape[2], as_bytes.shape[1])
                    writer.set_frame_size(as_bytes.shape[2], as_bytes.shape[1])

                # **The model was asked to carry alpha; check that it did.** `is_rgba` is inferred
                # upstream from the tensor's last dimension, so anything that quietly converts to
                # three channels on the way in leaves this path writing RGB bytes into a writer
                # expecting RGBA — every pixel shifted by one channel, which is a picture-shaped
                # result that is wrong everywhere. Refusing beats a plausible-looking master, and
                # this is the whole reason the flag can be trusted enough to test.
                if expect_channels is not None:
                    _assert_channels(as_bytes, expect_channels)

                # Reattach the alpha the reader took out, resized to whatever the model produced.
                # Frames are consumed in the order they were read, so the alpha for the *n*th
                # frame read is the *n*th entry — which is `alpha_cursor`, not `frames_written`,
                # once a restart has re-read frames that were already written.
                if carried_alpha is not None:
                    import numpy as np

                    # **The two counters must agree here, and one test could not tell them
                    # apart.** `alpha_cursor` advances on every frame *yielded* and
                    # `frames_written` on every frame *written*, which differ only across the
                    # discarded lead-in — and since `start_at + skip == frames_written` by
                    # construction, they are provably equal again by the time any frame is
                    # indexed. Deliberately breaking one into the other left the suite green.
                    #
                    # So the equality is asserted rather than argued. It is free, it states the
                    # invariant where it is relied on, and if a later change to the skip logic
                    # breaks it the failure is a message rather than an alpha channel that lags
                    # its colour by a chunk — which looks like a soft-edge artefact, not a bug.
                    if alpha_cursor != frames_written:
                        raise WorkerError(
                            INTERNAL,
                            "alpha bookkeeping drifted: cursor {} against {} frames written. "
                            "Every frame from here would carry another frame's alpha."
                            .format(alpha_cursor, frames_written),
                        )

                    height, width = as_bytes.shape[1], as_bytes.shape[2]
                    merged = np.empty((as_bytes.shape[0], height, width, 4), dtype=as_bytes.dtype)
                    merged[..., :3] = as_bytes
                    for index in range(as_bytes.shape[0]):
                        source_alpha = carried_alpha[alpha_cursor + index]
                        merged[index, ..., 3] = resize_alpha(source_alpha, width, height)
                    as_bytes = merged

                _tally_levels(levels, as_bytes)
                for frame in as_bytes:
                    writer.write(frame.tobytes())
                    frames_written += 1
                    # **F-2026-08-18-11: the count advances while the chunk is still running.**
                    # `on_chunk` used to fire only when a chunk landed, and a chunk holding the
                    # whole clip is the designed normal — so a 192-frame job reported `0/192` for
                    # its entire run and then `192/192` once. The `working()` heartbeat proved
                    # the worker was alive but could not move `frames_done`, because at the time
                    # it fired no frame had been written.
                    #
                    # Here a frame *has* been written: it is in the encoder's pipe, it is in the
                    # master, and it will not be produced again. So the number is true rather
                    # than optimistic, which is the property the ETA and the deadline guard both
                    # depend on. Every 16 rather than every frame, because the payload is a
                    # RunPod progress update and a 33 MP job would otherwise post thousands.
                    # **`boundary=False`: this moves the count, never the rate.** Model time is
                    # paid per chunk and the frames appear in a burst at the end of it, so
                    # `elapsed / frames_so_far` mid-chunk divides a whole chunk's compute by the
                    # handful written so far — an ETA inflated by `chunk / _PROGRESS_EVERY`, which
                    # at 8% through a 192-frame chunk is twelvefold (F-2026-08-18-26).
                    if on_chunk is not None and frames_written % _PROGRESS_EVERY == 0:
                        on_chunk(frames_written, boundary=False)
                alpha_cursor += as_bytes.shape[0]
                del as_bytes, result
                # Always at the chunk boundary too, whatever the modulus landed on, so the last
                # partial group is never left unreported.
                if on_chunk is not None:
                    on_chunk(frames_written)
            break
        except Exception as exc:  # noqa: BLE001 — classified by the ratchet, re-raised if not its
            # **Only an OOM, and only when the caller supplied a policy.** Everything else — a
            # non-finite tensor, a frame of the wrong length, a decoder that died — propagates
            # untouched. A recovery loop that swallowed those would turn a diagnosable failure
            # into an infinite one.
            if ratchet is None or not ratchet.handles(exc):
                raise
            step = ratchet.step(plan, frames_written, exc)
            # None means the ladder is spent. The OOM goes up as it always did, and the caller's
            # own refusal explains what was tried — with `last_ratchet` recording that the frames
            # already written are gone, so nothing above restarts the clip believing it is cheap.
            if step is None:
                raise

            plan = step["plan"]
            args = step["args"]
            # **The residency schedule restarts with the stream** — see `ResidencySchedule
            # .restart`. A rank-decrease cannot see this boundary, and the state that survives it
            # is state armed for a plan that no longer exists.
            if schedule is not None:
                schedule.restart(plan)
            chunk_size = plan["chunk_size"]
            # **Let go of the decoder that failed before taking another.** A job that ratchets
            # more than once would otherwise hold an open capture per attempt — each with a file
            # handle and the vendored reader's buffers on it — for the rest of a run measured in
            # hours. `release` is idempotent, so the caller releasing this again on its way out is
            # harmless.
            try:
                capture.release()
            except Exception:  # noqa: BLE001 — a decoder we cannot close must not fail the job
                pass
            capture = step["capture"]
            _stream.last_capture = capture
            # The lead-in cannot reach back further than the frames that exist.
            skip = min(int(step.get("lead_in") or 0), frames_written)
            start_at = frames_written - skip
            # **Truncated, not cleared.** The entries before the resume point belong to frames
            # already written and are never read again; the entries after it are about to be
            # appended a second time by the re-read, and leaving them would put every subsequent
            # alpha one chunk out of step with its frame.
            if carried_alpha is not None:
                del carried_alpha[start_at:]
            alpha_cursor = start_at

            # Position the decoder by reading and discarding. `CAP_PROP_POS_FRAMES` is the obvious
            # way and it is unreliable on B-frame and variable-rate sources — landing a frame or
            # two off, silently, which here would mean a master that skips or repeats content
            # while every count still balanced. Decoding to the point is exact, and cheap against
            # a chunk: the source is small, and it is the output that is 4K.
            for _ in range(start_at):
                ok, _frame = capture.read()
                if not ok:
                    raise WorkerError(
                        INTERNAL,
                        "resuming after an out-of-memory needed frame {} of the source but the "
                        "decoder ran out at frame {}. The master already holds {} frames and "
                        "cannot be completed.".format(start_at, _ + 1, frames_written),
                    )
            _stream.last_ratchet.append(step["record"])

    if frames_written == 0:
        raise WorkerError(INVALID_SOURCE, "the source decoded to no frames")

    # Mean and spread of everything written, on the 0-255 scale. A flat output has a standard
    # deviation near zero — which is what a broken run and a photograph of a white wall have in
    # common, and why this is reported rather than judged.
    #
    # **Read off the tally, and identical to what the old two passes produced.** Every quantity
    # here is an exact integer well inside a float64's 53-bit mantissa — the largest, the sum of
    # squares over an 8K clip, is under 2^49 — so this is the same arithmetic, not an
    # approximation of it.
    values = np.arange(256, dtype=np.int64)
    pixel_n = int(levels.sum())
    pixel_sum = float((levels * values).sum())
    pixel_sq = float((levels * values * values).sum())
    mean = pixel_sum / pixel_n if pixel_n else None
    variance = (pixel_sq / pixel_n - mean * mean) if pixel_n else None
    _stream.last_pixel_stats = {
        "mean": round(mean, 2) if mean is not None else None,
        "std": round(max(variance, 0.0) ** 0.5, 2) if variance is not None else None,
    }
    return frames_written

#: Frames converted to `uint8` in one pass. Sized so the transient float32 buffer stays around a
#: gigabyte at 8K (8 frames x 99.5M channels x 4 bytes x the three stages) rather than scaling
#: with the clip. Any value gives identical output; this one trades a negligible number of extra
#: kernel launches for a bounded peak.
_BYTES_SLICE = 8


#: The ids `inference_cli` uses when the CLI caches are on. Fixed strings on its side, so
#: eviction is a lookup rather than a guess.
_CLI_DIT_ID, _CLI_VAE_ID = "cli_dit", "cli_vae"

#: The plan's word for rung 2. A literal rather than an import of `planner.EVICTED`: this module
#: is the executor and the planner is embeddable stdlib-only code CF ships elsewhere, so the two
#: are asserted equal by rung 1 instead — the check that catches a drift an import would merely
#: have made impossible to notice.
EVICTED_RESIDENCY = "evicted"

#: The vendored phase order inside one chunk. A rank that goes *down* between two boundaries is a
#: seam: the loop has started the next chunk (F-2026-08-21-54).
_PHASE_RANK = {"vae_encode": 0, "dit_sample": 1, "vae_decode": 2, "postprocess": 3}

#: **The phases an eviction may fire in** — the ones that need *no* model. Amendment 9 says
#: "unload after each chunk's DiT", and *after the DiT phase* is a boundary in a later phase, not
#: the last batch of the DiT phase itself. The distinction is not pedantry:
#: `generation_phases.py:660` logs `Upscaling batch N/M` at the TOP of the loop body and uses the
#: model fifty lines later in the same iteration, so a hook that fired on the last DiT batch
#: nulled the weights *before that batch ran*.
#:
#: **`vae_decode` is not on this list, and leaving it here was the same bug on the other model.**
#: The eviction nulls `runner.vae` as well as `runner.dit`, and the decode loop has the identical
#: shape: `generation_phases.py:921` logs `Decoding batch N/M` at the top of the body,
#: `:937` calls `runner.vae_decode(...)` sixteen lines later in the same iteration. Firing at a
#: decode boundary therefore kills the decode it was announced by. Found in review of the fix for
#: F-2026-08-21-54, before it ever ran — the defect this whole finding is about, transplanted
#: verbatim onto the VAE by the person fixing it.
#:
#: Phase 4 references no model at all (`generation_phases.py:1070` onward holds no `runner.`
#: attribute access), and every chunk emits at least one `Post-processing batch N/M`
#: (`:1224`) — so an arming made at a decode boundary, which is the only place the guard may
#: reach a verdict, still has a firing point inside its own chunk.
MODEL_FREE_PHASES = ("postprocess",)

#: How much of the checkpoint an eviction has to give back before it counts as having happened.
#: Half, not all: the container is allocating on other threads while this runs, so demanding the
#: whole 16.4 GiB would call a working eviction a failure. Demanding *something* is the point —
#: the seam retest's eviction moved anon 41.0 to 40.89 and was recorded as a success.
EVICTION_PROVEN_GIB = 8.0


class ResidencySchedule(object):
    """Which rung the run is *executing* on — mutable, because the guard may promote it.

    The plan decides the rung before a GPU-second is spent, and that decision is right almost
    always. Almost: amendment 9 also puts rung 2 in the F-42 guard's hands as its first remedy,
    so the number the eviction hook consults has to be one two parties can see. A boolean copied
    into a closure at construction is a decision nobody can revise, and the case where it needs
    revising is exactly the case where a container is about to die.
    """

    def __init__(self, plan=None):
        self.rung_two = (plan or {}).get("residency") == EVICTED_RESIDENCY
        #: Set when the *guard* promoted rather than the plan, so the record can tell a run that
        #: was planned onto rung 2 from one that was driven there by drift. The second is a
        #: calibration finding; the first is a Tuesday.
        self.promoted_in_flight = False
        #: **What the PLAN decided**, kept separate from what the run is currently doing. A
        #: seam returns the run to this (F-2026-08-21-54): a rescue armed for one chunk's geometry
        #: is not a fact about the next chunk, and the next chunk re-decides from its own.
        self.planned_rung_two = self.rung_two
        #: **Whether this chunk's eviction is still ahead of us** (F-2026-08-21-50). The guard
        #: credits the checkpoint against its projection only while the eviction is *pending*:
        #: before it, the 16.4 GiB is a promise about the peak phase and the projection should
        #: hold it; after it, the live reading already reflects whatever was actually freed, and
        #: crediting again would under-project by the whole checkpoint — which is the one
        #: direction that loses a container rather than delaying a job.
        self.evicted_this_chunk = False
        #: What the last eviction measured, for the record.
        self.last_freed_gb = None
        #: **Set when an eviction gave nothing back** (F-2026-08-21-54, part 3). The seam retest's
        #: eviction moved anon 41.0 to 40.89 — the pointer nulled, the cost kept — and the run
        #: carried on as though 16.4 GiB had been handed over. A mechanism that has demonstrably
        #: failed is not a promise, so it stops being credited.
        self.eviction_failed = False
        #: The rank of the last phase seen, and how many seams have been crossed. The guard reads
        #: the ordinal to know when to let a chunk re-decide.
        self.last_phase_rank = None
        self.chunk_ordinal = 0

    @property
    def pending(self):
        """`True` when an eviction is scheduled for this chunk, unspent, and still believable."""
        return bool(self.rung_two) and not self.evicted_this_chunk and not self.eviction_failed

    def observe_phase(self, phase):
        """Follow the loop across chunk boundaries. Returns `True` if this boundary was a seam.

        **A rank that goes down is a new chunk** — the vendored loop runs encode, DiT, decode,
        postprocess per chunk, so the only way to see `vae_encode` after `postprocess` is that
        the seam has been crossed.
        """
        rank = _PHASE_RANK.get(phase)
        if rank is None:
            return False
        crossed = self.last_phase_rank is not None and rank < self.last_phase_rank
        self.last_phase_rank = rank
        if crossed:
            self.cross_seam()
        return crossed

    def restart(self, plan=None):
        """**A mid-stream ratchet re-enters the vendored loop, and that is a boundary too.**

        Found in review. `observe_phase` infers a seam from a phase rank going *down*, which is
        true of the chunk-to-chunk case and false of a restart: an OOM inside `vae_encode` steps
        the plan, re-opens the capture and begins again at `vae_encode`, and rank 0 is not below
        rank 0. Everything then survives that should not — `evicted_this_chunk` from the attempt
        that died, a `rung_two` armed for a geometry that has since changed, and a
        `chunk_ordinal` that never moves, which leaves the guard unable to promote for the rest
        of the run.

        So the ratchet says so explicitly rather than relying on an inference that does not hold.
        The new plan is re-read, because a ratchet changes the very geometry the rung was chosen
        from.
        """
        if plan is not None:
            self.planned_rung_two = (plan or {}).get("residency") == EVICTED_RESIDENCY
        self.last_phase_rank = None
        self.cross_seam()

    def cross_seam(self):
        """**A new chunk re-decides from its own geometry** (F-2026-08-21-54, part 2).

        An in-flight promotion dies here. It was a judgement about one chunk's container made
        from one chunk's readings, and carrying it forward is how an arming made during chunk 1's
        decode found its firing point in chunk 2's DiT. What survives is the *plan's* rung, which
        is a fact about the job rather than a rescue.
        """
        self.chunk_ordinal += 1
        self.rung_two = self.planned_rung_two
        self.evicted_this_chunk = False
        self.eviction_failed = False

    def record_eviction(self, verdict):
        """Bank what an eviction actually gave back, and believe it only if it gave something."""
        verdict = verdict or {}
        self.evicted_this_chunk = True
        self.last_freed_gb = verdict.get("freed_gb")
        self.eviction_failed = (self.last_freed_gb is None
                                or self.last_freed_gb < EVICTION_PROVEN_GIB)
        return not self.eviction_failed

    def promote(self, phase=None):
        """Arm the eviction for the rest of the run. Idempotent, and never raises."""
        if not self.rung_two:
            self.promoted_in_flight = True
        self.rung_two = True
        return True


def _scheduling_eviction(on_batch, plan, runner_cache, debug=None, evict=None, schedule=None,
                         log=print):
    """Rung 2's schedule, hung on the one hook that fires inside the vendored chunk loop.

    **Rung 1 gets the caller's hook back, unwrapped.** "Run exactly as today" is the amendment's
    own wording for the common case, and a wrapper that only ever forwards is still a wrapper —
    it changes what a traceback says and what a reader has to check.

    The trigger is the *last* DiT batch of a chunk. Before it, the model is being used; after it,
    nothing in the chunk needs it again — the decode, the postprocess and the assembly drain that
    follow are the phases the amendment names as the run's actual peak. The next chunk finds an
    empty cache and re-materialises, which is the same path every build before F-45 took on every
    chunk, so the reload is not new code being trusted for the first time.

    **Evict before the guard samples, not after.** The same hook carries F-42, and a guard that
    reads the container in the instant before 16.4 GiB is handed back would refuse a job that was
    about to fit. The eviction cannot raise, so the refusal path through this wrapper is
    unchanged.
    """
    if schedule is None and (plan or {}).get("residency") != EVICTED_RESIDENCY:
        # **No ladder in play at all**: rung 1 with no promotion channel wired gets its own hook
        # back, unwrapped. A wrapper that only ever forwards still changes what a traceback says.
        return on_batch
    if schedule is None:
        schedule = ResidencySchedule(plan)
    if evict is None:
        evict = evict_model

    def hook(phase, index, total):
        # **The seam first, so an arming from the previous chunk dies before it can fire**
        # (F-2026-08-21-54, part 2).
        schedule.observe_phase(phase)
        # **And the firing point is a phase that does not need the model** (part 1). Amendment 9
        # says "unload after each chunk's DiT"; the first boundary of the phase *after* DiT is
        # what that means. An arming made during chunk 1's decode fires at that same decode
        # boundary — immediately, in the phase it was armed in — rather than waiting for a
        # "last DiT batch" that this chunk has already had and that only exists again across
        # the seam.
        if schedule.rung_two and phase in MODEL_FREE_PHASES and not schedule.evicted_this_chunk:
            # Idempotent by construction: a second call finds nothing left to drop, so a repeated
            # boundary line costs a `gc.collect()` and reports honestly that it freed nothing.
            freed = schedule.record_eviction(evict(runner_cache, debug=debug))
            if not freed and log is not None:
                # **Reported, not swallowed** (part 3). An eviction that nulls the pointer and
                # keeps the memory is the worst of both, and the run has just been priced as
                # though 16.4 GiB came back. Saying so is what stops the next chunk being
                # credited for it — see `ResidencySchedule.pending`.
                log("[host] eviction FREED NOTHING: {} GiB given back, {} needed to count. The "
                    "model is gone and the memory is not; this run is no longer credited for it."
                    .format("unreadable" if schedule.last_freed_gb is None
                            else "{:.2f}".format(schedule.last_freed_gb), EVICTION_PROVEN_GIB))
        return on_batch(phase, index, total) if on_batch is not None else None

    return hook


def _drop_model_references(runner_cache=None, debug=None):
    """Let go of **every** reference this process holds to the model. Returns what it dropped.

    One implementation, because there are two callers with the same requirement and a second
    copy is the one nobody updates: the end-of-job release (F-45) and rung 2's mid-job eviction
    (amendment 9) must both leave zero references or neither frees anything.

    **The vendored cache is not the only holder, and it is not even the important one.** The
    global cache keeps the model under a fixed node id; the runner keeps it as `runner.dit`; and
    `runner_cache["runner"]` — a dict this worker owns — keeps the runner. Clearing the vendored
    side alone leaves the model reachable through our own dict, and a host eviction that frees
    nothing is worse than none: the plan sized the chunk for a container that did not appear.

    `release_model_memory` in the vendored tree is *not* used here, and the reason is the whole
    reason rung 2 needed new code. It frees `param.data` only where `param.is_cuda or
    param.is_mps` (`src/optimization/memory_manager.py:563`), so at `b=36` — where the DiT's
    swapped blocks live on the CPU by design — it walks past the 16.4 GiB this rung exists to
    reclaim. Dropping references and collecting is what returns host memory; the CUDA side comes
    back the same way, by deallocation, without a device move at exactly the moment the container
    can least afford one.
    """
    dropped = []
    # Ours first. The vendored cache cannot see this dict, so an eviction that starts on the far
    # side leaves the near side holding the model through the whole of the peak phase.
    if isinstance(runner_cache, dict):
        runner = runner_cache.pop("runner", None)
        if runner is not None:
            for attr in ("dit", "vae", "sampler", "schedule", "sampling_timesteps"):
                if getattr(runner, attr, None) is not None:
                    try:
                        setattr(runner, attr, None)
                    except Exception:  # noqa: BLE001 — a read-only attribute is not fatal here
                        pass
            dropped.append("runner")
        ctx = runner_cache.get("ctx")
        if isinstance(ctx, dict) and ctx.pop("cache_context", None) is not None:
            # `cache_context` carries `cached_dit`/`cached_vae` straight from `prepare_runner`.
            dropped.append("cache_context")
    try:
        from src.core.model_cache import _global_cache  # noqa: PLC0415 — vendored, GPU-only
    except Exception:  # noqa: BLE001 — no vendored tree (rung 1), or it moved: nothing more
        return dropped
    for remove, node_id in ((getattr(_global_cache, "remove_dit", None), _CLI_DIT_ID),
                            (getattr(_global_cache, "remove_vae", None), _CLI_VAE_ID)):
        if remove is None:
            continue
        try:
            if remove({"node_id": node_id}, debug):
                dropped.append(node_id)
        except Exception:  # noqa: BLE001 — see `release_runner_cache`
            pass
    return dropped


def _empty_cuda_cache():
    """Hand the freed blocks back to the driver. Never raises, never required."""
    try:
        import torch  # noqa: PLC0415 — GPU-only

        torch.cuda.empty_cache()
        return True
    except Exception:  # noqa: BLE001 — rung 1 has no torch and needs none
        return False


def evict_model(runner_cache=None, debug=None, read_anon=None, log=None, why="the peak phase"):
    """**Rung 2's eviction, measured rather than intended** (amendment 9, Build D).

    Drops every reference, collects, and reads the container's unreclaimable memory on both
    sides of it. The amendment is explicit that the eviction "must provably free (asserted on
    the anon drop, not on intent)" — so this returns the drop it measured, and a caller that
    wants to assert something has a number to assert it on.

    `gc.collect()` and not refcounting alone: `_protect_model_from_move` patches the model with
    a closure over the runner, and a cycle is freed by the collector or not at all.

    Never raises. An eviction that cannot happen leaves the run exactly where it was, and the
    F-42 guard — which is watching the same axis this reads — is what refuses if the container
    then goes where the plan said it would not.
    """
    if read_anon is None:
        read_anon = _anon_gb
    before = read_anon()
    dropped = _drop_model_references(runner_cache, debug)
    gc.collect()
    _empty_cuda_cache()
    after = read_anon()
    freed = None if (before is None or after is None) else round(before - after, 2)
    verdict = {"dropped": dropped, "anon_before_gb": before, "anon_after_gb": after,
               "freed_gb": freed}
    _bank_eviction(before, after)
    if log is not None:
        # **Printed with the number, or not printed at all.** "Evicted the model" with nothing
        # beside it is the sentence F-42 was filed about: a narration that reads like an action.
        log("[host] model evicted for {}: dropped {} · anon {} -> {} ({})".format(
            why,
            ",".join(dropped) or "nothing",
            "?" if before is None else "{:.2f}".format(before),
            "?" if after is None else "{:.2f}".format(after),
            "unreadable" if freed is None else "{:+.2f} GiB".format(-freed)))
    return verdict


def _bank_eviction(before, after):
    """An eviction is a host event, so it lands in the corpus and not only in the log.

    Same door as every other host reading (`phasewatch.observe`) — the four hand-banked banners
    used to be the whole corpus while everything else was printed and dropped, and a rung whose
    only evidence is a log line is a rung nobody can reprice later.
    """
    try:
        import phasewatch  # noqa: PLC0415

        phasewatch.observe("model-evicted", after if after is not None else 0.0)
    except Exception:  # noqa: BLE001 — banking a reading may never cost a job
        pass


def _anon_gb():
    """The container's unreclaimable memory, or `None` off a cgroup."""
    try:
        import hardware  # noqa: PLC0415 — stdlib-only

        return hardware.memory_breakdown_gb().get("anon")
    except Exception:  # noqa: BLE001
        return None


def release_runner_cache(debug=None, runner_cache=None, log=None):
    """Put the between-jobs posture back: evict the model this job cached (F-2026-08-20-45).

    **The fix for F-45 is within-job reuse, and nothing more.** Turning on `--cache_dit` also
    populates the vendored `GlobalModelCache`, which is keyed on a fixed id and would otherwise
    outlive the job — quietly making this worker model-resident between jobs. That may well be
    the right thing; it is not this wave's call. CF has ratified a residency-scheduling ladder as
    Build D, and a fix that pre-empted it by side effect would be the worst way to arrive there:
    a policy nobody chose, delivered by a bug fix.

    So the reuse lasts exactly as long as the chunks that need it. Called from a `finally`, so a
    refusal or a crash frees it too — the run that most needs the memory back is the one that
    just failed to fit in it.

    Never raises. An eviction that cannot find its target has nothing to report and nothing to
    fail; the next job builds a runner either way.
    """
    #: **Handed the job's own dict, and measured like every other eviction** (found in review).
    #: This was called as `release_runner_cache()` from the handler's `finally` — no dict, so
    #: `_drop_model_references` skipped its own near-side branch entirely and cleared only the
    #: vendored global cache, leaving `runner_cache["runner"].dit` holding 16.4 GiB. And it never
    #: collected: `evict_model`'s docstring already says why refcounting is not enough here
    #: (`_protect_model_from_move` closes over the runner, so the model is in a cycle).
    #:
    #: The reference outlives the call, which is what made it matter: `run.last_phases` is a
    #: *function attribute*, it holds the `PhaseWatch`, which holds the batch hook, which closes
    #: over this dict — so the weights stayed reachable from module scope across the idle window
    #: and, worse, across a retry. Attempt 2 then materialises a second copy beside attempt 1's,
    #: which is F-45 relocated to the attempt boundary, on the run that just failed to fit.
    if runner_cache is None:
        runner_cache = getattr(run, "last_runner_cache", None)
    verdict = evict_model(runner_cache, debug=debug, log=log, why="the end of the job")
    return bool(verdict["dropped"])


def _to_bytes(cli, result):
    """`[T, H, W, C]` in [0, 1] to a `uint8` numpy array, a few frames at a time.

    **Identical values, bounded buffers.** Each slice is widened to float32 before the clamp and
    the round, which is exactly what the vendored source used to do to the whole canvas before
    handing it over — so this is the same arithmetic on the same numbers, and the only thing that
    changes is how much of it exists at once.

    **The non-finite refusal moved in here** rather than running over the whole tensor first, for
    the same reason — it allocated a bool mask the size of the output to answer one question.
    Every slice is checked before any of them is written, so the refusal is exactly as early as
    it was. `clamp` does not fix `NaN`: it passes straight through, and converting it to `uint8`
    is undefined — in practice it lands on a flat value, so the worker writes a blank image,
    uploads it, writes a manifest, and reports success. Every structural check passes and the
    customer receives nothing, which is precisely the failure this worker exists to prevent,
    arriving through the one door that was not watched.

    `internal` and retryable: a numerical failure is worth one more attempt at a more
    conservative configuration, and it is not the caller's fault.
    """
    import numpy as np  # noqa: PLC0415 — kept off module scope like every other use here

    torch = cli.torch
    parts = []
    for start in range(0, result.shape[0], _BYTES_SLICE):
        piece = result[start:start + _BYTES_SLICE].to(torch.float32)
        if not bool(torch.isfinite(piece).all()):
            raise WorkerError(
                INTERNAL,
                "the model produced non-finite values (NaN or infinity) for this chunk; "
                "refusing to encode them rather than write a blank image that would pass "
                "every check",
            )
        parts.append((piece.clamp(0, 1) * 255.0).round().to(
            "cpu", dtype=_uint8(cli)).numpy())
        del piece
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


#: How many elements one `bincount` sees. It is the only allocation this tally makes, and
#: `bincount` widens its input to a pointer-sized integer to index with, so the slice governs the
#: temporary: 4M elements is 32 MB, against the 153 GB the float64 pass it replaces asked for at
#: 8K. Small enough to be invisible, large enough that the per-call overhead disappears — a 4K
#: frame is six slices, an 8K frame twenty-four.
_TALLY_SLICE = 1 << 22

#: Frames between progress updates inside a chunk. Small enough that a poller sees movement on
#: any job worth polling, large enough that a long 8K run does not spend its time posting.
#: **Below `_TALLY_SLICE`'s comment, not above it** — inserted in the middle, it left four lines
#: about `bincount`'s temporary sitting over an unrelated constant (F-2026-08-18-26).
_PROGRESS_EVERY = 16


def _tally_levels(levels, as_bytes):
    """Add one chunk's `uint8` values into a 256-bin tally, in bounded pieces.

    Sliced rather than counted whole because the widening `bincount` does internally is
    proportional to what it is handed, and handing it a whole 8K chunk would reintroduce exactly
    the temporary this replaces.
    """
    import numpy as np  # noqa: PLC0415 — kept off module scope like every other use here

    flat = as_bytes.reshape(-1)
    for start in range(0, flat.size, _TALLY_SLICE):
        levels += np.bincount(flat[start:start + _TALLY_SLICE], minlength=256)


def _uint8(cli):
    return cli.torch.uint8


def assert_source_exhausted(cli, capture, frames_written):
    """The check `docs/decisions.md` 0.2 exists for — and it asks the right question.

    **The obvious check is worthless.** Comparing "frames decoded" against "frames written" comes
    out equal by construction: both are counted by the same loop, so it would pass on every run
    including the one that dropped half the source. That is exactly the shape playbook §12 warns
    about — a verification that cannot fail is not a verification.

    The failure that can actually happen is stopping *early*: `_stream_video_chunks` runs
    `while frames_read < frames_to_process`, so a budget smaller than the source silently
    truncates, and the output is a shorter video that plays perfectly. CF's own trim produces
    exactly the sources where a container's count under-reports what decodes.

    So the question is whether the decoder had anything left, and the way to ask it is to read
    one more frame. If one comes back, frames were dropped.

    `internal` rather than `invalid_source`: the source decoded fine and the worker lost frames.
    """
    leftover, _ = capture.read()
    if leftover:
        raise WorkerError(
            INTERNAL,
            "the decoder still had frames after {} were written: the frame budget truncated the "
            "source. The output would be short while looking correct.".format(frames_written),
        )
