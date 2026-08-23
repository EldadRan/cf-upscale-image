"""Route C: `capture → INTERP → encoder`. No upscaler, no SeedVR2, no plan-and-retry ladder.

**The route with no model of ours in it.** Contract §6b: route C never loads SeedVR2, so this
path shares nothing with `_upscale_once` beyond the decoder and the writer — no rung ladder, no
residency schedule, no host guard promotion, none of which describe anything about a job whose
working set is one frame pair.

The pieces are already built and this is the wiring between them:

    pipeline.open_source   →   interpolate.Interpolator(rife.Rife)   →   encoder.MasterWriter

**cv2 arrives through the vendored CLI, deliberately** (CF, 2026-08-23). Importing
`inference_cli` is not loading a model: at module scope it inserts a path, sets env, sets the
spawn start method and imports torch, cv2 and numpy — the checkpoint load lives behind functions
route C never calls. And its process-global effects already happen on every production job, so
taking the vendored path inherits behaviour proven in the field, where importing cv2 directly
would have been the novel one and would have cost the deliberate one-cv2-per-process property.
"""
import numpy as np

from errors import INVALID_SOURCE, WorkerError

#: BGR uint8 out of cv2, RGB float in [0, 1] for RIFE, rgb24 bytes for the writer. Stated once
#: here because a channel order that is wrong is a picture that still plays.
_CHANNELS = 3


def source_frame_count(source):
    """How many frames the plan is sized from, and why it is not read from the container.

    `probe.probe_source` refuses to report a frame count on arbitrary input for a documented
    reason: CF's `media_trim` bounds a video with an MP4 edit list, leaving frames in the stream
    the container still counts, so a probe and a decode disagree on exactly the files CF sends.
    That argument holds here.

    So the count is **derived from two measured quantities** — the stream's own duration and its
    rate — and then checked against the decode. `stream()` already refuses a source shorter than
    the count it was given; `retime` below refuses one longer. Between them the derivation is
    verified in both directions rather than trusted.
    """
    fps = source.get("fps")
    duration = source.get("video_duration_s") or source.get("duration_s")
    if not fps or not duration:
        raise WorkerError(
            INVALID_SOURCE,
            "a retime needs the source's rate and duration and this container reports "
            "fps={!r} duration={!r}. Neither is guessed: the frame plan is sized from them and a "
            "wrong size is a wrong output length.".format(fps, duration))
    return int(round(float(duration) * float(fps)))


def _to_tensor(frame_bgr, torch):
    """cv2's BGR uint8 `H×W×3` to RIFE's RGB float `1×3×H×W` in [0, 1]."""
    rgb = frame_bgr[:, :, ::-1]
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _to_rgb24(tensor):
    """Back to the writer's contract: `rgb24`, `width × height × 3` bytes.

    Clamped before the cast because a synthesis is a model output and can land a hair outside
    [0, 1]; `uint8` would wrap that into a black pixel in a white region, which is the 16-bit
    downconvert defect in miniature.
    """
    array = tensor.detach().to("cpu", copy=False).float().clamp_(0.0, 1.0)[0]
    array = (array.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(array).tobytes()


def frames_from(capture, cli):
    """Decode the source into cv2 frames, in order, until it is exhausted."""
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            return
        yield frame


def retime(cli, source, source_path, master_path, interpolator, target_fps, identity,
           snap_tolerance=None, crf=None, audio_source=None, progress=None):
    """Decode, interpolate, encode. Returns the stats the plan produced.

    `snap_tolerance` is passed through as given and **is not defaulted here** (contract §5c): a
    tolerance defaulted to zero would ship the unsnapped plan as the ruled answer before the
    benchmark that decides it has run. `None` means unruled, and the shim reads it as zero for
    arithmetic while the request records that nobody chose.
    """
    import encoder  # noqa: PLC0415 — imported here so this module stays importable without one
    import interpolate  # noqa: PLC0415

    from pipeline import open_source  # noqa: PLC0415

    capture, shape = open_source(cli, source_path)
    try:
        width, height = shape["width"], shape["height"]
        n_in = source_frame_count(source)
        stream = interpolator.stream(
            _tensors(frames_from(capture, cli)),
            n_in=n_in, src_fps=shape["fps"], dst_fps=target_fps,
            tol=snap_tolerance or 0.0)

        writer_cm = encoder.MasterWriter(
            master_path, width, height, float(target_fps), identity,
            audio_source=audio_source, audio_codec=source.get("audio_codec"),
            audio_limit_s=source.get("video_duration_s"),
            crf=crf if crf is not None else encoder.DEFAULT_CRF)
        with writer_cm as writer:
            for frame in stream.frames:
                writer.write(_to_rgb24(frame))
        # **The other half of the derived count.** `stream()` refuses a source shorter than the
        # plan; this refuses one longer. A container whose duration and rate imply fewer frames
        # than it holds would otherwise deliver a silently truncated retime.
        surplus = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            surplus += 1
        if surplus:
            raise WorkerError(
                INVALID_SOURCE,
                "the source holds {} frame(s) beyond the {} its duration and rate imply, so the "
                "retime would have been truncated. The container's own numbers disagree with its "
                "content.".format(surplus, n_in))
        return stream.stats
    finally:
        if capture is not None:
            capture.release()


def _tensors(frames):
    """cv2 frames to tensors, lazily, one at a time — never a list, whatever the clip length."""
    import torch  # noqa: PLC0415 — the vendored CLI has already imported it by the time we run

    for frame in frames:
        yield _to_tensor(frame, torch)
