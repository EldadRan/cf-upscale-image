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

#: **Slack on the derived count, in source frames.** Zero was wrong: the count comes from a
#: duration stored to the millisecond times a rate, and edit-list trims are named in
#: `source_frame_count` as the reason a container and a decode disagree on real input. §2's
#: duration bound is +-2 output frames and this is its counterpart on the input side.
SURPLUS_TOLERANCE_FRAMES = 2

#: **Route C's own encode settings, passed as an override the production path never sends**
#: (contract §8c). The 8K run was reaped in x264 at ~46 GiB while this side held one frame and a
#: cached pair: the pipe is backpressure and it was working — what filled memory was the
#: encoder's own working set, one frame in flight per encoding thread plus `medium`'s 40-frame
#: lookahead plus references, dozens of 50 MiB frames at once on a 24-core host. At 4K the same
#: arithmetic fits, which is why five 4K runs showed nothing.
#:
#: `sliced-threads=1` is the large one: threads split ONE frame rather than each taking their
#: own, which cuts frames-in-flight from dozens to one at a modest speed cost. `threads` caps the
#: frame-level parallelism that multiplies the set. `rc-lookahead` shortens the window.
#:
#: **Every value here is an ordering hint and not a prediction.** The gate modelled x264 at ~4 GiB
#: against an observed 40-plus, so nothing in this line is trustworthy until a run reports
#: `encoder_peak_rss_gb` — which is why the writer now measures it.
FRUGAL_X264 = "sliced-threads=1:threads=4:rc-lookahead=10"


def source_frame_count(source):
    """How many frames the plan is sized from, and why it is not read from the container.

    `probe.probe_source` refuses to report a frame count on arbitrary input for a documented
    reason: an upstream trim can bound a video with an MP4 edit list, leaving frames in the
    stream that the container still counts, so a probe and a decode disagree on precisely the
    files this worker is sent. That argument holds here.

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
    count = int(round(float(duration) * float(fps)))
    if count < 2:
        # `build_plan` refuses this too, but with a bare `ValueError`; every other refusal on
        # this path is a `WorkerError` the caller can read.
        raise WorkerError(
            INVALID_SOURCE,
            "a retime needs at least two source frames and this container's duration ({}s) at "
            "{} fps implies {}".format(duration, fps, count))
    return count


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
    # **`clamp`, not `clamp_`.** A copy or a hold is yielded as it arrived — the caller's own
    # tensor, deliberately never cast — and `.to("cpu", copy=False).float()` on a CPU float32
    # tensor hands back that same object, so an in-place clamp would write through to the source
    # frame. Harmless here, since a decoded frame is already in [0, 1] by construction; wrong as
    # a mechanism, and under route A or B the frames entering the shim are model output.
    array = tensor.detach().to("cpu", copy=False).float().clamp(0.0, 1.0)[0]
    array = (array.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(array).tobytes()


def frames_from(capture, cli, expect=None):
    """Decode the source into cv2 frames, in order, until it is exhausted.

    `expect` is the `(height, width)` the writer was told, checked once on the first frame.
    **The writer is sized from the container and the bytes come from the decoder**, and nothing
    else compares the two: a rotation matrix or any decoder-side adjustment would produce a byte
    count ffmpeg does not expect, and rawvideo carries no shape — so the master shears from that
    frame on while the process exits 0. `MasterWriter.write` catches a wrong LENGTH; a transposed
    frame of the same length it cannot.
    """
    checked = False
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            return
        if not checked and expect is not None:
            checked = True
            if tuple(frame.shape[:2]) != tuple(expect):
                raise WorkerError(
                    INVALID_SOURCE,
                    "the decoder returns {}x{} frames but the container reports {}x{}; the "
                    "encode is sized from the container and would shear."
                    .format(frame.shape[1], frame.shape[0], expect[1], expect[0]))
        yield frame


def retime(cli, source, source_path, master_path, interpolator, target_fps, identity,
           snap_tolerance=None, crf=None, audio_source=None, progress=None,
           variant="direct", scale=None):
    """Decode, interpolate, encode. Returns the stats the plan produced.

    `snap_tolerance` is passed through as given and **is not defaulted here** (contract §5c): a
    tolerance defaulted to zero would ship the unsnapped plan as the ruled answer before the
    benchmark that decides it has run. `None` means unruled, and the shim reads it as zero for
    arithmetic while the request records that nobody chose.
    """
    import encoder  # noqa: PLC0415 — imported here so this module stays importable without one
    import variants  # noqa: PLC0415

    from pipeline import open_source  # noqa: PLC0415

    capture, shape = open_source(cli, source_path)
    try:
        width, height = shape["width"], shape["height"]
        n_in = source_frame_count(source)
        # **Peak VRAM measured rather than reported from a panel.** Route C has no plan, so
        # nothing else on this path produces one — and §8b's `--scale` axis exists to falsify
        # `w_scaling: FLAT`, which IS this reading. Reset before and read after, so the number is
        # this job's high-water mark rather than the process's history.
        peak_reset = _reset_peak()
        stream, stats = variants.run(
            variant, interpolator,
            _tensors(frames_from(capture, cli, expect=(height, width))),
            n_in=n_in, src_fps=shape["fps"], dst_fps=target_fps,
            tol=snap_tolerance or 0.0)

        writer_cm = encoder.MasterWriter(
            master_path, width, height, float(target_fps), identity,
            audio_source=audio_source, audio_codec=source.get("audio_codec"),
            audio_limit_s=source.get("video_duration_s"),
            crf=crf if crf is not None else encoder.DEFAULT_CRF,
            x264_params=FRUGAL_X264)
        # **The peak is read on the FAILURE path too, and that is the path it exists for.** It
        # was read only after the `with` — so an encoder reaped by the kernel, which is the exact
        # event this instrumentation was added for, propagated past the read and took the sampled
        # maximum with it. A second fifty-minute run would have had its ceiling inferred from a
        # kill again, which is what the measurement was meant to end. The number now rides on the
        # refusal's own message, where the diagnostics bundle and the run-record both carry it.
        try:
            with writer_cm as writer:
                for frame in stream:
                    writer.write(_to_rgb24(frame))
        except WorkerError as exc:
            peak = writer_cm.encoder_peak_rss_gb
            if peak is None:
                raise
            raise WorkerError(
                exc.code,
                "{} — ffmpeg reached {} GiB RSS before it stopped, over {} frame(s) written "
                "with x264-params {!r}".format(
                    exc.message, peak, writer_cm.frames_written, FRUGAL_X264),
                remedy=exc.remedy, shortfall=exc.shortfall) from exc
        finally:
            # Said out loud whichever way the encode ended, because a log line survives a bundle
            # that was never written.
            print("[encode] ffmpeg peak RSS {} GiB over {} frame(s)".format(
                writer_cm.encoder_peak_rss_gb, writer_cm.frames_written), flush=True)
        # **The other half of the derived count.** `stream()` refuses a source shorter than the
        # plan; this refuses one longer. A container whose duration and rate imply fewer frames
        # than it holds would otherwise deliver a silently truncated retime.
        # `grab()` rather than `read()`: this counts, it does not look. Decoding the remainder
        # of a long file to produce a number we immediately refuse on is work nobody asked for.
        surplus = 0
        while capture.grab():
            surplus += 1
        # **Two frames of slack, not zero, and the docstring above says why it is needed.** A
        # count derived from a duration stored to the millisecond drifts by a frame over a long
        # clip, and an edit-list trim is exactly the case where a container's numbers and a
        # decode disagree by a little. §2's bound is +-2 output frames; the same tolerance in
        # source frames is the smallest one that does not refuse arithmetic noise. Beyond it the
        # disagreement is structural rather than rounding, and a retime would be truncated.
        if surplus > SURPLUS_TOLERANCE_FRAMES:
            raise WorkerError(
                INVALID_SOURCE,
                "the source holds {} frame(s) beyond the {} its duration and rate imply, which "
                "is past the {}-frame tolerance for rounding, so the retime would have been "
                "truncated. The container's own numbers disagree with its content."
                .format(surplus, n_in, SURPLUS_TOLERANCE_FRAMES))
        # **Always present, None when nothing measured it.** Setting the key only on success
        # makes "no GPU here" and "nobody thought to ask" reach a ledger row identically — as an
        # absent key and a `KeyError` — which is the distinction `build_identity`'s docstring
        # already argues for every field it reports.
        return dict(stats, scale=scale, peak_vram_gb=_read_peak(peak_reset),
                    # ffmpeg's own high-water mark, beside the GPU's. The 8K ceiling was in the
                    # encoder rather than the model, and neither number alone would have said so.
                    encoder_peak_rss_gb=writer.encoder_peak_rss_gb,
                    x264_params=FRUGAL_X264)
    finally:
        if capture is not None:
            capture.release()


def _reset_peak():
    """Zero CUDA's high-water mark, returning whether it could be. `False` on CPU or no torch."""
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return False
        torch.cuda.reset_peak_memory_stats()
        return True if torch.cuda.memory_allocated() >= 0 else False
    except Exception:  # noqa: BLE001 — a measurement must never cost a delivered master
        return False


def _read_peak(was_reset):
    """The job's peak allocation in GiB, or **None where nothing measured it**.

    None rather than zero, and rather than a figure from a telemetry panel: a panel reading is
    not something a ledger row can cite, and a fabricated number is indistinguishable from a
    measurement — which is the rule four other places in this release already follow.
    """
    if not was_reset:
        return None
    try:
        import torch  # noqa: PLC0415
        return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
    except Exception:  # noqa: BLE001
        return None


def _tensors(frames):
    """cv2 frames to tensors, lazily, one at a time — never a list, whatever the clip length."""
    import torch  # noqa: PLC0415 — the vendored CLI has already imported it by the time we run

    for frame in frames:
        yield _to_tensor(frame, torch)
