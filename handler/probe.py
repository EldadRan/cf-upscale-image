"""Sniffing what arrived, and measuring what was written.

Two jobs that both come down to *read the bytes rather than believe a label*:

**Sniffing.** The vendored SeedVR2 CLI decides video-versus-image from the file's **extension**
(`get_input_type`, `docs/decisions.md` 0.8), and an unrecognised extension is not an error there —
it logs a warning and returns zero frames. Meanwhile a presigned GET's path is a CF-minted key,
which is opaque and carries no extension by design (`CF_storage`, Object Keys). So the extension
has to come from the content. Naming the download `source.input`, as the media worker does, would
silently produce a zero-frame job here.

**Measuring.** CF's output fields are measured from the file this worker wrote, never asserted
from what the encoder was asked for. There is no second reader downstream: these values are where
CF's own become.
"""

import json
import os
import subprocess

from errors import INVALID_SOURCE, INTERNAL, WorkerError

FFPROBE_TIMEOUT_S = 120

# Magic-number → extension. The extension is what the vendored CLI dispatches on, so this table
# only has to be right about the *kind* of file; what actually decodes is OpenCV's build, which
# is a separate question and a measured one.
#
# Deliberately narrow. An extension guessed wrong sends a video down the image path, and the
# failure is a one-frame output that looks like a successful job — so an unrecognised container
# is refused loudly rather than guessed at.
def _sniff(head):
    if head[:4] == b"\x89PNG":
        return ".png"
    if head[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if head[:2] == b"BM":
        return ".bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    if head[:4] == b"RIFF":
        # RIFF is a wrapper: the form type at byte 8 says which.
        if head[8:12] == b"WEBP":
            return ".webp"
        if head[8:11] == b"AVI":
            return ".avi"
        return None
    if head[:4] == b"\x1a\x45\xdf\xa3":
        # EBML. WebM and Matroska share the signature and differ by DocType, which sits in the
        # header within the first few dozen bytes.
        return ".webm" if b"webm" in head[:256] else ".mkv"
    if head[4:8] == b"ftyp":
        # ISO base media: MP4, MOV, M4V. The brand distinguishes them and OpenCV does not care,
        # but `.mov` is kept distinct because a QuickTime brand under a `.mp4` name is the sort
        # of thing that makes a later bug report confusing.
        return ".mov" if head[8:12] == b"qt  " else ".mp4"
    if head[:3] == b"FLV":
        return ".flv"
    if head[:4] == b"\x30\x26\xb2\x75":
        return ".wmv"
    return None


def detect_extension(path):
    """The extension the vendored CLI needs, read from the bytes.

    Refuses rather than guesses. `invalid_source` is the never-retryable table: bytes that
    arrived whole and are not a media container are the same bytes forever.
    """
    with open(path, "rb") as handle:
        head = handle.read(4096)
    extension = _sniff(head)
    if extension is None:
        raise WorkerError(
            INVALID_SOURCE,
            "source is not a recognised media container (first bytes: {})".format(
                head[:12].hex()
            ),
        )
    return extension


def named_with_extension(path, extension):
    """Rename the download so the CLI's extension dispatch sees the truth."""
    target = path + extension
    os.replace(path, target)
    return target


def _ffprobe(args, failure_code, what):
    command = ["ffprobe", "-v", "error", "-print_format", "json"] + args
    try:
        completed = subprocess.run(
            command, capture_output=True, timeout=FFPROBE_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        raise WorkerError(failure_code, "ffprobe timed out {}".format(what))
    if completed.returncode != 0:
        raise WorkerError(
            failure_code,
            "ffprobe failed {}: {}".format(what, completed.stderr.decode()[-400:].strip()),
        )
    try:
        return json.loads(completed.stdout or b"{}")
    except ValueError as exc:
        raise WorkerError(failure_code, "ffprobe returned unparseable JSON {}: {}".format(
            what, exc))


def _rate(value):
    """ffprobe reports frame rates as 'num/den'. Returns None rather than 0 for an absent rate,
    because 0 fps and 'no rate reported' are different claims and only one is a measurement."""
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            numerator, denominator = float(numerator), float(denominator)
        except ValueError:
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    try:
        return float(value)
    except ValueError:
        return None


#: Pixel formats that carry an alpha channel. **Enumerated rather than pattern-matched**: the
#: tempting `"a" in name` test calls `yuv420p` alpha-free correctly and `gray` alpha-free by luck,
#: but calls `pal8` alpha-free when a palette can hold transparency, and `ya8` — grey plus alpha —
#: is a name where the `a` that matters is not the one a pattern would find.
#:
#: `ffprobe -pix_fmts` reports an ALPHA flag per format and would be authoritative, but it is a
#: second subprocess on every job to answer a question with this few real answers.
_ALPHA_PIX_FMTS = frozenset((
    "rgba", "bgra", "argb", "abgr",
    "rgba64be", "rgba64le", "bgra64be", "bgra64le",
    "ya8", "ya16be", "ya16le",
    "yuva420p", "yuva422p", "yuva444p",
    "yuva420p9be", "yuva420p9le", "yuva422p9be", "yuva422p9le",
    "yuva444p9be", "yuva444p9le",
    "yuva420p10be", "yuva420p10le", "yuva422p10be", "yuva422p10le",
    "yuva444p10be", "yuva444p10le",
    "yuva420p16be", "yuva420p16le", "yuva422p16be", "yuva422p16le",
    "yuva444p16be", "yuva444p16le",
    "gbrap", "gbrap10be", "gbrap10le", "gbrap12be", "gbrap12le", "gbrap16be", "gbrap16le",
    "pal8",
))


#: Pixel formats carrying more than 8 bits per component. Enumerated for the same reason
#: `_ALPHA_PIX_FMTS` is: `"10" in name` also matches `yuv410p`, which is 8-bit.
_HIGH_DEPTH_PIX_FMTS = frozenset(
    name for name in (
        "yuv420p10le", "yuv420p10be", "yuv422p10le", "yuv422p10be",
        "yuv444p10le", "yuv444p10be", "yuv420p12le", "yuv420p12be",
        "yuv422p12le", "yuv422p12be", "yuv444p12le", "yuv444p12be",
        "yuv420p16le", "yuv420p16be", "yuv422p16le", "yuv422p16be",
        "yuv444p16le", "yuv444p16be",
        "yuva420p10le", "yuva420p10be", "yuva422p10le", "yuva422p10be",
        "yuva444p10le", "yuva444p10be",
        "gbrp10le", "gbrp10be", "gbrp12le", "gbrp12be", "gbrp16le", "gbrp16be",
        "gbrap10le", "gbrap10be", "gbrap12le", "gbrap12be", "gbrap16le", "gbrap16be",
        "p010le", "p010be", "p016le", "p016be",
        "rgb48le", "rgb48be", "rgba64le", "rgba64be",
        "bgr48le", "bgr48be", "bgra64le", "bgra64be",
        "xyz12le", "xyz12be",
    ))


def bits_per_component(pix_fmt):
    """More than 8 bits per component? **The worker reads video through `cv2.VideoCapture`,
    which always decodes to 8 bits**, so a 10-bit source is truncated before the model sees it —
    silently, with every structural check passing. Naming it is the least this can do until the
    reader is replaced.
    """
    return 16 if pix_fmt in _HIGH_DEPTH_PIX_FMTS else 8


def _pix_fmt_has_alpha(pix_fmt):
    return bool(pix_fmt) and pix_fmt in _ALPHA_PIX_FMTS


#: Extensions `_sniff` returns for sources with no time axis. **Decided from the sniffed bytes
#: rather than from a frame count**, because a frame count is the number `docs/decisions.md` 0.2
#: says not to act on — a container's claim and a decode disagree on exactly the files CF sends.
#: A one-frame MP4 is still a video here: it has a rate, an audio track is possible, and CF asked
#: for a video. What makes a job a still is that the *source* is an image.
STILL_EXTENSIONS = frozenset((".png", ".jpg", ".webp", ".bmp", ".tif"))


def is_still(extension):
    return extension in STILL_EXTENSIONS


def probe_source(path):
    """Dimensions, rate and audio presence of the source. **Never a frame count.**

    A container's frame count is exactly the number `docs/decisions.md` 0.2 says not to act on:
    CF's own `media_trim` can bound a video with an MP4 edit list, leaving frames in the stream
    that the container still counts, so a probe and a decode disagree on precisely the files CF
    sends. Every frame count this worker reports comes from its own decode loop.
    """
    data = _ffprobe(["-show_streams", "-show_format", path], INVALID_SOURCE, "on the source")
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise WorkerError(INVALID_SOURCE, "source has no video stream")

    width, height = video.get("width"), video.get("height")
    if not width or not height:
        raise WorkerError(INVALID_SOURCE, "source video stream reports no dimensions")

    duration = data.get("format", {}).get("duration")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {
        "width": int(width),
        "height": int(height),
        # **Two rates, because one field was being asked two questions.** `fps` is the DECLARED
        # cadence — `r_frame_rate`, the container's own base rate, exact as `30000/1001`. It is
        # what the contract's tables are stated against: §2b's nominal-fps premise, the
        # real-frame shares, the snap-tolerance ladder. Planning a 29.97 source at 29.869 moved
        # `t=0.05`'s predicted real share from 13.3% to 9.1%.
        #
        # `measured_fps` is `avg_frame_rate`, which is **frames divided by duration** — a
        # measurement, not a cadence. Multiplying it back by the duration returns the frame count
        # by construction, which is why `estimated_frames` reads it and must keep reading it: on
        # a 30 fps clip spliced with a 15 fps one they are 30.000 and 27.632, and the declared
        # rate over-estimates by three frames on 35. An over-estimate refuses work that would
        # have succeeded.
        #
        # **They differ on anything spliced or variable and agree on everything else**, which is
        # why one field survived this long and why the fixtures cannot catch it.
        "fps": _rate(video.get("r_frame_rate")) or _rate(video.get("avg_frame_rate")),
        "measured_fps": _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate")),
        "duration_s": float(duration) if duration else None,
        # The *video stream's* own duration, not the container's. The container's is the longest
        # stream, so on a source whose audio outruns its picture it reports the audio -- exactly
        # the case a bound on the carried track has to be measured against.
        "video_duration_s": (float(video["duration"]) if video.get("duration") else None),
        "codec": video.get("codec_name"),
        "has_audio": audio is not None,
        # **Whether the source carries an alpha channel**, which decides whether this job is a
        # 4-channel job end to end: the read, the model's alpha path, the writer's pixel format,
        # and what container the master can be (yuv420p cannot hold alpha at all).
        #
        # Read from the pixel format rather than the codec: PNG, WebP and TIFF all encode both
        # with and without alpha, so the codec name settles nothing.
        "has_alpha": _pix_fmt_has_alpha(video.get("pix_fmt")),
        "pix_fmt": video.get("pix_fmt"),
        # Which codec, so the mux can stream-copy where MP4 takes it and re-encode only where it
        # will not — the media worker's rule, and what keeps a carried track bit-exact when it can be.
        "audio_codec": audio.get("codec_name") if audio else None,
        "container": (data.get("format", {}).get("format_name") or "").split(",")[0] or None,
    }


def written_frame_count(path):
    """Frames in a file **this worker encoded**, read from the container. None if it does not say.

    `probe_source` refuses to read a frame count at all, and its reason is good: CF's `media_trim`
    can bound a video with an MP4 edit list, leaving frames in the stream that the container still
    counts, so a probe and a decode disagree on precisely the files CF sends. That argument is
    about arbitrary *input*. It does not carry to output this worker wrote itself moments earlier
    -- constant rate, one video stream, no edit list, an encoder this module invoked.

    It was extended to the output anyway, and the assumption underneath it was false. `probe_output`
    still says the output's frame count "is what the write loop wrote, which is a decode figure by
    construction". On 2026-08-15 the write loop put 48 frames into the pipe and the muxer wrote 46:
    `-shortest` let a source audio track 16 ms shorter than the video end the video. The master
    played correctly and was two frames short, `frames_match` passed, and nothing in this worker
    could have caught it -- both frame counts it held were upstream of ffmpeg, and agreed with each
    other.

    So this is the one number that is worth reading from a container, because it is the only one
    measured on the far side of the encode.

    **None is not a pass.** A container that does not report the count leaves the verification
    undone, and the caller must record that rather than read silence as agreement.
    """
    data = _ffprobe(["-show_streams", "-select_streams", "v:0", path], INTERNAL,
                    "counting frames in the written output")
    streams = data.get("streams") or []
    if not streams:
        raise WorkerError(INTERNAL, "the written output has no video stream")
    value = streams[0].get("nb_frames")
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def probe_output(path):
    """The fields CF takes as its own, measured from the file this worker wrote.

    `width`, `height`, `duration`, `fps`, `has_audio`, `codec` — and no frame count, for the same
    reason as above. The output's frame count is what the write loop wrote, which is a decode
    figure by construction and is carried separately.
    """
    data = _ffprobe(["-show_streams", "-show_format", path], INTERNAL, "on the output")
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        # The worker wrote this file, so a missing video stream is a worker fault rather than a
        # bad request — `internal`, not `invalid_source`.
        raise WorkerError(INTERNAL, "the written output has no video stream")

    duration = data.get("format", {}).get("duration")
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "duration_s": float(duration) if duration else None,
        # Split the same way as `probe_source`, and split at the same time: one call site changed
        # and the other left is the shape this project keeps finding. Here the declared rate also
        # reads back what this worker's own encoder was told to write, where an average over a
        # short clip is the one number guaranteed to disagree with it.
        "fps": _rate(video.get("r_frame_rate")) or _rate(video.get("avg_frame_rate")),
        "measured_fps": _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate")),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "codec": video.get("codec_name"),
    }


def is_faststart(path):
    """Whether the moov atom precedes mdat. Not a parameter and not conditional here — this
    worker writes its own container and always places it at the front — so this exists to
    *verify* that rather than to decide it. A claim nothing checks is a claim."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(2 * 1024 * 1024)
    except OSError:
        return None
    moov, mdat = head.find(b"moov"), head.find(b"mdat")
    if moov == -1:
        return None if mdat == -1 else False
    return mdat == -1 or moov < mdat
