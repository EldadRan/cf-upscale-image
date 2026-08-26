"""Writing the master, once, with everything CF requires in the same mux.

**This module exists because the vendored writer cannot satisfy the contract** (see
`docs/decisions.md` 0.3). SeedVR2's `save_frames_to_video` offers `mp4v` through OpenCV or
`libx264 -crf 12` through an ffmpeg pipe, and neither writes faststart, `-metadata` tags or
audio. CF requires all three **in the mux already being done** and explicitly forbids a second
pass — a remux that exists only to add tags is a rewrite, and CF has measured a stream-copy
faststart pass take a 342-frame trim to 343.

So the worker owns the encode. Frames arrive as they are produced and go straight down a pipe;
nothing is staged on disk and no finished file is ever reopened to fix what the first pass should
have set.

**On `+faststart` being one pass and not two.** ffmpeg relocates the moov atom after writing the
last frame, inside the same invocation. That is not a remux of a finished file: no packet is
re-encoded, no container is reinterpreted, and no edit list exists to be moved — this worker's
output has none, because it writes a fresh timeline rather than bounding an existing one. The
distinction matters because the failure CF measured comes from *reinterpreting* a container that
carried an edit list, which cannot arise here.
"""

import os
import subprocess
import time

import probe

from errors import INTERNAL, WorkerError

#: Audio codecs that mux into MP4 as-is. Anything else is re-encoded to AAC — the media worker's
#: rule, and the reason a copied track stays bit-exact where it can.
MP4_NATIVE_AUDIO = ("aac", "mp3", "alac", "ac3", "eac3")

#: The master's encoder settings. Not measured yet — the encode figures CF was owed were never
#: would justify them, and until those exist these are a starting point rather than a decision.
#: CRF 12 is the vendored writer's own choice, kept so the first measurements compare against
#: something rather than against a number invented here.
DEFAULT_CRF = 12
DEFAULT_PRESET = "medium"
DEFAULT_CODEC = "h264"

#: **What each codec name becomes on the command line.** `envelope.py` owns which names are legal
#: and resolves `"source"` against the probed source before anything reaches here — this map is
#: deliberately total over what it accepts, so an unresolved name raises a KeyError in the worker
#: rather than encoding something nobody asked for.
CODEC_LIBRARIES = {"h264": "libx264", "h265": "libx265"}


def _peak_rss_gb(pid):
    """The largest resident set this process reached, in GiB, or None where it cannot be read.

    **`/proc/<pid>/status`'s `VmHWM`, sampled while the process is alive.** `getrusage`'s
    `RUSAGE_CHILDREN` would be the easy answer and is the wrong one: it reports the maximum across
    every child this worker has ever reaped, so an ffprobe from a previous phase and the encode
    would be indistinguishable — a plausible number about a different process, which is the class
    this project keeps finding. `VmHWM` is that process's own high-water mark and it disappears
    when the process does, which is why it is sampled rather than read at the end.

    **This is a DIFFERENT PROCESS from `phasewatch.host_rss_gb`, not a different sampling
    strategy.** That one reads `/proc/self` — the worker, model included — and so can never answer
    "what did the encoder cost". This one reads the ffmpeg child.

    None on anything that is not Linux, which is honest: a figure that is absent says nothing and
    a figure that is zero says the encode used no memory.

    **RESTORED 2026-08-26 with the drain closed.** The original stopped sampling when the last
    `write()` returned and reassured that the shortfall was near zero *because x264's flush is
    memory-non-increasing*. That argument was written when x264 was the only encoder, and this
    instrument now decides an x265 question — where the lookahead and reference structure are not
    x264's and the flush is exactly where a heavier encoder could still be climbing. It would have
    under-read in the direction that makes h265 look affordable. `MasterWriter.__exit__` now keeps
    sampling across the drain, so the peak covers the whole encode rather than the fed part of it.

    **It still under-reads totally on a clip shorter than the encoder's buffering window**, where
    the entire encode happens after the final write and the drain poll is the only thing that sees
    anything. **A reassuringly low peak from a small fixture means nothing** — which matters
    because a small fixture is what somebody reaches for when checking that the measurement works.
    """
    try:
        with open("/proc/{}/status".format(pid), encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 2)
    except Exception:  # noqa: BLE001 — a measurement must never cost a delivered master
        return None
    return None


def _identity_tags(identity):
    """`-metadata` arguments. **Identity only** — this file is delivered.

    Timings, hardware, tiling configuration, worker ids and anything resembling a credential stay
    in the manifest and the diagnostics bundle. What goes in the container is what the file needs
    to say what it is when it is found in R2 with no job and no manifest beside it.

    It is a recovery aid and never a source of truth: CF's standing rule is to read the worker's
    reported fields rather than re-probe the file, and these tags are what someone falls back to
    when the response and the manifest are both gone.
    """
    args = []
    for key, value in identity.items():
        if value is None:
            continue
        args += ["-metadata", "{}={}".format(key, value)]
    return args


def still_master_extension(width, height):
    """Always `.png`.

    **PNG rather than lossless WebP, chosen for the people who have to look at the output.** Both
    are lossless and WebP is materially smaller at these dimensions, which is what this returned
    before. But a master is the thing a person opens to check the work, hands to a customer, or
    drags into an editor, and PNG opens everywhere without a thought while WebP still meets tools
    that will not preview it. Paying storage for that is the right trade: the file is written once
    and looked at many times, and an artefact nobody can conveniently open is an artefact nobody
    checks.

    It also removes a ceiling. WebP is limited to 16383 pixels per side by the format itself,
    which sits inside the range this worker is aimed at — 12K fits, 16K does not — so the old
    two-format rule had a real edge in it. PNG has no practical limit, so there is one format, one
    path, and no dimension at which the master silently changes type.

    Lossless WebP is still used for the `crop` derives, where CF asked for it by name and the
    files are small evidence images rather than the deliverable.

    The arguments are kept so the signature does not change if a size-dependent rule ever comes
    back.
    """
    del width, height
    return ".png"


class StillWriter:
    """The master for a single-frame job: **lossless, and alpha-capable.**

    Separate from `MasterWriter` rather than a mode of it, because almost nothing carries over.
    There is no rate, no audio, no faststart, no CRF, and — the reason this class exists —
    **`yuv420p` cannot hold an alpha channel at all**, so an RGBA job cannot produce an MP4 master
    whatever else is arranged. A still master that went through H.264 would also be lossy twice
    over by the time a WebP derive was taken from it.

    **Written with PIL rather than ffmpeg, and that is what keeps the identity tags.** ffmpeg's
    `-metadata` is silently dropped by both the WebP and PNG muxers — measured, exit code 0, no
    warning, exactly as the MP4 muxer dropped them until `+use_metadata_tags` was found
    (`docs/decisions.md` 3.3). PIL writes PNG `tEXt` and WebP XMP, both of which round-trip. So
    an orphaned still in R2 still says what it is, which is the whole point of the mechanism.
    """

    def __init__(self, path, width, height, identity, channels=3):
        if channels not in (3, 4):
            raise WorkerError(INTERNAL, "a still master needs 3 or 4 channels, got {}".format(
                channels))
        self.path = path
        self.width = width
        self.height = height
        self.channels = channels
        self.identity = dict(identity or {})
        self.frames_written = 0
        self._buffer = bytearray()

    def __enter__(self):
        return self

    def set_frame_size(self, width, height):
        """Adopt the size the model actually produced, before the first frame.

        The constructor is handed `estimator.output_dimensions`, which is what the *caller* asked
        for. The model rounds to its own grid and the two disagree at large scales. Since a still
        is written from a flat buffer, the declared size is what reshapes it — so believing the
        prediction over the frame would either refuse a good job or, worse, reshape it wrongly.
        """
        if self.frames_written:
            raise WorkerError(INTERNAL, "the still master's size was changed after it was written")
        self.width, self.height = int(width), int(height)
        self.identity["cf_output"] = "{}x{}".format(self.width, self.height)

    def write(self, frame_bytes):
        """One frame, `width × height × channels` bytes.

        **Refuses a second frame rather than overwriting or appending.** A still master reached
        with more than one frame means the caller decided "still" on something that is not one,
        and either outcome of guessing is worse than saying so.
        """
        if self.frames_written:
            raise WorkerError(INTERNAL, "a still master was given more than one frame")
        expected = self.width * self.height * self.channels
        if len(frame_bytes) != expected:
            raise WorkerError(INTERNAL, "still frame is {} bytes, expected {}".format(
                len(frame_bytes), expected))
        self._buffer += frame_bytes
        self.frames_written = 1

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None or not self.frames_written:
            return False
        self._save()
        return False

    def _save(self):
        import numpy as np
        from PIL import Image

        array = np.frombuffer(bytes(self._buffer), dtype=np.uint8).reshape(
            self.height, self.width, self.channels)
        mode = "RGBA" if self.channels == 4 else "RGB"
        image = Image.fromarray(array, mode=mode)
        if self.path.endswith(".png"):
            from PIL.PngImagePlugin import PngInfo
            info = PngInfo()
            for key, value in self.identity.items():
                info.add_text(str(key), str(value))
            image.save(self.path, format="PNG", pnginfo=info, optimize=False)
        else:
            image.save(self.path, format="WEBP", lossless=True, quality=100, method=4,
                       xmp=_identity_xmp(self.identity))
        if not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
            raise WorkerError(INTERNAL, "the still master was not written")


def _identity_xmp(identity):
    """The identity fields as XMP, which is the only metadata WebP will carry through PIL."""
    fields = "".join(
        "<cf:{0}>{1}</cf:{0}>".format(key, _xml_escape(str(value)))
        for key, value in sorted(identity.items())
    )
    return (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:cf="https://cf.invalid/ns/upscale/1.0/">'
        "<rdf:Description>{}</rdf:Description>"
        "</rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
    ).format(fields).encode("utf-8")


def _xml_escape(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;"))


class MasterWriter:
    """A one-pass ffmpeg encode fed frame by frame.

    Used as a context manager so the pipe is closed and the process reaped on any path,
    including an OOM raised mid-generation by the model upstream of it.
    """

    def __init__(self, path, width, height, fps, identity,
                 audio_source=None, audio_codec=None, audio_limit_s=None,
                 crf=DEFAULT_CRF, preset=DEFAULT_PRESET, codec=DEFAULT_CODEC,
                 head_keyframes=False):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_written = 0
        #: **ffmpeg's own high-water mark, not this worker's.** The 8K run died at ~46 GiB inside
        #: x264's working set while this side held one frame and a cached pair — and nothing
        #: reported it, so the ceiling had to be inferred from a kernel kill. A path built to find
        #: a memory ceiling that does not report memory has to be run twice to learn anything, and
        #: each run is fifty minutes of A40. None where it cannot be measured.
        self.encoder_peak_rss_gb = None
        self._proc = None
        self._identity = dict(identity or {})
        self._audio_source = audio_source
        self._audio_codec = audio_codec
        #: Seconds of audio to read, or None to read it all. Bounds the carried track to the
        #: picture without the muxer being allowed to bound the picture to the track.
        self._audio_limit_s = audio_limit_s
        #: Frames the container reports once ffmpeg has exited, or None where it does not say.
        #: The only frame count this class holds that was measured after the encode.
        self.verified_frames = None
        self._crf = crf
        self._preset = preset
        self._codec = codec
        #: **Off unless the request asked.** There is no module-level default that can turn this
        #: on and no path to it from anywhere else in the worker: it is a parameter, threaded from
        #: the derived config, and the flag below is emitted only inside `if self._head_keyframes`.
        #: `codec_default_unmoved` therefore holds by CONSTRUCTION rather than by a value — the
        #: gate this wave had to pass is that a request naming none of the knobs still produces
        #: byte-for-byte what production produces today.
        self._head_keyframes = head_keyframes

    def set_frame_size(self, width, height):
        """Adopt the size the model actually produced, before ffmpeg is started.

        **This is why ffmpeg is started lazily.** `-s` on a rawvideo input is a promise about
        bytes that carry no shape of their own: declare 8210×4320 and feed 8208×4320, and ffmpeg
        does not complain — it reads across frame boundaries and writes a master that shears
        progressively, exiting 0 with a plausible file size. The still path caught the same
        disagreement as a byte-count refusal; this path would not have caught it at all.
        """
        if self._proc is not None:
            raise WorkerError(INTERNAL, "the master's frame size was changed after ffmpeg started")
        self.width, self.height = int(width), int(height)
        self._identity["cf_output"] = "{}x{}".format(self.width, self.height)

    def _build_command(self):
        width, height, fps = self.width, self.height, self.fps
        identity = self._identity
        audio_source, audio_codec = self._audio_source, self._audio_codec
        crf, preset = self._crf, self._preset
        path = self.path

        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            # Input 0: raw frames on stdin, exactly as the model produces them.
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "{}x{}".format(width, height), "-r", str(fps), "-i", "-",
        ]

        carry_audio = audio_source is not None
        if carry_audio:
            # **The trim goes on the audio input, never on the output.** `-t` before `-i` bounds
            # how much of *that* input is read and can only ever shorten the audio.
            if self._audio_limit_s:
                command += ["-t", "{:.6f}".format(float(self._audio_limit_s))]
            command += ["-i", audio_source]

        # **The codec the request asked for.** This was the literal `"libx264"` until 2026-08-26;
        # `envelope.py` validates the name and resolves `"source"` against the probed source, so
        # what arrives here is always one this map holds.
        command += ["-map", "0:v:0", "-c:v", CODEC_LIBRARIES[self._codec],
                    "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]

        # **Five keyframes at the head, and nothing else changes.** `expr:lt(n,5)` makes frames
        # 0-4 I-frames; the encoder's own scene-cut keyframes survive beside them. Without it a
        # master carries a single keyframe at 0 — no GOP flag is set anywhere in this file, so
        # x264 and x265 run their 250-frame default and model output rarely trips a scene cut —
        # and trimming one frame off the FRONT downstream costs a full re-encode. Trimming the
        # END was never affected and needs nothing.
        #
        # **The price is per ADDED I-frame, not per rule**, which is why this is off by default:
        # +0.52% on a clip that already had six keyframes, +11.53% on one that had a single
        # keyframe like ours. A twenty-fold spread from the same flag.
        if self._head_keyframes:
            command += ["-force_key_frames", "expr:lt(n,5)"]

        if carry_audio:
            # `?` makes the mapping optional, so a source whose audio stream vanished between the
            # probe and the mux does not fail the encode of an expensive master.
            command += ["-map", "1:a:0?"]
            command += ["-c:a", "copy"] if audio_codec in MP4_NATIVE_AUDIO else \
                       ["-c:a", "aac", "-b:a", "192k"]
            # **No `-shortest` here, and the reason is a delivered defect.** The flag was added
            # to stop a longer source track leaving an audio-only tail past the last frame, under
            # the comment "the video stream is authoritative". It is symmetric and does the
            # opposite: it ends the output when *any* input ends, so an audio track shorter than
            # the video truncates the video. On 2026-08-15 a 1.984 s AAC track against 2.000 s of
            # picture cost two frames of a delivered master; reproduced locally at 45 of 48 frames
            # with the flag and 48 of 48 without, from the same source and the same mux.
            #
            # AAC frames are 1024 samples, so a track almost never lands exactly on the video's
            # duration and is usually a fraction short. Every audio job was exposed. It had never
            # shown because every fixture that had been run at size was silent.
            #
            # The tail it was defending against is handled by `_audio_limit_s` above, which cannot
            # touch the picture. Where no limit is known the tail is accepted: audio playing past
            # the last frame is a cosmetic fault, and a master missing frames is not.

        command += _identity_tags(identity)
        # Two flags, and the second is not optional despite looking like a detail.
        #
        # `+faststart` puts the moov atom at the front, in this pass. Never a later one.
        #
        # `+use_metadata_tags` is what makes the identity tags above actually exist. **The MP4
        # muxer silently discards any metadata key it does not recognise** — `comment` and
        # `title` survive, `cf_request_id` does not — with a zero exit code and no warning.
        # Measured 2026-08-12 (`docs/decisions.md` 3.3). Without it the whole "a file found in
        # R2 with no job and no manifest still says what it is" mechanism is absent from every
        # file while every check around it passes.
        command += ["-movflags", "+faststart+use_metadata_tags", path]

        return command

    def __enter__(self):
        return self

    def _start(self):
        self.command = self._build_command()
        try:
            self._proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise WorkerError(INTERNAL, "could not start ffmpeg: {}".format(exc))

    def write(self, frame_bytes):
        """One frame, already `rgb24` and `width × height × 3` bytes."""
        # **The check the still path had and this one did not.** rawvideo carries no shape, so a
        # frame of the wrong length is not an error to ffmpeg — it is the first bytes of the next
        # frame, and the master shears from that point on while the process exits 0. Cheap to
        # check, and it turns the worst failure mode in this file into a refusal.
        expected = self.width * self.height * 3
        if len(frame_bytes) != expected:
            raise WorkerError(INTERNAL, "frame {} is {} bytes, expected {} for {}x{}".format(
                self.frames_written, len(frame_bytes), expected, self.width, self.height))
        if self._proc is None and not self.frames_written:
            self._start()
        if self._proc is None or self._proc.poll() is not None:
            raise WorkerError(INTERNAL, self._died("ffmpeg exited before the frames did"))
        try:
            self._proc.stdin.write(frame_bytes)
            self._sample_peak()
        except BrokenPipeError:
            raise WorkerError(INTERNAL, self._died("ffmpeg closed the pipe"))
        self.frames_written += 1

    def _sample_peak(self):
        """One `/proc` read against the encoder, keeping the maximum.

        **Called where the loop already blocks.** `stdin.write` returns when the pipe accepts the
        frame, which is exactly when the encoder is working — so the samples land across the whole
        encode without a thread, and one `/proc` read per frame is noise against a 50 MiB write.
        """
        if self._proc is None:
            return
        peak = _peak_rss_gb(self._proc.pid)
        if peak is not None and peak > (self.encoder_peak_rss_gb or 0.0):
            self.encoder_peak_rss_gb = peak

    #: How often the drain is sampled, in seconds. **Small enough to catch a rising flush, large
    #: enough that the poll costs nothing** — one `/proc` read per interval against an encode
    #: measured in minutes at 8K.
    DRAIN_SAMPLE_S = 0.2

    def _drain(self):
        """Wait for ffmpeg to finish, sampling its memory the whole way down.

        **This is the half the original instrument did not have, and it decides an x265 question.**
        The last `write()` returns when the last frame is *accepted*, not when it is encoded: the
        encoder then drains its lookahead and flushes, and nothing sampled that phase. The original
        docstring reassured that the shortfall was near zero because x264's flush is
        memory-non-increasing — an argument about the codec it was written for, and this wave
        exists to measure the other one, whose lookahead and reference structure are not x264's.
        **It would have under-read exactly where a heavier encoder is heaviest, in the direction
        that makes h265 look affordable.**

        `VmHWM` is monotone for the child's life and disappears with the process, so the sampling
        has to happen *before* the reap — which is why this polls rather than reading once after
        `wait()`.

        **SAMPLE FIRST, THEN POLL, AND NOTHING AFTER THE LOOP.** `Popen.poll()` performs the
        `waitpid(WNOHANG)` itself, so the instant it observes the exit the child is REAPED and
        `/proc/<pid>` is gone. A sample taken after the loop could therefore never return a
        reading — it would take `_peak_rss_gb`'s `except` path on every run — and worse, if the
        kernel had recycled the pid it would adopt an unrelated process's `VmHWM`. That is
        precisely the "plausible number about a different process" this file rejects `getrusage`
        for, arrived at by another route. An earlier version of this method had that trailing
        sample and a docstring explaining why it was needed; it was unreachable in the good case
        and wrong in the bad one.

        **What that costs, stated rather than reassured away:** the last interval before exit is
        unsampled, bounded by `DRAIN_SAMPLE_S`. It cannot be closed from this side — the process
        must be alive to be read, and the only event that says it has finished is the one that
        reaps it. Ordering the loop this way makes the gap at most one interval instead of
        leaving it unbounded.
        """
        if self._proc is None:
            return
        while True:
            self._sample_peak()
            if self._proc.poll() is not None:
                return
            time.sleep(self.DRAIN_SAMPLE_S)

    def _died(self, why):
        stderr = b""
        if self._proc is not None:
            try:
                stderr = self._proc.stderr.read() or b""
            except Exception:  # noqa: BLE001 — we are already reporting a failure
                pass
        detail = stderr.decode(errors="replace")[-400:].strip()
        return "{}{}".format(why, ": " + detail if detail else "")

    def __exit__(self, exc_type, exc, traceback):
        if self._proc is None:
            return False
        try:
            self._proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        # **Sampled across the drain, then reaped.** `_drain` returns only once the process has
        # exited, so the `wait()` below is what collects the status rather than what waits.
        self._drain()
        self._proc.wait()
        # An exception on the way in owns the failure; do not replace it with one about ffmpeg,
        # which most likely died *because* of it. The original diagnosis is the useful one —
        # especially for an OOM, where the exception carries the phase and the allocation that
        # failed and no log gives better.
        if exc_type is not None:
            return False
        if self._proc.returncode != 0:
            raise WorkerError(INTERNAL, self._died(
                "ffmpeg exited {}".format(self._proc.returncode)))
        if not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
            raise WorkerError(INTERNAL, "ffmpeg wrote no output to {}".format(self.path))

        # **The first count this worker takes on the far side of the encode**, and the reason it
        # exists is that every other one is on the near side. `decoded_in` and `written_out` are
        # both counters in this process, so `frames_match` compares the write loop to itself and
        # passes on a master the muxer silently truncated. It did exactly that on 2026-08-15.
        #
        # Refusing is the right end for it. A short master is not a degraded success — it is a
        # video that plays correctly and is missing frames, which is the one failure a caller
        # cannot detect downstream either. `internal` is honest: the request was fine and this
        # worker wrote the wrong file.
        self.verified_frames = probe.written_frame_count(self.path)
        if self.verified_frames is not None and self.verified_frames != self.frames_written:
            raise WorkerError(INTERNAL, (
                "the master was written with {} frames but the file holds {} — the encode lost "
                "{} frame(s) after the write loop. The file plays; it is short.").format(
                    self.frames_written, self.verified_frames,
                    self.frames_written - self.verified_frames))
        return False


def encode_proxy(source_path, destination, max_duration_s=None):
    """A delivery rendition of the **output**, to the platform's semantics.

    Inter-coded, 8-bit, 1280 on the long edge, H.264 in MP4 with AAC, audio carried through,
    frame rate inherited. 8-bit even from a 10-bit source, deliberately — it is a delivery
    rendition and 10-bit costs compatibility for a file whose whole purpose is to play anywhere.

    Derived from the master this job wrote, never from the source, and **never upscaled**: the
    scale filter's `min()` clamps it, so a master already under 1280 is copied at its own size
    rather than enlarged.
    """
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", source_path]
    if max_duration_s is not None:
        # A cap from frame zero, not a window — no start offset, so no seek. Past the end it
        # clamps, because *at most* is what was asked for.
        command += ["-t", str(max_duration_s)]
    command += [
        "-vf", "scale='min(1280,iw)':'min(1280,ih)':force_original_aspect_ratio=decrease:"
               "force_divisible_by=2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", destination,
    ]
    _run(command, "encoding the proxy")
    return destination


def extract_poster(source_path, destination, frame_index, fps):
    """One frame of the **output** as WebP, at the output's own resolution — never upscaled.

    Selected by index rather than by time: `round(at_fraction × (frame_count − 1))`, where the
    frame count is the one this job's write loop produced. Seeking by index is what keeps the
    poster on the frame CF asked for rather than on whichever frame a timestamp rounds to.
    """
    # No `-vsync` and no `-fps_mode`, deliberately — **neither is portable across the builds
    # this code has to run on.** `-vsync` was removed in ffmpeg 9.x, and `-fps_mode`, its
    # replacement, did not arrive until 5.0; Ubuntu 22.04 ships 4.4. So the pair that looks like
    # "old flag / new flag" has no version that accepts either. Neither is needed here: with
    # `-frames:v 1` the output is a single image, and the frame-duplication `-vsync 0` guards
    # against cannot arise. Found 2026-08-12 (`docs/decisions.md` 3.3).
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", source_path,
        "-vf", "select=eq(n\\,{})".format(frame_index), "-frames:v", "1",
        "-c:v", "libwebp", "-lossless", "0", "-quality", "88", destination,
    ]
    _run(command, "extracting the poster")
    if not os.path.isfile(destination) or os.path.getsize(destination) == 0:
        raise WorkerError(
            INTERNAL, "poster extraction produced no file at frame {}".format(frame_index)
        )
    return destination


def extract_frame_png(source_path, destination, frame_index):
    """One frame as a lossless PNG, by index. Used to build the `crop` comparison.

    PNG rather than WebP here because it is an intermediate this code reads back into an array —
    a lossy step between the decode and the comparison would put artefacts into the evidence,
    which is the same reason the crop itself is written lossless.
    """
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", source_path,
        "-vf", "select=eq(n\\,{})".format(frame_index), "-frames:v", "1",
        "-c:v", "png", destination,
    ]
    _run(command, "extracting frame {}".format(frame_index))
    if not os.path.isfile(destination) or os.path.getsize(destination) == 0:
        raise WorkerError(INTERNAL, "no frame {} in {}".format(frame_index, source_path))
    return destination


def _run(command, what):
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise WorkerError(INTERNAL, "could not start ffmpeg while {}: {}".format(what, exc))
    if completed.returncode != 0:
        raise WorkerError(INTERNAL, "ffmpeg failed while {}: {}".format(
            what, completed.stderr.decode(errors="replace")[-400:].strip()))
    return completed
