"""The codec request surface — the worker's half of it.

`fable/envelope_oracle.py` in `cf-upscale-project` is this file's specification and
`fable/envelope_cases.py` is the suite both are held to. **The oracle is the authority**: where
this file and it disagree, this file is the bug. `tests/envelope_conformance.py` runs the oracle's
own cases against `derive` below, unmodified, which is what makes agreement evidence rather than
two readings of the same paragraph.

**This file is the code path, not a second opinion.** `validation.py` calls it; nothing else
derives a codec, a crf or a preset. A conformance suite that certified a module production did not
use would be the defect class this project keeps collecting — a check whose exercise avoids the
case it exists for — so the module under test and the module in the request path are the same one
on purpose.

**REBUILT 2026-08-26.** An earlier `envelope.py` existed and left with the RIFE cut on 2026-08-25;
it is not restored from that commit. It was written FROM contract §5c rather than copied from the
oracle, and this is the same exercise against a spec that never moved — minus the interpolation
half, which departed to `cf-rife-project`.

**THE INTERPOLATION FIELD NAMES ARE NOT LISTED HERE, and their absence is the design** (CF,
2026-08-26). A first version listed them and refused each
by name with a pointer to the other repository. **That is a copy of another project's field names
living here, and this file cannot know when they are renamed there** — the duplication rule broken
in the file whose job is to describe this worker. `validation.py`'s `_refuse_unknown` on `params`
already turns a misrouted retime away with no list and no code, because **the separation is what
makes those names unknown.**
"""
from errors import INVALID_FIELD_VALUE, MISSING_REQUIRED_FIELD, WorkerError

import encoder

#: **The worker's constants are `encoder`'s, not copies.** The oracle carries its own `DEFAULT_CRF`
#: and `DEFAULT_PRESET` because it must run with no worker on the path; the conformance link is
#: what keeps the two honest, and a drift between them fails there rather than in a delivery.
CODECS = ("h264", "h265", "source")
DEFAULT_CODEC = "h264"

CRF_MIN, CRF_MAX = 0, 51

#: The keyframe ladder. Each rung is priced — roughly 1-3% of master size per added keyframe, no
#: cliff — and `default` is the shipped behaviour: one keyframe, whatever the encoder chooses.
KEYFRAME_MODES = ("default", "all", "frames", "seconds")
DEFAULT_KEYFRAMES = "default"

#: **A cap on the OUTCOME, not on the request.** A hundred cut points is already most of the way
#: to all-intra on a short clip, and past it a caller wants `all` and its price rather than a list.
MAX_KEYFRAMES = 100

#: The same nine names in x264 and x265, and **not the same quality at the same number** — the
#: scales are offset by several points. That is why h265's shipped default CRF is a ruling rather
#: than something a measurement discovers, and why two codecs never share a table keyed on CRF.
PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast",
           "medium", "slow", "slower", "veryslow")

SIZING_FIELDS = ("target_short_edge_px", "output_size")

def derive(params):
    """`params` in, a normalised codec config out, or `WorkerError`.

    Returns `{codec, crf, preset, release_2_equivalent}`. Absent fields produce the release-2
    answer exactly, which is the property `codec_default_unmoved` exists to hold.
    """
    p = dict(params or {})
    output = p.get("output")
    if output is not None and not isinstance(output, dict):
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'params.output' must be an object like {'codec': 'h265'}. Note that the "
            "top-level 'output' is the R2 destination and a different field entirely.",
        )
    output = dict(output or {})
    has_size = any(p.get(field) is not None for field in SIZING_FIELDS)

    # ---- sizing --------------------------------------------------------------------------
    # Release-2 behaviour, restated only because the codec work must not weaken it.
    #
    # **`MISSING_REQUIRED_FIELD`, and the message is `validation.py`'s WORD FOR WORD.** Running
    # `derive` ahead of `validation`'s own sizing branch made that branch unreachable, so this
    # line inherited its job — and a shorter message under a different code is a CONTRACT-VISIBLE
    # change for any caller branching on `cf_error.code`. The wave that introduced it was about
    # codecs and said nothing about refusal codes, which is exactly how a wire contract moves
    # without anyone deciding to move it.
    if not has_size:
        raise WorkerError(
            MISSING_REQUIRED_FIELD,
            "a request must say what size it wants: either 'target_short_edge_px' (one edge, "
            "aspect preserved) or 'output_size' as {'width': W, 'height': H} (an exact "
            "canvas). Neither was given in 'params'.",
        )

    # ---- codec ---------------------------------------------------------------------------
    codec = output.get("codec", DEFAULT_CODEC)
    if codec not in CODECS:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "output.codec must be one of {}; got {!r}".format(CODECS, codec),
        )

    # ---- crf -----------------------------------------------------------------------------
    # **`isinstance(crf, bool)` is checked, and it is not pedantry.** `True == 1` in Python, so a
    # bool sails through a naive range check and encodes the master at CRF 1 — a request that
    # asked for nothing in particular silently buying the most expensive picture the encoder can
    # make. A string is refused for the same reason in the other direction: `"12"` is a caller
    # who believes they set the value.
    crf = p.get("crf", encoder.DEFAULT_CRF)
    if not isinstance(crf, int) or isinstance(crf, bool):
        raise WorkerError(INVALID_FIELD_VALUE, "field 'crf' must be an integer, got {}".format(
            type(crf).__name__))
    if not CRF_MIN <= crf <= CRF_MAX:
        # **The range is named without a codec and the calibration sentence is gone.** This message
        # used to read "within x264's range" and claim the default was "what every measurement in
        # its calibration was taken at". Once `codec` is request-carried the first half is wrong
        # for half the requests, and the second half cites an x264 calibration that says nothing
        # about x265 — there is no h265 calibration to point at, and asserting one per codec would
        # invent the number the sentence exists to cite.
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'crf' must be within {}-{}, got {}. Lower is better quality and a larger "
            "file; {} is this worker's default. The scale is NOT comparable between codecs — "
            "the same number is a different quality in h264 and h265.".format(
                CRF_MIN, CRF_MAX, crf, encoder.DEFAULT_CRF),
        )

    # ---- preset --------------------------------------------------------------------------
    # **Refused rather than lowercased.** `"Medium"` is a caller who believes they set the preset;
    # silently correcting it is the silent-reinterpretation class this project has paid for twice
    # — an endpoint renamed by a defaulted `--name`, a 16-bit source downconverted without a word.
    preset = p.get("preset", encoder.DEFAULT_PRESET)
    if preset not in PRESETS:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "preset {!r} is not one of {}. A preset is a SPEED label, not a quality one: at a "
            "fixed crf a slower preset buys a smaller file for more time, and it moves encoder "
            "memory where crf barely does.".format(preset, ", ".join(PRESETS)),
        )

    # ---- head_keyframes --------------------------------------------------------------------
    # **A strict boolean: `True` or `False` and nothing else.** `"false"` must REFUSE rather than
    # evaluate true, which is what a truthiness test would do to it — a caller who wrote the word
    # false and got the behaviour they were switching OFF is the silent-reinterpretation class,
    # and this is the field where it costs a re-encoded master. `1` and `0` are refused for the
    # same reason `crf` refuses a bool: the types are not interchangeable just because Python
    # will compare them.
    #
    # **What it does, so the name is never read as what it is FOR.** It makes the first five
    # frames keyframes (`-force_key_frames expr:lt(n,5)` in `encoder.py`), so that trimming a
    # frame or two off the START downstream does not force a whole re-encode. **This worker does
    # not trim anything** — an earlier draft was called `trim_head`, which named the caller's
    # intent, and a master delivered with all its frames under that name would have been answered
    # rather than served.
    head_keyframes = p.get("head_keyframes", False)
    if not isinstance(head_keyframes, bool):
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'head_keyframes' must be true or false, got {!r}. It makes the first frames "
            "keyframes so a downstream head trim does not force a re-encode; it does not trim "
            "anything here.".format(head_keyframes),
        )

    # ---- keyframes -------------------------------------------------------------------------
    # **A NAMED LADDER, not ffmpeg's expression.** The same reason `codec` is an enum rather than
    # a raw `-c:v` string: an expression nobody has seen cannot be certified, priced or supported,
    # and it is injection surface on a worker that runs a subprocess.
    mode = p.get("keyframes", DEFAULT_KEYFRAMES)
    if mode not in KEYFRAME_MODES:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "'keyframes' must be one of {}; got {!r}".format(KEYFRAME_MODES, mode))

    # **A companion field outside its mode is an ORPHAN and is refused.** A request carrying
    # `keyframe_frames` under `keyframes: "seconds"` has two answers in it, and honouring one is
    # the silent-reinterpretation class — the same reason release 2 refused `target_fps` with no
    # `interpolate`.
    frames = p.get("keyframe_frames")
    seconds = p.get("keyframe_seconds")
    if mode != "frames" and frames is not None:
        raise WorkerError(INVALID_FIELD_VALUE,
                          "'keyframe_frames' has no meaning under keyframes: {!r}".format(mode))
    if mode != "seconds" and seconds is not None:
        raise WorkerError(INVALID_FIELD_VALUE,
                          "'keyframe_seconds' has no meaning under keyframes: {!r}".format(mode))

    if mode == "frames":
        if not isinstance(frames, (list, tuple)) or not frames:
            raise WorkerError(INVALID_FIELD_VALUE,
                              "keyframes: 'frames' needs 'keyframe_frames', a non-empty list of "
                              "frame numbers")
        if len(frames) > MAX_KEYFRAMES:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "'keyframe_frames' holds {} entries; the limit is {}. A list this long is "
                "all-intra at all-intra's price — ask for keyframes: 'all' if that is what you "
                "want.".format(len(frames), MAX_KEYFRAMES))
        for entry in frames:
            # `bool` first: `True == 1`, so a bool passes an int check and would name frame 1.
            # Same trap as `crf`, same answer.
            if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0:
                raise WorkerError(
                    INVALID_FIELD_VALUE,
                    "'keyframe_frames' takes whole frame numbers from 0; got {!r}".format(entry))
        # **Duplicates refused rather than collapsed** (CF): a repeat is more likely a mistake in
        # the input than a request, and accepting it delivers a file the caller did not ask for
        # while reporting success. **Order is NOT checked** — a set of cut points has no order, so
        # sorting one is not a reinterpretation, unlike lowering `preset="Medium"`.
        if len(set(frames)) != len(frames):
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "'keyframe_frames' repeats a frame number; every entry names one cut point and a "
                "repeat is more likely a mistake than a request")
        frames = sorted(frames)

    if mode == "seconds":
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "keyframes: 'seconds' needs 'keyframe_seconds', a positive number of seconds "
                "between keyframes")

    # **TWO RULES THIS DOOR CANNOT CHECK, and they are the worker's** — both need the probe or the
    # encode, and both are implemented in `encoder.py` rather than here:
    #   - a frame number beyond the clip's last frame. The true count exists only once the encode
    #     has run: `probe_source` returns none, and `nb_frames` is a header claim this project has
    #     already ruled against trusting (`delivery_witness`: "the header is a claim; the packets
    #     are the file"). Detected at the end of the encode against `frames_written`.
    #   - a `keyframe_seconds` short enough to force more than MAX_KEYFRAMES across the duration.
    #     The cap is on the OUTCOME; this door can only see the list half of it.

    return {
        "codec": codec,
        "crf": crf,
        "preset": preset,
        "head_keyframes": head_keyframes,
        "keyframes": mode,
        "keyframe_frames": frames if mode == "frames" else None,
        "keyframe_seconds": seconds if mode == "seconds" else None,
        # **True only when every knob is where release 2 left it.** One boolean answering "could
        # this request have changed production's output" — so `crf 8`, a better picture, breaks it
        # exactly as `h265` does. The field is about movement, not about quality.
        # **`head_keyframes` counts, and it is the reason the flag is safe.** It changes the
        # master's bytes when set, so a request naming it has moved production's output and must
        # not report otherwise. Default off is what makes `codec_default_unmoved` hold.
        "release_2_equivalent": (codec == DEFAULT_CODEC
                                 and crf == encoder.DEFAULT_CRF
                                 and preset == encoder.DEFAULT_PRESET
                                 and head_keyframes is False
                                 and mode == DEFAULT_KEYFRAMES),
    }


def check_keyframe_cap(config, duration_s):
    """The OUTCOME cap under `seconds`, which the request surface cannot see.

    `derive` counts a list; it cannot count what an INTERVAL produces, because that is a fact
    about the clip rather than the request. One keyframe every 0.1 s is four on a short clip and
    six hundred on a minute — the same request, two different files.

    **Checked right after the probe and before any expensive work**, so a request that cannot be
    served does not pay for a download, a decode and a plan first. Refuses rather than clamps, per
    CF: a caller who asked for an interval this dense has most likely made an input error, and
    quietly delivering a hundred keyframes instead of six hundred is the file they did not ask for
    reported as a success.
    """
    if config.get("keyframes") != "seconds" or not duration_s:
        return
    interval = float(config["keyframe_seconds"])
    # One at t=0 and one at every interval boundary the clip actually reaches.
    count = int(float(duration_s) // interval) + 1
    if count > MAX_KEYFRAMES:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "keyframe_seconds {} over {:.3f}s of source would force {} keyframes; the limit is "
            "{}. That is all-intra at all-intra's price — ask for keyframes: 'all' if that is "
            "what you want, or a longer interval.".format(
                interval, float(duration_s), count, MAX_KEYFRAMES),
        )


def resolve_codec(codec, source_codec):
    """`"source"` into a concrete codec, against what the source actually is.

    **The master is re-encoded from raw frames, so `"source"` can never mean `-c:v copy`** — there
    is no stream to copy. It means "encode with the codec the source used", and that is only
    answerable once the source has been probed, which is why it is resolved here rather than in
    `derive`: the request surface is validated at the door, and the door has not opened the file.

    **UNRULED, and implemented conservatively rather than left to guess.** Contract §5c lists
    `"source"` as a legal value and says nothing about what the worker does with it;
    `envelope_cases.py` section 2 asserts only that `derive` ACCEPTS it. So the behaviour below is
    the builder's reading and is filed as a claim: a source whose codec this worker can encode is
    honoured, and **anything else is refused rather than quietly encoded as h264.** A silent
    fallback would be the silent-reinterpretation class — a caller who asked to stay in their own
    codec receiving a different one and no word about it.

    Neither of the two codec-cost jobs exercises this path: both name their codec outright.
    """
    if codec != "source":
        return codec
    family = {"h264": "h264", "avc1": "h264", "hevc": "h265", "h265": "h265"}.get(
        (source_codec or "").lower())
    if family is None:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "output.codec 'source' asks for the source's own codec and this source is {!r}, "
            "which this worker does not encode. The master is re-encoded from raw frames, so "
            "'source' selects an encoder rather than copying a stream. Name 'h264' or 'h265' "
            "explicitly.".format(source_codec or "unknown"),
        )
    return family


def default_off_identity(release_2_params):
    """THE SAFETY PROPERTY, executable — `codec_default_unmoved`.

    A request naming none of the codec fields must derive a config that changes nothing. **This is
    what lets the development tier run new codec code while medium and high serve production**, and
    it is enforced by a local run before a dispatch, never by CI. Returns True or raises.
    """
    cfg = derive(release_2_params)
    assert cfg["codec"] == DEFAULT_CODEC, "an omitted codec must deliver h264"
    assert cfg["crf"] == encoder.DEFAULT_CRF, "an omitted crf must deliver 12"
    assert cfg["preset"] == encoder.DEFAULT_PRESET, "an omitted preset must deliver medium"
    assert cfg["head_keyframes"] is False, "an omitted head_keyframes must deliver off"
    assert cfg["release_2_equivalent"] is True
    return True
