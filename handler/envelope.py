"""Release 3's request surface: the codec, the retime-only spelling, the interpolation block.

`fable/envelope_oracle.py` is this section of the contract executable and is the authority — if
this file and that one disagree, one of them is a bug and it gets a decision entry rather than a
patch. The rules are written from contract §5c rather than copied from the oracle, for the reason
the retime plan is: two independent statements of one rule is the point, and agreement reached by
sharing code proves nothing.

**Kept out of `validation.py` deliberately.** That module is release 2's surface and is large; a
release-3 block folded into it would be indistinguishable from the fields that have always been
there, and the one property protecting production — that a request carrying none of these fields
behaves exactly as it did — is easiest to keep true when the new surface is one file that can be
read end to end.
"""
from errors import INVALID_FIELD_VALUE, MISSING_REQUIRED_FIELD, WorkerError

#: **`source` means "match the input's codec"**, which is a release-3 field and not a default.
CODECS = ("h264", "h265", "source")

#: **Unchanged, so an omitted field cannot move anything.** Every release-2 caller encodes h264
#: today and must still encode h264 after this ships — `default_off_identity` is the assertion.
DEFAULT_CODEC = "h264"

#: CF: request-carried, default 60.
DEFAULT_TARGET_FPS = 60.0

#: Relative to the upscale. **No default** — Phase 2's A/B rules it.
ROUTES = ("before", "after")

#: The two spellings of "what size do you want", either of which satisfies the release-2 rule.
SIZING_FIELDS = ("target_short_edge_px", "output_size")

INTERPOLATE_FIELDS = ("target_fps", "snap_tolerance", "route")


def derive(params):
    """`params` in, a normalised release-3 config out, or `WorkerError`.

    **Absent fields produce the release-2 answer exactly**, which is the property that lets the
    development tier run this code while other tiers serve customers.
    """
    params = dict(params or {})
    interpolate = params.get("interpolate")
    upscale = params.get("upscale")
    output = dict(params.get("output") or {})
    has_size = any(params.get(field) is not None for field in SIZING_FIELDS)

    codec = output.get("codec", DEFAULT_CODEC)
    if codec not in CODECS:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'output.codec' must be one of {}; got {!r}".format(CODECS, codec))

    # **`upscale: false` is explicit, and the tempting spelling was a trap.** Letting a missing
    # size field mean "no upscale" needs no new field and reuses the sizing refusal as the
    # discriminator — and a caller who wants interpolation AND an upscale but forgets the size
    # field would then silently receive a retime instead of an error. This project has paid for
    # the silent-reinterpretation class twice: an endpoint renamed by a defaulted `--name`, and
    # 16-bit sources downconverted without a word. A forgotten field must still refuse; a
    # deliberate retime says so in words.
    if upscale is not None:
        if upscale is not False:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "field 'upscale' is not a toggle: the only legal value is false, which asks for "
                "a retime with no upscaling. Omit it to upscale.")
        if has_size:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "'upscale: false' contradicts a sizing field. A retime does not resize, so say "
                "one or the other.")
        if interpolate is None:
            raise WorkerError(
                MISSING_REQUIRED_FIELD,
                "'upscale: false' with no 'interpolate' asks the worker to do nothing.")
    elif not has_size:
        # Release-2 behaviour, restated only because release 3 must not weaken it.
        raise WorkerError(
            MISSING_REQUIRED_FIELD,
            "a request must say what size it wants: either 'target_short_edge_px' (one edge, "
            "aspect preserved) or 'output_size' as {'width': W, 'height': H} (an exact canvas). "
            "Neither was given in 'params'.")

    if interpolate is None:
        # **Named at the top level, they mean nothing and are refused rather than ignored.** A
        # caller who put `target_fps` beside `params` instead of inside `interpolate` has asked
        # for something; silence would deliver the opposite of it.
        for orphan in INTERPOLATE_FIELDS:
            if orphan in params:
                raise WorkerError(
                    INVALID_FIELD_VALUE,
                    "field '{}' has no meaning without 'interpolate'".format(orphan))
        return {"codec": codec, "interpolate": None, "upscale": True,
                "release_2_equivalent": codec == DEFAULT_CODEC}

    if not isinstance(interpolate, dict):
        raise WorkerError(INVALID_FIELD_VALUE, "field 'interpolate' must be an object")
    unknown = sorted(set(interpolate) - set(INTERPOLATE_FIELDS))
    if unknown:
        raise WorkerError(
            INVALID_FIELD_VALUE, "unknown field(s) in 'interpolate': {}".format(unknown))

    target_fps = interpolate.get("target_fps", DEFAULT_TARGET_FPS)
    if isinstance(target_fps, bool) or not isinstance(target_fps, (int, float)) \
            or target_fps <= 0:
        raise WorkerError(
            INVALID_FIELD_VALUE, "field 'interpolate.target_fps' must be a positive number")

    # **No default, and absent is not zero** (§5c). A `snap_tolerance` defaulted to 0 would ship
    # the unsnapped plan as the ruled answer before the benchmark that decides it has run.
    # Unruled must be visible as unruled, including in the code.
    tolerance = interpolate.get("snap_tolerance")
    if tolerance is not None and (isinstance(tolerance, bool)
                                  or not isinstance(tolerance, (int, float))
                                  or not 0.0 <= tolerance < 0.5):
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'interpolate.snap_tolerance' is a fraction of one source interval, in "
            "[0, 0.5)")

    # **No default either**, for the same reason: "before" by omission would settle Phase 2's A/B
    # without the A/B.
    route = interpolate.get("route")
    if route is not None and route not in ROUTES:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'interpolate.route' must be one of {}".format(ROUTES))
    if route is not None and upscale is False:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "'interpolate.route' names where interpolation sits relative to the upscale, and "
            "'upscale: false' has no upscale to sit beside.")

    return {
        "codec": codec,
        "interpolate": {"target_fps": float(target_fps), "snap_tolerance": tolerance,
                        "route": route},
        "upscale": upscale is not False,
        "release_2_equivalent": False,
    }


def default_off_identity(release_2_params):
    """One assertion: a request carrying none of release 3's fields behaves as it always did.

    **This is what lets the development tier run release-3 code while other tiers serve
    customers** — h264, no interpolation, upscaling on. Enforced by a local run before a dispatch,
    never by CI, which no longer sees the suite.
    """
    config = derive(release_2_params)
    return (config["codec"] == DEFAULT_CODEC
            and config["interpolate"] is None
            and config["upscale"] is True
            and config["release_2_equivalent"] is True)
