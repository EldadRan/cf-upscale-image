"""Request validation.

Two rules that pull in opposite directions, and both matter:

  **Unrecognised fields at the top level of `input` are ignored, not refused.** That is where CF
  adds things, and a worker that refused every unfamiliar name would reject every job the moment
  CF sent a new one.

  **A name the contract defines, offered where it is not accepted, is refused by name.** An
  ignored field there changes the output silently, which is the failure the rule prevents.

**The safety argument is structural, and it was not always.** Everything affecting the output
lives in `params`, so an unknown name at the *top level* is metadata by construction and ignoring
it is provably safe rather than conventionally safe.

Until CF's answers of 2026-08-12 this contract had no `params` block: `target_short_edge_px` and
`allow_oom_retry` sat at the top level and did affect the output, so the leniency rested on the
handoff's wording — *anything that changes the output is a named field* — which holds only while
everyone remembers it. CF adopted the media worker's split instead, and the pair of failure
directions is what it buys:

  **unknown inside `params` → refused by name.** CF has sent something that changes the output
  and this worker does not implement it. Failing loudly is correct.

  **unknown at the top level → ignored.** That is where CF adds things, and a worker that refused
  every unfamiliar name would reject every job the moment CF sent a new one.

Inside a `derive` entry there is no leniency either: anything the role does not take is refused,
known or not.
"""

import encoder
import envelope
import planner
from errors import (
    FIELD_NOT_SUPPORTED,
    INVALID_FIELD_VALUE,
    MISSING_REQUIRED_FIELD,
    WorkerError,
)

# The envelope: identity, where the bytes come from, where they go, and what set to produce.
# None of these change *how* the output is made. `debug` is a worker-side testing facility rather
# than a contract field.
TOP_LEVEL_FIELDS = {
    "request_id",
    "source_url",
    "output",
    "diagnostics",
    # **A presigned PUT the worker keeps, for the failure with no job attached** (CF, 2026-08-15).
    # Every diagnostics URL until now arrived *with* a job, so the class CF cannot see at all is a
    # worker that dies before it can report: an init that fails, a weight that will not load, a
    # driver mismatch. This one is retained across jobs and used when nothing job-scoped exists.
    #
    # Top level rather than in `params`, for `execution_timeout_ms`' reason exactly: it changes
    # whether a failure can be *reported*, never what the output looks like.
    #
    # **Accepted before CF sends it**, deliberately, and this time the leniency rule is the whole
    # argument. An unknown top-level field is ignored rather than refused, so a worker that did not
    # know this name would discard it silently — no error, no diagnostic, and nothing to indicate
    # either. Which is also why the name was asked for rather than guessed.
    "diagnostics_reserve",
    # **Where this run's record goes** (CF, 2026-08-19, F-2026-08-19-36). A presigned PUT beside
    # `diagnostics` and minted the same way, because the two answer different questions about the
    # same job: the bundle says what went wrong and exists only when something did, this says what
    # happened and exists on every run.
    #
    # **The address arrives with the request rather than living on the endpoint.** A standing
    # credential provisioned as an endpoint secret was designed and rejected: the caller owns its
    # telemetry destination, and a worker holding a long-lived write credential to somebody's
    # bucket is a durable liability for an object written once per job.
    #
    # Absent is a supported state, not a degraded one — the record is skipped and says so. Same
    # leniency argument as `diagnostics_reserve`: an unknown top-level field is ignored, so the
    # name was ruled rather than guessed.
    "run_record",
    "derive",
    "params",
    "debug",
    # **A calibration facility, not part of CF's contract.** The estimator chooses from measured
    # peaks, and it can only measure the rung it ran — so with an empty table it runs the floor
    # for ever and never learns that anything faster fits. Pinning the rung is how the table gets
    # its first rows. CF never sends this; if it ever appears in a CF request that is a mistake
    # worth catching, so it is validated rather than ignored.
    "force_rung",
    # **The same calibration facility, one level finer.** `batch_size` is *frames per model
    # batch* — the model's temporal window, and on CF's account the dominant quality lever for
    # video: a bigger batch sees more of the motion at once. It only ever moved as part of a rung,
    # bundled with five other changes, so its effect has never been isolated from theirs. These
    # two let one knob move at a time. CF never sends them.
    "force_batch_size",
    "force_temporal_overlap",
    # **And the one that turned out to gate the other.** `chunk_size` is how many frames are
    # streamed to the model at a time, so the temporal window is `min(batch_size, chunk_size)` —
    # a batch larger than the chunk is silently the chunk. Measured the hard way: on `swapped`,
    # which chunks at 9, batch 21, 33, 49 and 65 all produced byte-identical masters at an
    # identical 23.15 GB peak. Three runs of an experiment that had already finished.
    #
    # Without this there is no way to raise the window on a memory-constrained rung at all: every
    # rung that can hold 4K in an A40 chunks below its own batch size, so the quality lever CF
    # names as dominant cannot be moved from outside.
    "force_chunk_size",
    # **The rest of the levers, so that "one knob at a time" is true rather than aspirational.**
    # Until these existed the only way to change tiling or block-swapping was to change rung —
    # which also changes the window, which is the confound that made every tiling question
    # unanswerable (`decisions.md` 4.41). Each of these moves exactly one thing.
    #
    # Encode and decode tiling are separate flags on separate *frames*: encode works on the
    # input, decode on the output. At 1080p in and 4K out the same tile size gives a 2x2 grid on
    # one side and 5x3 on the other, so a single number applied to both is two different
    # decisions wearing one name. Every calibration run so far tiled the encode at whatever the
    # rung set — 512 on `swapped`, which cross-fades 44% of every input frame before the model
    # sees it.
    "force_vae_encode_tiled",
    "force_vae_encode_tile_size",
    "force_vae_encode_tile_overlap",
    "force_vae_decode_tiled",
    "force_vae_decode_tile_size",
    "force_vae_decode_tile_overlap",
    # Block-swapping is the one memory lever with **no quality cost at all** — it trades VRAM for
    # time and nothing else — and it has never moved on its own.
    "force_blocks_to_swap",
    "force_swap_io_components",
    # **A pinned configuration must fail, not ratchet.** Without this every limit-finding run
    # silently becomes a run of something else: the ratchet steps the rung, the job succeeds, and
    # the row banked describes a configuration nobody asked for. That is how 68 of 70 peak
    # measurements ended up unattributable. `pin` says: run exactly this, and if it does not fit,
    # say so.
    "pin",
    # **Which upscaler handles the alpha channel, and it is a temporary field.** This worker takes
    # alpha out before the model and resizes it with Lanczos; the model can carry it itself and
    # interpolate it along the edges it just produced (`decisions.md` 4.9). The second is better
    # by every reading of the vendored source and is measured at nothing, so it is a flag until a
    # real cutout has gone through both. Whichever wins becomes the behaviour and this name goes
    # away. CF never sends it.
    "keep_alpha_in_model",
    # **The job's own deadline, set by CF at submit and sent to RunPod in the same breath.**
    # Spelled exactly as RunPod's execution policy spells it, so one integer reaches two
    # recipients with nothing translated between them.
    #
    # Top level rather than `params` because `params` is provably "everything that changes the
    # output"; a deadline changes *whether there is* an output, not what it looks like. It is
    # envelope, next to `request_id` and the URLs.
    #
    # **Accepted before CF sends it**, deliberately: unknown top-level fields are ignored under
    # the leniency rule, so a worker that did not know this name would silently discard it and
    # keep failing at the wall. Support ships first; CF starts sending second.
    "execution_timeout_ms",
    # **Price the job and return, without touching the GPU.** Here rather than in `params` for
    # the same reason as the deadline: it does not change what the output looks like, it changes
    # whether one is produced. The kit uses it to certify the deployed planner by calling it —
    # acceptance stops needing paid runs to check the decision logic — and it goes through the
    # *production* code path, so what it reports is what a real run would have done.
    "plan_only",
}

REQUIRED_TOP_LEVEL = ("request_id", "source_url", "output", "params")

# Everything that changes the output. Strict: a name here that this worker does not implement is
# refused by name rather than ignored.
PARAMS_FIELDS = {
    "target_short_edge_px",
    # **An exact canvas, for callers who have one.** `target_short_edge_px` fixes one edge and
    # derives the other from the source's aspect, which cannot express "land on this frame".
    # Measured need: CF separates an image into RGBA layers, the separator returns them at 864x480
    # against a 1376x768 canvas, and a short-edge target produces 1382x768 — six pixels that make
    # the composite wrong. The separator had rounded 860 up to 864 to reach its own 16 grid and
    # *stretched* the content 0.465% doing it, so cropping cannot repair it and a resize to the
    # canvas can (`decisions.md` 4.15).
    #
    # **An object, not two scalars** (CF, 2026-08-15). This is CF's existing vocabulary on its image
    # models, and it carries `dimension_bounds` — the only capability CF has that bounds *a pixel
    # budget* rather than an edge, which is the shape every measurement in this repo took. The
    # megapixel ceiling maps onto `max_pixels` with nothing invented. Shipped first as
    # `target_width` + `target_height`; that spelling never reached a caller and is gone rather
    # than aliased, because two names for one field is how a contract acquires a wrong one.
    "output_size",
    "allow_oom_retry",
    "color_correction",
    "keep_audio",
    # **The master's constant-rate factor** (CF, 2026-08-18, pulled forward from the parked
    # encoder track). Default 12, which is what `encoder.py` has silently baked since the first
    # commit — so the default is today's behaviour named rather than changed. Applies to the
    # master's encode only: codec, preset, pixel format and the derives' own settings stay with
    # the encoder track. Recorded in the manifest and the ledger, because a master's CRF is part
    # of what that master *is*.
    "crf",
    # **The decode-seam lever** (CF, 2026-08-18). `default` takes the formula's own pick, the
    # coarsest grid under the decode time knee; `high` waives the knee and buys the coarsest grid
    # memory alone allows — 4K 2x1 at ~2.3x decode time, 8K 3x2 at ~3.7x. Nothing in between is
    # offered until the E3 A/B shows seams are worth intermediate rungs, because an option nobody
    # can choose between on evidence is a way of moving the decision to the caller rather than
    # making it.
    "tile_quality",
    # **The tail lever** (CF, 2026-08-18, decision 8's switch). `max_window` keeps the window at
    # the card's maximum and blends a short final pass in with 1-2 frames of overlap;
    # `balanced` steps the window down the lattice until no pass falls below the floor, trading
    # every window's quality for an even tail. Neither is imposed: the default favours the body,
    # balanced the tail, and which matters more is per-job judgment only the caller has.
    "schedule",
    # ── release 3 ────────────────────────────────────────────────────────────────────────────
    # **Validated in `envelope.py`, not here.** Release 2's surface is large and a release-3
    # block folded into it would be indistinguishable from the fields that have always been
    # there — and the one property protecting production is that a request carrying none of
    # these behaves exactly as it did. Naming them here is what stops `_refuse_unknown` rejecting
    # them by name; the rules that govern them live in one file that can be read end to end.
    "upscale",
    "interpolate",
    # **`params.output` is the ENCODE, and the top-level `output` is the DESTINATION.** One word,
    # two objects, and the collision is the contract's spelling (§5c) rather than a choice made
    # here. Raised to the gate as a claim.
    "output",
}

#: **Nothing is unconditionally required, and that is the point.** `target_short_edge_px` is
#: required only when no `output_size` is given (CF, 2026-08-15). The alternative — keeping it
#: required always — would have had CF derive a short edge purely to satisfy a field it was
#: simultaneously overriding, putting *two* sizing forms on every exact-canvas request in order to
#: preserve a convention about there being one.
#:
#: CF guarantees exactly one form arrives, rejecting a caller who sends both before dispatch. The
#: precedence rule below is kept anyway, as a backstop against a CF bug: a job that completes and
#: warns beats one that dies on a technicality after the GPU is spent.
SIZING_FIELDS = ("target_short_edge_px", "output_size")

# **Default `true`, and it inverts the platform's `keep_audio: false` deliberately.** That default
# exists because several generators invent a soundtrack nobody asked for, so silence is the safe
# answer. Here the track is the caller's **own source audio**, and returning a customer's video
# muted is losing something they supplied rather than suppressing something a model made up — so
# the same rule applied to a different question gives the opposite answer.
#
# The model carries no audio at all (`docs/decisions.md` 0.4); this comes out of the worker's own
# mux as a stream copy. CF sends the value explicitly, as it does `color_correction`; this default
# covers a bare invocation.
DEFAULT_KEEP_AUDIO = True

# Read off the pinned SeedVR2 source's own argparse `choices`, not from a log or a README, so
# this set cannot drift from what the vendored code accepts (docs/decisions.md 0.5). The CLI's
# own default is `lab` — described upstream as "perceptual color matching, recommended" — and
# **not** `wavelet`, which is what the image worker hardcodes and what CF's specs discuss.
#
# CF decides what is advertised and what the default is. The worker decides nothing here; what
# it owes CF is what each value does observably on video, which is a measurement nobody has
# taken. Until then this worker sends the upstream default rather than inheriting the image
# worker's choice, because inheriting it would carry a decision nobody made for video.
COLOR_CORRECTIONS = ("lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none")
DEFAULT_COLOR_CORRECTION = "lab"

#: x264's own range, and the whole of it. 0 is lossless and 51 is unwatchable; both are legal
#: things to ask an encoder for, and a worker inventing a narrower band would refuse work that
#: would have succeeded. The default is `encoder.DEFAULT_CRF`, imported rather than repeated.
CRF_MIN, CRF_MAX = 0, 51

#: The two decode-seam settings, and deliberately only two.
TILE_QUALITIES = ("default", "high")
DEFAULT_TILE_QUALITY = "default"

#: The two tail policies. `max_window` is the default because the body of a clip is almost
#: always more of it than the tail — but "almost always" is not "always", which is why the other
#: exists rather than a formula choosing between them.
SCHEDULES = ("max_window", "balanced")
DEFAULT_SCHEDULE = "max_window"

DERIVE_ROLES = ("poster", "proxy", "crop")

# Strict inside a derive entry: a field the *role* does not take is refused. So `at_fraction` on
# a proxy and `max_duration_s` on a poster are both errors — each would otherwise be accepted
# and silently ignored.
DERIVE_FIELDS_BY_ROLE = {
    "poster": {"role", "at_fraction"},
    "proxy": {"role", "max_duration_s"},
    # `crop` takes `at_fraction` too — every crop comes from one frame and the set is only
    # comparable if they share it. Agreed with CF (bf4471c).
    "crop": {"role", "count", "select", "at_fraction"},
}

CROP_SELECT_MODES = ("detail", "centre", "spread")
CROP_COUNT_DEFAULT = 3
CROP_COUNT_MAX = 8
DERIVE_FIELDS = {f for fs in DERIVE_FIELDS_BY_ROLE.values() for f in fs}

OUTPUT_FIELDS_REQUIRED = ("endpoint", "bucket", "prefix", "access_key_id", "secret_access_key")
#: `name` is the caller's stem for the master (F-2026-08-19-38) — optional, and absent is a
#: supported state that delivers today's names byte-for-byte. It is validated only as a string
#: here; what makes it safe to use as a key segment is `keys.sanitize_stem`, which is where the
#: rule belongs because it is the module that owns what a name may be.
OUTPUT_FIELDS_OPTIONAL = ("session_token", "name")

# Every field name the contract defines, anywhere. A name in here, offered somewhere it is not
# accepted, is refused; a name outside it is metadata at the top level and ignored.
KNOWN_FIELD_NAMES = set(TOP_LEVEL_FIELDS) | set(PARAMS_FIELDS) | set(DERIVE_FIELDS) \
    | set(OUTPUT_FIELDS_REQUIRED) | set(OUTPUT_FIELDS_OPTIONAL)


def _rung_name(value):
    """`None`, or one of the estimator's rung names. Anything else is refused.

    Validated against `estimator.RUNGS` rather than a list restated here, so a rung renamed in one
    place cannot be silently unreachable from the other.
    """
    if value is None:
        return None
    import estimator  # local: validation must stay importable without the estimator's deps

    names = [rung["name"] for rung in estimator.RUNGS]
    if value not in names:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "force_rung must be one of {}, got {!r}".format(", ".join(names), value),
        )
    return value


def _positive_int_or_none(value, field, maximum, minimum=0):
    """Absent, or a plain integer within reach.

    `minimum` is per field rather than fixed at zero: `force_temporal_overlap: 0` is meaningful
    (blend nothing), while `execution_timeout_ms: 0` is not — and zero would be read as *absent*
    by every falsy test downstream, so a caller sending it would silently get no deadline at all
    instead of a refusal. A value that means something different from what it says is worse than
    one that is rejected.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerError(INVALID_FIELD_VALUE,
                          "{} must be an integer, got {!r}".format(field, value))
    if value < minimum or value > maximum:
        raise WorkerError(INVALID_FIELD_VALUE,
                          "{} must be between {} and {}, got {}".format(
                              field, minimum, maximum, value))
    return value


def _refuse_unknown(present, allowed, where):
    """Strict: anything not accepted here is refused, known to the contract or not."""
    for field in sorted(set(present) - set(allowed)):
        raise WorkerError(
            FIELD_NOT_SUPPORTED, "field '{}' is not accepted {}".format(field, where)
        )


def _refuse_known_but_unaccepted(present, allowed, where):
    """Lenient: refuse names the contract defines, ignore names it does not."""
    for field in sorted(set(present) - set(allowed)):
        if field in KNOWN_FIELD_NAMES:
            raise WorkerError(
                FIELD_NOT_SUPPORTED, "field '{}' is not accepted {}".format(field, where)
            )


def _require(mapping, field, where):
    if mapping.get(field) is None:
        raise WorkerError(
            MISSING_REQUIRED_FIELD, "field '{}' is required {}".format(field, where)
        )
    return mapping[field]


def _as_int(value, field):
    # bool is an int subclass in Python; an explicit reject keeps `true` out of a pixel count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerError(INVALID_FIELD_VALUE, "field '{}' must be an integer".format(field))
    return value


def _bool_or_none(value, field):
    """A forced boolean, where absent means "the configuration decides" rather than False.

    Distinct from `_as_bool` because these three states are all meaningful for a forcing field:
    force it on, force it off, or do not force it. Collapsing absent to False would make
    `force_vae_decode_tiled` unsettable to True by omission — and, worse, would silently turn
    tiling *off* on every job that never mentioned it.
    """
    return None if value is None else _as_bool(value, field)


def _as_bool(value, field):
    if not isinstance(value, bool):
        raise WorkerError(INVALID_FIELD_VALUE, "field '{}' must be a boolean".format(field))
    return value


def _as_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerError(INVALID_FIELD_VALUE, "field '{}' must be a number".format(field))
    return float(value)


def _as_str(value, field):
    if not isinstance(value, str):
        raise WorkerError(INVALID_FIELD_VALUE, "field '{}' must be a string".format(field))
    return value


#: Both keys, and only these two. An `output_size` carrying one of them has said something the
#: contract cannot act on, and guessing which of the two readings it meant — "this width, aspect
#: free" or "this width, and derive the height" — is how a caller gets an output they did not ask
#: for and no message explaining why. The first reading already has a field.
OUTPUT_SIZE_FIELDS = ("width", "height")

#: Not a capacity limit — the capacity refusal is, from measured VRAM against the card in hand.
#: This only catches a value that cannot be a canvas at all, so a transposed field or a byte count
#: fails here rather than at the fit, after the GPU is spent.
OUTPUT_SIZE_MAX_EDGE = 65_536


def _validate_output_size(value):
    """`{'width': W, 'height': H}` on the wire, `(W, H)` out, None if absent."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'output_size' must be an object like {'width': 1376, 'height': 768}, got "
            "{}. For one edge with the aspect left free, use 'target_short_edge_px'."
            .format(type(value).__name__),
        )
    _refuse_unknown(value, OUTPUT_SIZE_FIELDS, "in 'output_size'")
    for field in OUTPUT_SIZE_FIELDS:
        if value.get(field) is None:
            raise WorkerError(
                MISSING_REQUIRED_FIELD,
                "'output_size' needs both 'width' and 'height'; '{}' is missing. An exact canvas "
                "is two numbers, and one of them alone is ambiguous against "
                "'target_short_edge_px'.".format(field),
            )
    return tuple(
        _positive_int_or_none(value[field], "output_size." + field, OUTPUT_SIZE_MAX_EDGE,
                              minimum=1)
        for field in OUTPUT_SIZE_FIELDS
    )


def _validate_output(output):
    if not isinstance(output, dict):
        raise WorkerError(INVALID_FIELD_VALUE, "field 'output' must be an object")
    _refuse_unknown(output, OUTPUT_FIELDS_REQUIRED + OUTPUT_FIELDS_OPTIONAL, "in 'output'")
    for field in OUTPUT_FIELDS_REQUIRED:
        _as_str(_require(output, field, "in 'output'"), "output." + field)
    if output.get("session_token") is not None:
        _as_str(output["session_token"], "output.session_token")
    if output.get("name") is not None:
        _as_str(output["name"], "output.name")
    prefix = output["prefix"]
    # Every file goes under the prefix and the keys are the worker's within it. A prefix that
    # does not end in `/` would make `prefix + name` a sibling of the prefix rather than a child
    # of it, which is a write outside the scope the credential grants — so it fails at R2 rather
    # than here, with a message about credentials instead of about the request.
    if not prefix.endswith("/"):
        raise WorkerError(
            INVALID_FIELD_VALUE, "field 'output.prefix' must end with '/', got {!r}".format(prefix)
        )
    return output


def _validate_derive(derive):
    if not isinstance(derive, list):
        raise WorkerError(INVALID_FIELD_VALUE, "field 'derive' must be an array")

    seen = set()
    validated = []
    for entry in derive:
        if not isinstance(entry, dict):
            raise WorkerError(INVALID_FIELD_VALUE, "each 'derive' entry must be an object")
        role = _as_str(_require(entry, "role", "in a 'derive' entry"), "derive.role")
        if role not in DERIVE_ROLES:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "derive role {!r} is not one of {}".format(role, ", ".join(DERIVE_ROLES)),
            )
        # A role may appear at most once. Each writes one deterministic key, so a repeat
        # overwrites in silence and returns a manifest claiming two files where one exists.
        if role in seen:
            raise WorkerError(
                INVALID_FIELD_VALUE, "derive role {!r} appears more than once".format(role)
            )
        seen.add(role)

        _refuse_unknown(entry, DERIVE_FIELDS_BY_ROLE[role], "on a '{}' derive".format(role))

        clean = {"role": role}
        if role in ("poster", "crop"):
            at_fraction = entry.get("at_fraction")
            at_fraction = 0.25 if at_fraction is None else _as_number(
                at_fraction, "derive.at_fraction")
            if not 0.0 <= at_fraction <= 1.0:
                raise WorkerError(
                    INVALID_FIELD_VALUE,
                    "field 'at_fraction' must be between 0 and 1, got {}".format(at_fraction),
                )
            clean["at_fraction"] = at_fraction

        if role == "crop":
            count = entry.get("count")
            count = CROP_COUNT_DEFAULT if count is None else _as_int(count, "derive.count")
            if not 1 <= count <= CROP_COUNT_MAX:
                raise WorkerError(
                    INVALID_FIELD_VALUE,
                    "field 'count' must be between 1 and {}, got {}".format(
                        CROP_COUNT_MAX, count),
                )
            clean["count"] = count
            select = entry.get("select")
            select = "detail" if select is None else _as_str(select, "derive.select")
            if select not in CROP_SELECT_MODES:
                raise WorkerError(
                    INVALID_FIELD_VALUE,
                    "select {!r} is not one of {}".format(
                        select, ", ".join(CROP_SELECT_MODES)),
                )
            clean["select"] = select
        elif role == "proxy":
            max_duration_s = entry.get("max_duration_s")
            if max_duration_s is not None:
                max_duration_s = _as_number(max_duration_s, "derive.max_duration_s")
                if max_duration_s <= 0:
                    raise WorkerError(
                        INVALID_FIELD_VALUE,
                        "field 'max_duration_s' must be positive, got {}".format(max_duration_s),
                    )
                clean["max_duration_s"] = max_duration_s
        validated.append(clean)
    return validated


def validate(job_input):
    """Return the request, normalised. Raises `WorkerError` on anything the contract refuses."""
    if not isinstance(job_input, dict):
        raise WorkerError(INVALID_FIELD_VALUE, "'input' must be an object")

    # Lenient at the top level: unknown names are metadata by construction, since everything that
    # changes the output lives in `params`.
    _refuse_known_but_unaccepted(job_input, TOP_LEVEL_FIELDS, "at the top level of 'input'")

    for field in REQUIRED_TOP_LEVEL:
        _require(job_input, field, "at the top level of 'input'")

    request_id = _as_str(job_input["request_id"], "request_id")
    if not request_id.strip():
        raise WorkerError(INVALID_FIELD_VALUE, "field 'request_id' must not be empty")

    params = job_input["params"]
    if not isinstance(params, dict):
        raise WorkerError(INVALID_FIELD_VALUE, "field 'params' must be an object")
    # Strict inside `params`: a name CF sent here changes the output, and this worker not
    # implementing it must be loud rather than silent.
    _refuse_unknown(params, PARAMS_FIELDS, "in 'params'")

    output_size = _validate_output_size(params.get("output_size"))

    # **Release 3's surface, derived before the sizing rule because it can suspend it.**
    # `upscale: false` is the explicit retime spelling, and a retime does not resize — so the
    # requirement that a request say what size it wants is release 2's rule and stays exactly
    # that. `envelope.derive` refuses the contradiction (a size beside `upscale: false`) and the
    # emptiness (`upscale: false` with nothing to do) itself.
    release_3 = envelope.derive(params)

    # **What the contract accepts and this worker cannot yet serve is REFUSED, not ignored.**
    # `envelope.derive` is §5c complete and correct; the paths behind two of its answers are not
    # wired. Accepting them would deliver the opposite of what was asked — an `upscale: false`
    # request would be planned as an upscale and die on a null size, and an `h265` request would
    # return h264 without a word. That is the silent-reinterpretation class this whole section
    # exists to prevent, and a field that validates and then does nothing is worse than one
    # refused by name, because the caller has evidence it was understood.
    if not release_3["upscale"]:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "'upscale: false' is contract-legal and this worker cannot serve it yet: the retime "
            "path exists but no request reaches it (release-3 plan, Phase 1 step 4). Refused "
            "rather than silently upscaled.")
    if release_3["codec"] != envelope.DEFAULT_CODEC:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "'output.codec: {}' is contract-legal and this worker cannot serve it yet; only {!r} "
            "is implemented. Refused rather than silently encoded as {}.".format(
                release_3["codec"], envelope.DEFAULT_CODEC, envelope.DEFAULT_CODEC))
    if release_3["interpolate"] is not None:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "'interpolate' is contract-legal and this worker cannot serve it yet: the shim and "
            "the retime path exist but no request reaches them. Refused rather than silently "
            "delivering the source rate.")

    target = params.get("target_short_edge_px")
    if target is None and not release_3["upscale"]:
        pass
    elif target is None:
        # One of the two has to arrive. Naming both in the message matters: a caller who omitted
        # the short edge because they *meant* to send `output_size` and mistyped it has already
        # been told about the typo by `_refuse_unknown`, and a caller who sent neither is being
        # told the contract rather than scolded about one field.
        if output_size is None:
            raise WorkerError(
                MISSING_REQUIRED_FIELD,
                "a request must say what size it wants: either 'target_short_edge_px' (one edge, "
                "aspect preserved) or 'output_size' as {'width': W, 'height': H} (an exact "
                "canvas). Neither was given in 'params'.",
            )
    else:
        target = _as_int(target, "target_short_edge_px")
        # Type and positivity only. **No maximum, deliberately.** The bounds are CF's product
        # choice, and a bound the worker invents refuses work that would have succeeded — the
        # failure this model row has already produced once, on an input rule CF withdrew. A target
        # below the source's short edge is a downscale, which the model supports: permitted, warned
        # in the response, never refused. What the worker owes CF is where quality and memory
        # actually fall off, which is a measurement, not a constant.
        if target <= 0:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "field 'target_short_edge_px' must be positive, got {}".format(target),
            )

    allow_oom_retry = params.get("allow_oom_retry")
    allow_oom_retry = True if allow_oom_retry is None else _as_bool(
        allow_oom_retry, "allow_oom_retry")

    keep_audio = params.get("keep_audio")
    keep_audio = DEFAULT_KEEP_AUDIO if keep_audio is None else _as_bool(
        keep_audio, "keep_audio")

    color_correction = params.get("color_correction")
    color_correction = DEFAULT_COLOR_CORRECTION if color_correction is None else _as_str(
        color_correction, "color_correction")
    if color_correction not in COLOR_CORRECTIONS:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "color_correction {!r} is not one of {}".format(
                color_correction, ", ".join(COLOR_CORRECTIONS)),
        )

    crf = params.get("crf")
    if crf is None:
        crf = encoder.DEFAULT_CRF
    else:
        crf = _as_int(crf, "crf")
        if not CRF_MIN <= crf <= CRF_MAX:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "field 'crf' must be within x264's range {}-{}, got {}. Lower is better quality "
                "and a larger file; {} is this worker's default and what every measurement in "
                "its calibration was taken at.".format(
                    CRF_MIN, CRF_MAX, crf, encoder.DEFAULT_CRF),
            )

    tile_quality = params.get("tile_quality")
    tile_quality = DEFAULT_TILE_QUALITY if tile_quality is None else _as_str(
        tile_quality, "tile_quality")
    if tile_quality not in TILE_QUALITIES:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "tile_quality {!r} is not one of {}. 'default' takes the coarsest decode grid under "
            "the time knee; 'high' waives the knee for the coarsest grid memory allows, which "
            "minimises seams and costs decode time the plan prices for you.".format(
                tile_quality, ", ".join(TILE_QUALITIES)),
        )

    schedule = params.get("schedule")
    schedule = DEFAULT_SCHEDULE if schedule is None else _as_str(schedule, "schedule")
    if schedule not in SCHEDULES:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "schedule {!r} is not one of {}. 'max_window' keeps the temporal window at the "
            "card's maximum and blends a short final pass into its predecessor; 'balanced' "
            "narrows every window until no pass falls below the {}-frame floor, which costs the "
            "whole clip a little quality to even out the tail.".format(
                schedule, ", ".join(SCHEDULES), planner.MIN_WINDOW),
        )

    derive = job_input.get("derive")
    derive = [] if derive is None else _validate_derive(derive)

    diagnostics = job_input.get("diagnostics")
    if diagnostics is not None:
        diagnostics = _as_str(diagnostics, "diagnostics")

    reserve = job_input.get("diagnostics_reserve")
    if reserve is not None:
        reserve = _as_str(reserve, "diagnostics_reserve")

    run_record = job_input.get("run_record")
    if run_record is not None:
        run_record = _as_str(run_record, "run_record")

    # Flattened for the handler's use. The **wire** shape is nested; this is the normalised form
    # everything downstream reads, so the nesting exists exactly once — here — rather than being
    # threaded through every caller.
    return {
        "request_id": request_id,
        "source_url": _as_str(job_input["source_url"], "source_url"),
        "target_short_edge_px": target,
        # The normalised release-3 config, one object rather than four loose keys, so a caller
        # downstream cannot read `interpolate` without also having what decided it.
        "release_3": release_3,
        "allow_oom_retry": allow_oom_retry,
        "keep_audio": keep_audio,
        "color_correction": color_correction,
        "crf": crf,
        "tile_quality": tile_quality,
        "schedule": schedule,
        "derive": derive,
        "output": _validate_output(job_input["output"]),
        "diagnostics": diagnostics,
        "diagnostics_reserve": reserve,
        "run_record": run_record,
        "debug": bool(job_input.get("debug")),
        "force_rung": _rung_name(job_input.get("force_rung")),
        # RunPod's own ceiling is 7 days. No lower bound beyond positive: a caller who sends a
        # deadline this worker cannot meet gets a refusal with the arithmetic, which is more
        # useful than an argument about whether the number was reasonable.
        "execution_timeout_ms": _positive_int_or_none(
            job_input.get("execution_timeout_ms"), "execution_timeout_ms", 604_800_000,
            minimum=1),
        # Snapped to the 4n+1 lattice by the pipeline, not refused here: a caller asking for 20
        # means "about twenty", and the nearest valid value is a better answer than an error.
        "force_batch_size": _positive_int_or_none(
            job_input.get("force_batch_size"), "force_batch_size", 129),
        "force_temporal_overlap": _positive_int_or_none(
            job_input.get("force_temporal_overlap"), "force_temporal_overlap", 32),
        # No lattice — chunk size is a streaming granularity, not a model constraint. The ceiling
        # is generous because the ceiling is not what protects memory: the capacity refusal and
        # the OOM ladder are, and a chunk that does not fit should be discovered by measurement
        # rather than forbidden by a number chosen in advance.
        # `(width, height)` or None. The wire shape is an object; this is the normalised form, and
        # the tuple is deliberate — it is passed straight to the fit, which takes a pair.
        "output_size": output_size,
        "force_chunk_size": _positive_int_or_none(
            job_input.get("force_chunk_size"), "force_chunk_size", 4096, minimum=1),
        # Tile sizes are floored to a multiple of 8 by the VAE (the grid is laid out in latent
        # space at scale factor 8), so a value between two multiples is silently the lower one.
        # Not snapped here: the worker reports what it was given and what actually applied, and
        # rounding on the caller's behalf hides the quantisation from the person calibrating.
        "force_vae_encode_tiled": _bool_or_none(
            job_input.get("force_vae_encode_tiled"), "force_vae_encode_tiled"),
        "force_vae_encode_tile_size": _positive_int_or_none(
            job_input.get("force_vae_encode_tile_size"), "force_vae_encode_tile_size",
            4096, minimum=8),
        "force_vae_encode_tile_overlap": _positive_int_or_none(
            job_input.get("force_vae_encode_tile_overlap"), "force_vae_encode_tile_overlap",
            2048),
        "force_vae_decode_tiled": _bool_or_none(
            job_input.get("force_vae_decode_tiled"), "force_vae_decode_tiled"),
        "force_vae_decode_tile_size": _positive_int_or_none(
            job_input.get("force_vae_decode_tile_size"), "force_vae_decode_tile_size",
            4096, minimum=8),
        "force_vae_decode_tile_overlap": _positive_int_or_none(
            job_input.get("force_vae_decode_tile_overlap"), "force_vae_decode_tile_overlap",
            2048),
        # 0-36 on the 7B checkpoint this image bakes. Refused above that rather than clamped: a
        # caller asking for 48 has misread the model, and silently giving them 36 would report a
        # calibration row against a configuration they did not run.
        "force_blocks_to_swap": _positive_int_or_none(
            job_input.get("force_blocks_to_swap"), "force_blocks_to_swap", 36),
        "force_swap_io_components": _bool_or_none(
            job_input.get("force_swap_io_components"), "force_swap_io_components"),
        "pin": (False if job_input.get("pin") is None
                else _as_bool(job_input["pin"], "pin")),
        "keep_alpha_in_model": (
            False if job_input.get("keep_alpha_in_model") is None
            else _as_bool(job_input["keep_alpha_in_model"], "keep_alpha_in_model")),
        # **Plan-only: price the job and return, without touching the GPU.** Top level rather
        # than in `params`, because it does not change what the output *is* — it says whether to
        # produce one. The source is still fetched and probed, because the plan is a function of
        # the real geometry and a plan computed from a caller's guess at the frame count would be
        # answering a different question from the one the run would ask.
        "plan_only": bool(job_input.get("plan_only")),
    }
