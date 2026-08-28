"""The shadow time model. **Computed on every job, recorded, and consumed by nothing.**

    T_seconds  =  frames * output_megapixels * r_card

CF ruled on 2026-08-28 that the old model stays authoritative until the new one delivers, and
that both are computed and both recorded. `docs/gate/time-model.md` §9 is the design and §9's
"what delivers has to mean" is the exit criterion.

**THIS MODULE IS A LEAF ON PURPOSE.** Nothing in the planning or refusal path imports it — the
only caller is the run-record assembly. That is what makes "shadow" a property of the code rather
than a promise about it: if this number ever reaches a decision, the import graph says so before
any reviewer has to. The deadline gate goes on consuming `calibration.json`'s lookup, unchanged.

## What it can promise, and what it must never be rendered as

**A LOWER BOUND WITH A KNOWN ERROR DIRECTION.** `r_card` is fitted at the clean MINIMUM per card,
so `true >= estimate` by construction. It is NOT a point estimate: two runs at 13.61 Mpx, window
1, same tiling, same worker id, same datacentre, same core count and the same image came out 21.6
and 40.7 s/frame — 1.9x apart with every recorded variable identical. A bound stated as an ETA is
a promise the model has no basis for, and this module returns `estimate_seconds` under a key that
says `bound`, never `eta`.

## An unmeasured card produces NO number

Not zero, not a fallback, not another card's rate — **absent, with the absence recorded.** That is
the whole defect this replaces: the old model's `same_card or comparable` filter matched nothing,
filtered nothing, and then `max()` preferred the row furthest away, pricing a delivered job at
897 s against 161. **Hardware priors do not rescue a missing card either** — rates do not track
bandwidth or FLOPs cleanly across these four (the B200 measures SLOWER per megapixel than the
H200), so cross-card reasoning gives an ordering sanity check and not a calibration.

A row saying `absent, this card has never been measured` is the row that tells CF coverage is
missing. A row carrying a borrowed number tells CF nothing and looks like an answer.
"""

import math

#: **Copied by hand from `docs/gate/registry-v1.json`'s `time_model_v0`, and that file is its
#: home.** The build cannot read it: the docker context is `./handler` in the public repository
#: and the private project repository is not checked out on the runner at all — the same
#: constraint that makes `planner.py`'s registry constants and `bake_weights.py`'s hashes copies.
#: Anything that changes there has to be brought here by hand until something generates it.
#:
#: **THIS IS NOT A CALIBRATED REGISTRY LINE AND MUST NOT ACQUIRE THAT AUTHORITY** by sitting in
#: the same file as the memory constants. One to seven runs per card, bounds rather than fitted
#: coefficients, no span rule, and no `_matching_runs` discipline behind it. The registry block
#: says so itself and this comment repeats it because a constant in a handler is read by people
#: who will never open the registry.
R_CARD = {
    "NVIDIA H200": {"rate": 0.407, "n_clean": 6, "mpx_span": (3.89, 33.18)},
    "NVIDIA A40": {"rate": 1.222, "n_clean": 7, "mpx_span": (8.29, 8.29)},
    "NVIDIA B200": {"rate": 0.520, "n_clean": 2, "mpx_span": (33.18, 33.18)},
    #: **SINGLE MEASUREMENT.** May be recorded; may not drive anything, now or when the shadow
    #: lifts. A single row cannot establish that it was a clean run — the 21.6/40.7 twins are the
    #: proof — and a degraded row baked into `r_card` inflates every future bound, manufacturing
    #: refusals that nobody can trace back to it.
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": {
        "rate": 0.635, "n_clean": 1, "mpx_span": (8.29, 8.29),
        "single_measurement": True},
}

#: The fit's own basis, carried into every record so a row can be read years later without the
#: document. `time-model.md` §3 and the registry's `_basis`.
MODEL_VERSION = "v0"
FIT_BASIS = ("24 delivered run-records with a measured strip, batched only; rates are "
             "(attempt_seconds - strip) / frames / output_megapixels, fitted at the clean "
             "minimum per card")


def predict(gpu_name, frames, output_pixels, still=False):
    """The shadow estimate for one job, or a stated absence. **Never raises.**

    Returns a dict that always carries `model`, `keyed_on` and `estimate_seconds`, so a record
    with no estimate is distinguishable from a record written before this existed. Every refusal
    to answer names itself in `absent_because`.

    `frames` is the count the job will actually run; `output_pixels` the delivered plane. The
    strip is EXCLUDED by construction — the fit was taken net of it — so this predicts model time
    and not wall time, and the record says so rather than leaving the two to be conflated.
    """
    row = R_CARD.get(gpu_name)
    answer = {
        "model": MODEL_VERSION,
        "keyed_on": gpu_name,
        "estimate_seconds": None,
        "r_card": None,
        "absent_because": None,
        # **Named on every row, because a bound read as an ETA is the misuse this model is one
        # rename away from.** `time-model.md` §3: the number a refusal consumes must be the
        # bound; the ETA shown to a person is a different quantity.
        "kind": "lower_bound",
        "excludes": "the load strip (import + prepare); this is model time, not wall time",
    }

    if still:
        # Stills are a separate code path and were excluded from the fit — one frame, a different
        # branch in the planner, and setup perfectly confounded with per-frame work.
        answer["absent_because"] = "stills are outside this model's fit"
        return answer
    if row is None:
        answer["absent_because"] = (
            "this card has never been measured; an unmeasured card produces no number rather "
            "than a borrowed one")
        return answer
    try:
        n = int(frames)
        mpx = float(output_pixels) / 1e6
    except (TypeError, ValueError):
        answer["absent_because"] = "frames or output_pixels was not a number"
        return answer
    if n < 1 or mpx <= 0:
        answer["absent_because"] = "frames or output_pixels was not positive"
        return answer

    answer["r_card"] = row["rate"]
    answer["n_clean"] = row["n_clean"]
    # **FLOORED to a tenth, never rounded to nearest.** `round()` can round UP, and an estimate
    # that rounds up is not a lower bound — which matters here far more than the magnitude
    # suggests, because `r_card` is fitted AT the clean minimum, so the runs the fit was taken
    # from land exactly ON the bound and have no margin to absorb a rounding step. Checked
    # against the 30 delivered records: with `round()` four of them violate the bound by 0.1%
    # or less, every one of them a fit minimum; with the floor, none does.
    #
    # This does not make the model conservative — it makes the arithmetic stop taking away the
    # one property the model promises.
    floored = math.floor(n * mpx * row["rate"] * 10.0) / 10.0
    if floored <= 0.0:
        # **Zero is not an answer this module is allowed to give.** Its own doctrine two
        # paragraphs up is that a number it cannot honestly produce is ABSENT — "not zero, not a
        # fallback" — and a floored `0.0` sitting beside a populated `r_card` and
        # `kind: lower_bound` reads as "instant" rather than as "smaller than the resolution
        # this is stated to". 2 frames of 320x240 on an H200 is 0.0625 s and floors to 0.0.
        answer["absent_because"] = (
            "the product floors below 0.1s, which this model cannot express as a bound; "
            "absent rather than reported as zero")
        answer["r_card"] = None
        return answer
    answer["estimate_seconds"] = floored
    # **Carried so a later reader can tell interpolation from extrapolation** without refitting.
    # Within a card at a fixed size the rate is extremely stable — four H200 8K runs span
    # 0.517-0.523 — and it rises systematically ACROSS sizes, so a job outside the measured span
    # is a different claim from one inside it.
    low, high = row["mpx_span"]
    answer["mpx_span"] = [low, high]
    answer["within_measured_span"] = bool(low <= mpx <= high)
    if row.get("single_measurement"):
        answer["single_measurement"] = True
    return answer
