"""The four §8b variants, composed from the shim rather than reimplementing it.

**RIFE is most accurate at its midpoint.** `t=0.5` is what it was trained on, and every variant
here is a different attempt to spend more of the budget near that value. That is the whole reason
more than one exists, and it is what the creative team is being asked to judge.

    direct   one arbitrary-timestep call per instant, t = frac(k * src/dst)
    cas      24->48 by midpoints, then ONE arbitrary pass to each 60 fps instant
    casdec   24->48->96 by midpoints, then SELECT the 60 nearest, synthesise nothing
    pull     no model called at all; each instant takes its nearest source frame

**The cascades are the shim run twice, not new arithmetic.** A 24->48 retime has ratio 0.5, so
every odd output lands at exactly `t=0.5` and every even one is a copy — the midpoint stage falls
out of the existing plan rather than being special-cased. Same for 48->96. `cas` then runs the
ordinary 48->60 plan over that stream; `casdec` selects instead. Nothing here re-derives a
position, which is why `retime_cases` still covers the arithmetic under all of them.
"""
from interpolate import build_plan, target_count

#: §8b's codes, in the order the contract lists them. `pull` last because it is the control.
VARIANTS = ("direct", "cas", "casdec", "pull")


def _select_nearest(frames, n_in, src_fps, dst_fps):
    """Yield, for each output instant, the source frame whose timestamp is nearest to it.

    **Nearest, not floor.** Flooring is what an interpolator does before synthesising between the
    pair; a pulldown has no second frame to reach toward, so it should pick the closer neighbour
    rather than always the earlier one. Flooring would bias every instant late by up to a whole
    source interval and make the control judder worse than the technique it stands in for.
    """
    n_out = target_count(n_in, src_fps, dst_fps)
    ratio = src_fps / dst_fps
    held = {}
    highest = -1
    source = iter(frames)
    counts = {"copy": 0, "hold": 0}

    for k in range(n_out):
        index = int(round(k * ratio))
        if index > n_in - 1:
            index = n_in - 1
            counts["hold"] += 1
        else:
            counts["copy"] += 1
        while highest < index:
            try:
                frame = next(source)
            except StopIteration:
                raise ValueError(
                    "the stream ended at frame {} but the selection needs {}".format(
                        highest + 1, index))
            highest += 1
            held = {highest: frame}
        yield held[index]

    _select_nearest.last_stats = {
        "n_out": n_out, "n_copy": counts["copy"], "n_synth": 0, "n_hold": counts["hold"],
        "real_frames": counts["copy"], "real_share": 1.0, "worst_snap_frac": 0.0,
    }


def run(variant, interpolator, frames, n_in, src_fps, dst_fps, tol=0.0):
    """Return `(frames_out, stats)` for one variant. `stats` carries how the work was spent.

    `stats` is the shape `RetimeResult.stats` has, plus `variant` and — on the cascades — the
    per-stage syntheses, because "382" and "192 + 384" are the same output frame count bought
    very differently and a ledger row that could not tell them apart would be unreadable.
    """
    if variant not in VARIANTS:
        raise ValueError("unknown variant {!r}; §8b lists {}".format(variant, list(VARIANTS)))

    if variant == "direct":
        result = interpolator.stream(frames, n_in, src_fps, dst_fps, tol=tol)
        return result.frames, dict(result.stats, variant="direct", stages=("direct",),
                                   synth_total=result.stats["n_synth"])

    if variant == "pull":
        # **The control, and it calls no model at all.** 100% real footage and it will judder
        # visibly — that is the point: it answers whether interpolation buys anything, and a
        # variant folder without it cannot. It also costs no GPU time, which makes skipping it
        # indefensible.
        stream = _select_nearest(frames, n_in, src_fps, dst_fps)
        n_out = target_count(n_in, src_fps, dst_fps)
        # Counted from the same arithmetic the selection uses rather than from `direct`'s plan.
        # Every delivered frame is a real one, so the share is 1.0 by construction; what varies
        # is how many instants fall beyond the last source frame and repeat it.
        picks = [int(round(k * src_fps / dst_fps)) for k in range(n_out)]
        held = sum(1 for index in picks if index > n_in - 1)
        stats = {"n_out": n_out, "n_copy": n_out - held, "n_synth": 0,
                 "n_hold": held, "real_frames": n_out, "real_share": 1.0,
                 "worst_snap_frac": 0.0, "variant": "pull", "stages": ("select",),
                 "synth_total": 0}
        return stream, stats

    # ── the cascades ─────────────────────────────────────────────────────────────────────────
    # **Stage one is a midpoint stage by arithmetic rather than by special case.** 24->48 has
    # ratio 0.5: every even output is a copy and every odd one lands at exactly t=0.5, which is
    # where RIFE is strongest. The stats come back before the generator is touched, so the next
    # stage's `n_in` is known without consuming anything.
    first = interpolator.stream(frames, n_in, src_fps, src_fps * 2, tol=0.0)

    if variant == "cas":
        # One arbitrary-timestep pass from the 48 fps stream to each 60 fps instant. Every
        # instant lands where the 60 fps grid says it should, and each non-midpoint call is taken
        # across a 48 fps interval — half `direct`'s temporal distance, so the same fractional
        # error is half the displacement.
        second = interpolator.stream(first.frames, first.stats["n_out"], src_fps * 2, dst_fps,
                                     tol=tol)
        stats = dict(second.stats, variant="cas", stages=("midpoint", "arbitrary"),
                     stage_synth=(first.stats["n_synth"], second.stats["n_synth"]),
                     synth_total=first.stats["n_synth"] + second.stats["n_synth"])
        return second.frames, stats

    # casdec: a second midpoint stage to 96, then SELECT — no arbitrary timestep anywhere.
    #
    # **Perfect midpoints, approximate timing, and that is the trade.** Every synthesis is t=0.5;
    # but 96 does not divide into 60, so the chosen instants are displaced by up to half a 96 fps
    # interval — 5.2 ms, a third of a 60 fps frame. Cadence judder bought with synthesis quality.
    # It costs more than `cas` for fewer exact frames, so if it does not visibly win it is
    # strictly worse — and that is a result.
    second = interpolator.stream(first.frames, first.stats["n_out"], src_fps * 2, src_fps * 4,
                                 tol=0.0)
    stream = _select_nearest(second.frames, second.stats["n_out"], src_fps * 4, dst_fps)
    n_out = target_count(n_in, src_fps, dst_fps)
    synth = first.stats["n_synth"] + second.stats["n_synth"]
    # **Counted, not borrowed from `direct`'s plan.** A first draft reported `direct`'s copy count
    # here, which is a plausible number about a different variant — `casdec` delivers whichever
    # 96 fps frames the selection lands on, and how many of THOSE are original is a property of
    # this variant's own arithmetic. An original survives the cascade at every 96 fps index that
    # is a multiple of four, because two midpoint stages put the source frames at 0, 4, 8, …; the
    # selection takes `round(k * 96/60)`, so a delivered frame is real exactly when that index is
    # divisible by four. That is exact and needs no measurement.
    picks = [int(round(k * (src_fps * 4) / dst_fps)) for k in range(n_out)]
    last_96 = second.stats["n_out"] - 1
    real = sum(1 for index in picks if index <= last_96 and index % 4 == 0)
    held = sum(1 for index in picks if index > last_96)
    stats = {
        "n_out": n_out,
        "n_copy": real,
        "n_synth": n_out - real - held,
        "n_hold": held,
        "real_frames": real,
        "real_share": real / n_out,
        "worst_snap_frac": 0.0, "variant": "casdec",
        "stages": ("midpoint", "midpoint", "select"),
        "stage_synth": (first.stats["n_synth"], second.stats["n_synth"]),
        "synth_total": synth,
    }
    return stream, stats
