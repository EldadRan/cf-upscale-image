"""Route C's fit predicate and time model — the FORM, and an honest refusal to quote.

Contract §6. **Every guarantee the planner makes is about model memory** — `fits`, the rung
ladder, `host_chunk_cap`, the eviction schedule — so none of them answers anything about a job
with no model, and none may be silently reused. Route C needs its own, and this is it.

**No coefficient exists yet and none is invented here.** §6 fixes the form so Phase 2 measures
the right things; the numbers are Phase 2's output. A placeholder would be worse than a refusal
because it is indistinguishable from a measurement — the same rule as `snap_tolerance`'s absent
default and the omitted `cf_model_build` tag. Three places in this release now where the honest
shape of an unknown is nothing at all.

    interpolate:
      formula     A + C_pair * pad_MP            at one (scale, precision)
      pad_rule    each dimension up to a multiple of max(128, 128/scale), cropped to [:h, :w]
      plane       the SOURCE plane under route A and C, the OUTPUT plane under route B
      w_scaling   FLAT
      scale_note  --scale reduces flow-pyramid resolution; expected to move C, not A
      precision   fp16 against fp32 is a separate line or a ruled factor, never assumed

**`w_scaling: FLAT` is the structural claim and it is what makes route C cheap.** RIFE holds
exactly one frame pair whatever the clip length: no window axis, no chunk axis, no batch axis.
Every SeedVR2 phase in the registry has at least one, and the planner's whole ladder exists to
trade against them. A phase whose peak does not move with `w` needs none of that machinery, and
the fit predicate is a single comparison rather than a search.

**If the measurement shows peak moving with clip length, this claim is wrong and the design above
it changes**, so it is the first thing Phase 2 should try to falsify.
"""
from interpolate import pad_multiple

#: The registry line this route would read, once it has one. Named here so the absence is a
#: lookup that fails rather than a silence: `registry-v1.json` models the four SeedVR2 phases and
#: nothing else, and **a route-C job cannot be quoted from the current registry.**
REGISTRY_PHASE = "interpolate"

#: What the benchmark must report beyond §8's peak VRAM: **at least three distinct padded areas**
#: per (scale, precision) pair. One number per variant fits nothing — a line needs three points to
#: have a residual, which is what every other registry phase carries and what makes a prediction
#: quotable rather than a guess. The 4K reading alone gives one point.
MINIMUM_FIT_POINTS = 3


class Unquotable(Exception):
    """Raised where a number would be invented. Carries what is missing and who owns it."""


def padded_megapixels(width, height, scale=1):
    """The area the formula is a function of, in megapixels, after §9's padding rule.

    **This much is arithmetic and is knowable today** — it is the fit's independent variable, and
    it is computable without a single measurement. Reported so Phase 2's readings can be plotted
    against the same quantity the predicate will use, rather than against raw frame size.
    """
    if int(width) <= 0 or int(height) <= 0:
        # **The one function here that could invent a number, so it does not.** Negative
        # dimensions produced a positive, entirely plausible megapixel figure — and a negative
        # one flowed into the formula and made `fits` return True for an impossible job, while
        # zero made it `A <= usable` at any resolution. In a module whose whole thesis is that a
        # placeholder is worse than a refusal because it is indistinguishable from a measurement,
        # this was the placeholder.
        raise ValueError("padded area needs positive dimensions, got {}x{}".format(width, height))
    multiple = pad_multiple(scale)
    padded_w = -(-int(width) // multiple) * multiple
    padded_h = -(-int(height) // multiple) * multiple
    return (padded_w * padded_h) / 1_000_000.0


def peak_vram_gb(width, height, scale=1, precision="fp32", registry=None):
    """`A + C_pair * pad_MP`, when A and C_pair exist. They do not.

    Refuses rather than guessing. A caller that wants to know whether route C fits a card today
    has to measure it; that is Phase 2's job and the refusal names it.
    """
    line = (registry or {}).get(REGISTRY_PHASE)
    if not line or "A" not in line or "C_pair" not in line:
        raise Unquotable(
            "route C has no calibrated line in the registry: '{}' is unmeasured, so peak VRAM "
            "for {}x{} at scale {} in {} cannot be quoted. registry-v1.json models the four "
            "SeedVR2 phases and nothing else. Phase 2 measures it, at {} or more distinct padded "
            "areas per (scale, precision) pair.".format(
                REGISTRY_PHASE, width, height, scale, precision, MINIMUM_FIT_POINTS))
    return line["A"] + line["C_pair"] * padded_megapixels(width, height, scale)


def fits(width, height, usable_vram_gb, scale=1, precision="fp32", registry=None):
    """**A single comparison rather than a search**, because the peak does not move with `w`.

    The rung ladder exists to trade a window, a chunk or a batch against memory. Route C has none
    of those axes, so there is nothing to step down to: either one padded pair fits the card or
    the job does not run. That is the whole predicate, and it is why route C is cheap to plan.
    """
    return peak_vram_gb(width, height, scale, precision, registry) <= usable_vram_gb


def seconds(stats, per_synthesis_s=None):
    """**Driven by SYNTHESES, not by output frames** — and `n_synth` is already in the plan.

    A copied frame costs essentially nothing; a synthesised one costs a model pass. Under §5's
    arithmetic the difference is not marginal: 20->60 synthesises 398 of 600 frames while
    23.976->60 synthesises 5,996 of 6,001. **Same output frame count, half again the work** —
    quoting on delivered frames would misprice by that much.

    `per_synthesis_s` is Phase 2's, measured per synthesis at a padded area rather than per
    delivered frame. The handoff's ~30-60 interpolated fps at 1080p is the vendor's shape and not
    ours: unquotable until measured here.
    """
    if per_synthesis_s is None:
        raise Unquotable(
            "route C's time model is n_synth x t(pad_MP) and t is unmeasured. This plan holds "
            "{} syntheses out of {} output frames, which is the multiplier the estimate needs — "
            "the seconds are Phase 2's.".format(
                (stats or {}).get("n_synth"), (stats or {}).get("n_out")))
    return stats["n_synth"] * per_synthesis_s
