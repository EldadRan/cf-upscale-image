"""The formula-hearted planner: geometry + budget + constants in, configuration out.

**One pure function decides every configuration this worker runs, and the same function
answers CF's routing question before a job is ever sent.** No GPU, no torch, no worker state,
no table lookup — six steps over registry v1.2's per-phase lines, each number traceable to a
registry entry by hand. That traceability is the point: a plan nobody can re-derive is a plan
nobody can audit, and the campaign that produced these constants exists precisely because the
previous heart could not explain itself.

**The decision law, stated once so no step re-derives it** (release-2-planner.md, ruled by CF
2026-08-18): *quality is the objective* — maximize the window, coarsen the grids, both
monotone; *memory is a hard constraint* — a phase price never exceeds usable; *time is a soft
constraint that exists only where it is super-linear* — the decode knee, and nowhere else;
*cost appears nowhere* — dollars are CF's, decided at routing time through this module's own
predicate, and a worker handed a card extracts the most quality that card holds.

**Standard library only, and deliberately.** CF embeds this file to answer "does this job fit
this endpoint" against its own inventory, so the module that plans inward and the module that
answers outward are the same module — the two can never disagree about what fits. It imports
`tiles` for the grid geometry rather than carrying a second copy of that arithmetic, because
two implementations of the same tiler is one that silently disagrees; both files are pure
stdlib and CF embeds the pair.

**`REGISTRY_VERSION` travels with every answer.** CF compares the version its embedded copy
reports against the one the worker reports and alerts the moment the two disagree (ratified
2026-08-18: the mismatch check is the intended use, not a nicety). A measurement is keyed to
the constants that planned it, and constants only change through a rebuild.
"""

import tiles

#: Bumped by the gate when `docs/gate/registry-v1.json` is re-keyed, and carried in every
#: manifest, ledger row, bundle and predicate answer. **A string, never a float** — this is an
#: identity compared for equality, and the repo's standing rule is that identity comparisons
#: never use floats. The registry spells it `1.2` as a JSON number; the equality that matters
#: is between two copies of this file, so the string is the authority.
REGISTRY_VERSION = "1.2"

# ---- registry v1.2 per-phase lines (GiB; MP = 1e6 px; w in frames) --------------------------
# Copied by hand from docs/gate/registry-v1.json. Every one of these is a fitted line with its
# residual band recorded beside it in the registry; none is a guess.

#: `vae_encode`: A + C_slab·enc_tile_MP + C_w·R_MP·w. Residuals ±0.46 across nine anchors.
#: **The encode plane is R, not the source.** The formula prices the tile against the model
#: plane because that is where the VAE's slab actually lives, and the arithmetic is decisive:
#: R3 ran a 2176 tile on a 3840x2160 output and measured 47.64, where the R-plane reading
#: predicts 47.31 (+0.33, inside the band) and a source-plane reading predicts 23.9.
ENC_A, ENC_SLAB, ENC_W = 1.14, 8.91, 0.00609

#: `dit_sample`: A_fixed + A_r·R_MP + C·R_MP·w at b=36. Residuals ±0.01 over five anchors and
#: three R classes — the tightest line in the registry, and the one that sets the window.
#: **Untileable**, which is why step 1 gives it the budget first: it is the quality lever and
#: it has no cheaper way to be bought.
DIT_A, DIT_R, DIT_W = 1.19, 0.486, 0.1245

#: `vae_decode`: A + C·dec_tile_MP. **Flat in w and in R** — verified w21..w85 and R 2.1..33.2
#: MP; only the tile moves it. Residuals ±0.35.
DEC_A, DEC_C = 0.94, 16.63

#: `postprocess`: A + C·O_MP·w. Confirmed at 8K to +0.03 on a 4x area extrapolation.
POST_A, POST_C = 0.03, 0.0969

#: **Held back from planning, and 1.0 rather than 2.0 by a ruled decision** (CF 2026-08-18,
#: quality first, production at the measured ceiling). Grounded in two direct measurements of
#: the true reserve rather than in caution: R3's 4K w85 DiT lived inside physical on 1.27 GiB,
#: E1b's 8K w29 on 1.03. The consequence is that the planner aims at the measured ceiling
#: everywhere — 4K w85, 8K w29 — both already measured delivering masters. The registry's
#: revisit clause: if production shows a pattern of DiT OOMs at those cells, this moves up on
#: that evidence, and every such OOM is a banked measurement of exactly the margin in question.
VRAM_RESERVE = 1.0

#: **RETIRED 2026-08-20, and it is the named defendant of F-2026-08-20-41.** It stood for
#: "everything that isn't frames" at 4.0 GiB while the measured constant was 18 to 29 — the whole
#: DiT lives in host RAM at `b=36` by design, plus CUDA host context. Kept as a name so nothing
#: silently re-imports it expecting the old arithmetic; `host_constant_gb` is what replaced it.
HOST_RESERVE_RETIRED = 4.0

#: **The temporal quality floor, and it is temporal** — below 21 frames the model stops beating
#: a plain enlargement *on video*. A still has no time axis to floor, which is why step 0
#: branches before this is ever consulted (ruled 2026-08-18).
MIN_WINDOW = 21

#: A decode grid that blends more than this of the frame is refused: past it the seams cost
#: more than the coarser grid saved.
MAX_BLEND_DECODE = 0.24

#: **The decode time knee, the one place time enters the decision.** Decode wall time is flat
#: below roughly 2 MP per tile and turns ~linear in tile area above it — measured (B1): 1024
#: and 1376 tiles decode 4K in the same ~465 s, while 2176 costs 2.34x. So the default is the
#: coarsest grid whose tile stays under the knee; everything above it is available, priced, and
#: only ever entered because the caller asked (`tile_quality="high"`).
KNEE_MP = 2.0

#: The tile overlap the vendored tiler is run with, on both planes.
LAP = 128

#: Host bytes per frame. `S·12` is the source fp32 stack; the output side carries the canvas at
#: 6 **plus the tail term at 5.3** — the post-phase-4 encode segment holds its own frame-scaled
#: memory, measured once at 8K/192 and consistent with one more canvas-sized copy alive during
#: the drain. **The tail term is not optional**: a tail RAM breach is a cgroup SIGKILL, which
#: writes no bundle, raises no exception and offers no walk, so plan-time pricing is the only
#: defense that exists. Provisional on one anchor until the `[host]` banners fit it properly.
#: Per-frame host bytes: the source stack, and the output canvas. **`O*12` is gone**
#: (ratified 2026-08-20). The registry carried `S*12 + O*12 + O*6` — the fp32 handoff copy plus
#: the bf16 canvas, 18 bytes a pixel — and `vendor_patch.py` deleted the widening that made the
#: copy three builds ago, in the 8K release. The formula never followed its own resolved note.
#: Measured against the instrumented corpus: 0.2734 GiB/frame at 8K and 0.0703 at 4K, both
#: within 2% of `S*12 + O*6`, on two cards 4x apart in output pixels.
HOST_SRC_BPP = 12.0
HOST_OUT_BPP = 6.0

#: **The constant that `HOST_RESERVE = 4.0` was standing in for**, and it is five to seven times
#: larger. `b=36` puts the whole 16.4 GiB DiT in host RAM by design; the rest is CUDA host
#: context, and how much of that a driver takes is a card-class fact — ~11 GiB more on the big
#: cards than on the A40. Carried as the *maximum* of each class: this is subtracted from the
#: budget, so over-stating it costs chunk size and under-stating it costs the container.
#:
#: An unknown card takes the larger figure. A card nobody has measured is not an argument for
#: optimism, and the failure mode on the wrong side of this number is a SIGKILL with no bundle.
HOST_CONSTANT_GIB = {"A40": 18.4}
HOST_CONSTANT_DEFAULT_GIB = 29.1

#: **The postprocess transient, which is what actually kills an 8K container** (ratified
#: 2026-08-20). `wavelet_reconstruction` builds a working set over the frames it holds at once
#: and releases it after; a boundary sample cannot see it, and `VmHWM` can.
#:
#: It is bounded, not linear in the window: the reconstruction is split along dim 0 because
#: `F.conv2d` indexes the *padded* tensor and 2^31 caps it — 84 frames at 4K, 21 at 8K. That
#: bound is why the 8K transient (26.68 GiB) is nowhere near four times the 4K one (23.27).
POST_TRANSIENT_BPP = 41.1
WAVELET_SPLIT_4K, WAVELET_SPLIT_8K = 84, 21
WAVELET_SPLIT_PIXELS = 16e6

#: A still's postprocess transient, flat. **The per-frame form does not describe it**: at `w=1`
#: it predicts 0.05 GiB against 14.62 measured, because a still's postprocess is not a windowed
#: reconstruction. This is the still-host price, and until 2026-08-20 it was zero — a still job
#: has no chunking lever, so there was nothing between "fits" and a SIGKILL with no bundle.
STILL_TRANSIENT_GIB = 14.62

#: **What the constants' own limits cost, priced where they were measured** (CF ruling 2b,
#: 2026-08-20, on the builder's flag). The refit as first implemented planned to 99.7–99.9% of
#: the slice — the ceiling policy's *form* without its earned substance.
#:
#: `VRAM_RESERVE = 1.0` earned plan-to-the-ceiling: two direct reserve measurements, ±0.46
#: residuals over nine anchors, and a breach lands in certified recovery — an OOM this worker
#: catches, diagnoses and re-plans. **A host breach is a cgroup SIGKILL: no exception, no bundle,
#: no walk.** The two are not comparable risks and must not carry comparable margins.
#:
#: Scaled to the transient rather than flat, because the transient is where the uncertainty
#: actually lives: 41.1 B/px·frame is the *maximum* of three points spanning 14%, its
#: `min(w, bound, frames)` form is chosen rather than proven, and it is the term that grows with
#: the geometry. The per-frame term has a mechanism behind it and agrees within 2%; the constant
#: is carried as a class maximum. Charging the doubt to the doubtful term keeps the margin
#: proportional to the exposure instead of taxing small jobs for a large job's uncertainty.
#:
#: **Retirement condition, so this cannot become permanent by inertia:** at ≥8 instrumented
#: transient points with the `min(w, bound, frames)` form validated across them, this may be
#: re-ruled toward the VRAM posture — by ratification, never by drift. Every diagnostic-pair
#: completion moves that count.
HOST_FIT_MARGIN_FRACTION = 0.15
HOST_FIT_MARGIN_FLOOR_GIB = 1.0

#: **The residency ladder's three answers** (amendment 9, CF ratified 2026-08-20 — Build D).
#: A three-state reply rather than a boolean, so CF's router can walk its tier list on it: a job
#: that fits nowhere on this card is a different fact from one that fits with the model
#: scheduled out of the way, and a predicate that answers "no" to both loses the cheaper tier.
RESIDENT, EVICTED, ROUTE_UP = "resident", "evicted", "route_up"

#: The checkpoint's own host copy, and **the only evictable term in the constant**. 16.4 GiB on
#: every card measured — registry `host.ratified_2026_08_20`, "checkpoint 16.4 on all", whose
#: own presence test is `load-end >= 16.4`. `b=36` puts it there by design (the DiT's swapped
#: blocks live on the offload device) and it is needed only while `dit_sample` runs.
#:
#: **Two readings of one decomposition, agreeing.** Subtracting it from each class constant
#: leaves 2.0 GiB on the A40 and 12.7 on an unmeasured card — and the registry line records the
#: same gap from the other side, "~11 GiB is CUDA host context". The residue is the driver's,
#: does not leave while the process lives, and is not evictable at any rung.
MODEL_RESIDENT_GIB = 16.4


def host_fit_margin_gb(output_pixels, window, frames, still=False):
    """What the chunk holds back for the transient's own uncertainty."""
    return max(HOST_FIT_MARGIN_FLOOR_GIB,
               HOST_FIT_MARGIN_FRACTION * postprocess_transient_gb(
                   output_pixels, window, frames, still=still))


def host_constant_gb(gpu_name=None, resident=True):
    """The materialised constant for this card class; `resident=False` drops the checkpoint.

    **Never negative.** A class constant below the checkpoint's own size would mean the card
    holds less than the thing it is holding, which is a table bug rather than a free budget —
    clamping at zero refuses to hand out memory that arithmetic invented.
    """
    for name, value in HOST_CONSTANT_GIB.items():
        if name in (gpu_name or ""):
            return value if resident else max(0.0, value - MODEL_RESIDENT_GIB)
    return (HOST_CONSTANT_DEFAULT_GIB if resident
            else max(0.0, HOST_CONSTANT_DEFAULT_GIB - MODEL_RESIDENT_GIB))


def wavelet_split_bound(output_pixels):
    """Frames the postprocess reconstruction can hold at once, from the 2^31 index limit."""
    return WAVELET_SPLIT_8K if output_pixels > WAVELET_SPLIT_PIXELS else WAVELET_SPLIT_4K


def postprocess_transient_gb(output_pixels, window, frames, still=False):
    """What postprocess adds above the plateau, and gives back."""
    if still:
        return STILL_TRANSIENT_GIB
    held = max(1, min(int(window or 1), wavelet_split_bound(output_pixels), int(frames or 1)))
    return POST_TRANSIENT_BPP * held * output_pixels / 1024.0 ** 3


def host_source_per_frame_gb(source_pixels):
    """The source stack alone — `S*12`, materialised before the model runs and resident after."""
    return source_pixels * HOST_SRC_BPP / 1024.0 ** 3


def host_canvas_per_frame_gb(output_pixels):
    """The output canvas alone — `O*6`, which is what a frame **still to be produced** adds.

    Split from the source term for F-2026-08-21-49: the plan prices a chunk's whole resting
    ramp, and a live projection starting from a container that already holds the source stack
    must add only what has not arrived yet. One function each, so neither caller can take the
    other's half by accident.
    """
    return output_pixels * HOST_OUT_BPP / 1024.0 ** 3


def host_per_frame_gb(source_pixels, output_pixels):
    return (host_source_per_frame_gb(source_pixels)
            + host_canvas_per_frame_gb(output_pixels))


def host_peak_gb(source_pixels, output_pixels, frames, window, still=False, gpu_name=None,
                 residency=RESIDENT):
    """**The three-term host model** (CF ratified 2026-08-20, F-2026-08-20-41's remedy).

        materialised constant + resident frames x per-frame + postprocess transient

    Priced against `memory.current`, not RSS: that is the number the OOM killer acts on, and the
    anon verdict says there is no reclaim cushion to discount — file cache measured ≤0.7 GiB at
    every boundary of every instrumented run.

    The old model was `(host_ram - 4.0) / per_frame` and it under-predicted about twofold, which
    is not a margin question: the chunk was *chosen from* that budget, so the error manufactured
    chunks that could not fit the slice they were sized for. Two workers died on it (F-41).

    **`residency=EVICTED` prices two phases, never their sum** (amendment 9). The model is home
    while `dit_sample` runs and gone while the peak phase runs, so the job's high-water mark is
    the larger of those two states — adding them would price a container that never exists, and
    subtracting the checkpoint outright would price one that does not either.
    """
    constant = host_constant_gb(gpu_name)
    ramp = max(0, int(frames or 0)) * host_per_frame_gb(source_pixels, output_pixels)
    transient = postprocess_transient_gb(output_pixels, window, frames, still=still)
    if residency != EVICTED:
        return constant + ramp + transient
    return max(constant + ramp,
               host_constant_gb(gpu_name, resident=False) + ramp + transient)


def host_phase_fits(source_pixels, output_pixels, frames, window, host_ram_gb,
                    still=False, gpu_name=None, residency=RESIDENT):
    """Does a chunk of `frames` clear **every phase** of the run at this rung?

    Split out from `host_chunk_cap` because the evicted rung has two constraints rather than one
    and both branches of the cap — the closed form and the walk-down — have to spend the same
    budget. Two branches spending different budgets is one branch that is wrong, and it would be
    wrong on exactly the jobs that reach here: the ones already tightest against the slice.

    The model-home phase carries **the margin floor rather than the scaled margin**: the margin
    is 0.15 of the postprocess transient, and in the phase where the model is still resident
    that transient has not been allocated. The floor is what exists for exactly this case — a
    phase whose own uncertainty is small still holds something back.
    """
    ramp = max(0, int(frames or 0)) * host_per_frame_gb(source_pixels, output_pixels)
    constant = host_constant_gb(gpu_name)
    transient = postprocess_transient_gb(output_pixels, window, frames, still=still)
    margin = host_fit_margin_gb(output_pixels, window, frames, still=still)
    if residency != EVICTED:
        return constant + ramp + transient + margin <= host_ram_gb
    return (constant + ramp + HOST_FIT_MARGIN_FLOOR_GIB <= host_ram_gb
            and host_constant_gb(gpu_name, resident=False) + ramp + transient + margin
            <= host_ram_gb)


def host_chunk_cap(source_pixels, output_pixels, frames, window, host_ram_gb,
                   still=False, gpu_name=None, residency=RESIDENT):
    """The largest chunk whose predicted peak fits the slice. `0` when even one frame does not.

    **Solved rather than divided**, because the transient is not linear in the chunk: it grows
    with the frames held until the wavelet bound, then stops. Dividing the budget by a per-frame
    figure — the shape of the model this replaces — either ignores the transient or charges its
    saturated value to a chunk too small to reach it.
    """
    if not host_ram_gb:
        return int(frames or 0)
    per_frame = host_per_frame_gb(source_pixels, output_pixels)
    if per_frame <= 0:
        return int(frames or 0)
    constant = host_constant_gb(gpu_name)
    # Saturated first: the common case, and the only one where the arithmetic is a division.
    saturated = postprocess_transient_gb(output_pixels, window, frames, still=still)
    margin = host_fit_margin_gb(output_pixels, window, frames, still=still)
    if residency == EVICTED:
        # **Two phases, and the chunk has to clear both** — see `host_phase_fits`. The model-home
        # phase has no transient and buys the bigger number; the evicted phase has no checkpoint
        # and buys the other. Taking the smaller is the whole of the ladder's honesty: rung 2
        # frees the model, it does not free the frames.
        with_model = int((host_ram_gb - constant - HOST_FIT_MARGIN_FLOOR_GIB) / per_frame)
        without_model = int((host_ram_gb - host_constant_gb(gpu_name, resident=False)
                             - saturated - margin) / per_frame)
        cap = min(with_model, without_model)
    else:
        cap = int((host_ram_gb - constant - saturated - margin) / per_frame)
    held = max(1, min(int(window or 1), wavelet_split_bound(output_pixels)))
    if cap < held:
        # Below the bound the transient shrinks with the chunk, so the budget reopens. Walk down
        # from the saturated answer rather than solving the quadratic — the range is one window.
        # **The margin rides this branch too.** Two branches spending different budgets is one
        # branch that is wrong, and it would be wrong on exactly the small-chunk jobs that reach
        # here — the ones already tightest against the slice.
        def _fits(n):
            return host_phase_fits(source_pixels, output_pixels, n, window, host_ram_gb,
                                   still=still, gpu_name=gpu_name, residency=residency)

        for candidate in range(max(0, min(cap, held)), 0, -1):
            if _fits(candidate):
                return min(int(frames or 0), candidate)
        for candidate in range(held, 0, -1):
            if _fits(candidate):
                return min(int(frames or 0), candidate)
        return 0
    return max(0, min(int(frames or 0), cap))

def residency_rung(source_pixels, output_pixels, frames, window, host_ram_gb,
                   still=False, gpu_name=None):
    """**The three-rung ladder** (amendment 9): `(rung, cap)`, decided before a GPU-second is spent.

        RESIDENT   the peak fits with the model home. Run exactly as today — no eviction, no
                   reload cost. The common case, and the amendment names it as such.
        EVICTED    it fits only if the model is scheduled out of the peak phase. The planner
                   commits to that now rather than discovering it at frame 50, and the chunk is
                   computed against the without-model peak — so rung 2 also buys larger chunks,
                   which is fewer hard cuts, which is quality.
        ROUTE_UP   it fits at neither. The remedy is the card, and saying so is what lets CF's
                   router walk its tier list instead of guessing.

    **Rung 1 wins whenever it fits, even where rung 2 would buy a bigger chunk**, because the
    amendment defines rung 2 as "fits *only* evicted". That is a deliberate reading of a ruled
    ladder and not an oversight — a note has gone to the gate that quality-first arguably wants
    the larger chunk at either rung, and it is theirs to rule, not this module's to assume.

    A blind host — no reading at all — stays on rung 1: eviction is a schedule, and scheduling
    against a number nobody has is how a plan invents its own budget.
    """
    if not host_ram_gb:
        return RESIDENT, None
    need = min(max(1, int(window or 1)), max(1, int(frames or 1)))
    resident_cap = host_chunk_cap(source_pixels, output_pixels, frames, window, host_ram_gb,
                                  still=still, gpu_name=gpu_name, residency=RESIDENT)
    if resident_cap >= need:
        return RESIDENT, resident_cap
    evicted_cap = host_chunk_cap(source_pixels, output_pixels, frames, window, host_ram_gb,
                                 still=still, gpu_name=gpu_name, residency=EVICTED)
    if evicted_cap >= need:
        return EVICTED, evicted_cap
    # **The larger of the two is reported**, because the refusal's job is to say how close the
    # card came. Reporting the resident cap on a job that was judged at the evicted one would
    # understate the shortfall by the whole checkpoint.
    return ROUTE_UP, max(resident_cap, evicted_cap)


#: The largest budget any constant was ever measured against (E1b, H200, 139.07 free). Past it
#: the lines are extrapolating rather than interpolating, and the predicate says so out loud
#: instead of pretending — today this is what flags a B200 until its one certification run.
ANCHORED_MAX_USABLE = 138.07

#: The coarsest decode tile any anchored budget could ever afford — `DEC_A + DEC_C·MP` at the
#: largest measured budget. Nothing above it is a rung anyone could choose, so bounding the
#: ladder by it costs no reachable configuration and makes the walk a function of the registry
#: rather than of the caller's requested target.
MAX_DECODABLE_TILE = int(((ANCHORED_MAX_USABLE - DEC_A) / DEC_C * 1e6) ** 0.5)

#: §5.1's amended gap rule. Fragmentation is suspected only when the reserved-but-unallocated
#: gap is **both** a real quantity and a real share of what was asked for. The absolute floor
#: is the half that was missing: B2 failed a 20 MiB allocation on a full card with a 25 MiB
#: gap, and the ratio alone called that fragmentation and bought a pointless retry.
FRAGMENTATION_GAP_SHARE = 0.25
FRAGMENTATION_GAP_FLOOR_GB = 1.0


def _lattice_floor(x):
    """The largest 4n+1 window at or below `x`, never below 1.

    The model's temporal lattice is 4n+1; a window off the lattice is silently rounded down by
    the sampler, so planning off it plans a number that will not be run.
    """
    return max(1, 4 * int((x - 1) // 4) + 1)


def _pad(x):
    """The smallest 4n+1 at or above `x` — what a pass of `x` frames actually costs the model."""
    return x if (x - 1) % 4 == 0 else 4 * ((x - 1) // 4 + 1) + 1


def ideal_window(frames):
    """The window a shot would get if memory were free: one pass over the whole clip."""
    return 1 if frames == 1 else _pad(frames)


def window_floor(frames):
    """The quality floor that applies to **this shot**: `MIN_WINDOW`, or the shot when shorter.

    **The floor is a statement about temporal context, and a shot cannot be refused for lacking
    context it never had.** `MIN_WINDOW` says that below 21 frames the model stops beating a
    plain enlargement *when 21 frames were available to attend to*; a ten-frame clip has no
    eleventh frame to give it, so a 13-frame window is that clip's ceiling and its floor at once.

    This is the same argument CF ratified for stills on 2026-08-18 — "MIN_WINDOW is temporal and
    does not apply to N = 1" — carried to its own boundary rather than stopped one frame short of
    it. The one-frame case remains its own branch because the contract writes it that way, but it
    falls out of this function identically.
    """
    return min(MIN_WINDOW, ideal_window(frames))


def _schedule_metrics(frames, window, overlap):
    """`(count, last_pass, padded_total)` for a schedule, **without building it.**

    Everything the sweep in `best_overlap` needs to rank a candidate is closed-form in
    `(frames, window, overlap)`, and computing it that way is the difference between a sweep
    that allocates thirty-odd lists and one that allocates none. At a million frames the old
    shape built ~33 lists of up to a million integers each to choose between them and then threw
    every one away — reachable from `plan_only`, so without spending a GPU-second
    (F-2026-08-18-15).

    The arithmetic is the loop's own, stated rather than walked: passes advance by
    `stride = window - overlap`, every pass with a full window ahead of it is `window` long, and
    what is left over is the tail. The tail is dropped when it is no longer than the overlap,
    because it would be entirely context the previous pass already saw.
    """
    if overlap >= window:
        overlap = 0
    if frames <= 0 or window <= 0:
        return 0, 0, 0
    stride = window - overlap
    if frames < window:
        return 1, frames, _pad(frames)
    full = (frames - window) // stride + 1
    tail = frames - full * stride
    if tail > 0 and not (full > 0 and tail <= overlap):
        return full + 1, tail, full * _pad(window) + _pad(tail)
    return full, window, full * _pad(window)


def _passes(frames, window, overlap):
    """The schedule itself. Built **once**, for the winner, after the sweep has chosen it.

    Kept as the loop rather than reconstructed from `_schedule_metrics`, because this is the
    definition the metrics are derived from and two spellings of one schedule is one that
    silently disagrees. `check_the_schedule_metrics_match_the_schedule` holds them together.
    """
    if overlap >= window:
        overlap = 0
    out, start = [], 0
    while start < frames:
        end = min(window, frames - start)
        if start > 0 and end <= overlap:
            break
        out.append(end)
        start += window - overlap
    return out


def best_overlap(span, window):
    """§2.5's simulation under **decision 8's default policy**: a short final pass is lived with.

    The cost surface here is jagged — one frame of overlap can remove a whole pass or add one —
    so every candidate is run and the cheapest wins, cheapest meaning the fewest frames the
    model is actually asked to attend to (each pass padded to the lattice), tie-broken toward a
    small non-zero overlap because a blended join is worth more than a hard cut.

    **The runt rejection is gone, and its removal is a ruling** (decision 8, CF 2026-08-18):
    `MIN_WINDOW` governs the *repeating* window of the walk, never the final pass. A schedule of
    29,29,29,5 is acceptable — the tail is what it is — while 15,15,15,15 is refused, because
    that is a main window below the floor, and step 1 refuses it long before this function sees
    it. Rejecting short tails here is also what manufactured F-15's pathology: with every sane
    candidate refused, the stride-1 degenerate schedule was the only one left standing and won.

    **The mitigation is a blend, not padding** ("no such thing" — CF). Where the cheapest
    schedule ends sub-floor on a hard cut, one or two frames of overlap fold that tail into its
    predecessor, and the few padded frames it costs are worth paying. The tie-break above
    already does this wherever the costs tie; the explicit step covers the shapes where a hard
    cut is strictly cheapest.

    **The sweep allocates nothing** (F-2026-08-18-15): count, last pass and padded total are
    closed-form in `(span, window, overlap)`, and only the winner is ever built.
    """
    best = None
    for overlap in range(0, min(33, window)):
        _count, _last, padded = _schedule_metrics(span, window, overlap)
        key = (padded, -min(overlap, 4))
        if best is None or key < best[0]:
            best = (key, overlap)
    overlap = 0 if best is None else best[1]

    count, last, _padded = _schedule_metrics(span, window, overlap)
    if count > 1 and last < MIN_WINDOW and overlap == 0:
        overlap = min((_schedule_metrics(span, window, v)[2], v) for v in (1, 2))[1]
    return overlap, _passes(span, window, overlap)


#: How much of a window a *balanced* schedule may spend on overlap. Past a quarter the schedule
#: re-attends more than it advances — the same threshold spirit as the 0.24 decode blend cap —
#: and without the bound the stride-1 degenerate schedule would qualify as "clean".
MAX_BALANCED_OVERLAP_SHARE = 4


def strict_schedule(span, window):
    """A schedule with **no pass below the floor**, or `None` when this window cannot give one.

    Balanced's admissibility test, and the strict no-short-tail rule the default policy gave up
    — kept alive here rather than deleted, because it is exactly the trade decision 8 says no
    formula may impose and the caller may choose.
    """
    best = None
    for overlap in range(0, min(window // MAX_BALANCED_OVERLAP_SHARE + 1, 33, window)):
        count, last, padded = _schedule_metrics(span, window, overlap)
        if not count:
            continue
        # Every pass is the window except the last, so the shortest pass *is* the last one —
        # and a single pass below the floor is not clean either.
        if last < MIN_WINDOW:
            continue
        key = (padded, -min(overlap, 4))
        if best is None or key < best[0]:
            best = (key, overlap)
    if best is None:
        return None
    overlap = best[1]
    return overlap, _passes(span, window, overlap)


def spans(frames, window, cap):
    """The spans that actually run under decision 7: the full chunk, and the tail chunk."""
    if cap is not None and cap < frames:
        chunk = (cap // window) * window
        return chunk, (frames % chunk if chunk else 0)
    return frames, 0


def balanced_window(frames, w_max, cap):
    """The largest lattice window at or below the card's whose **whole** schedule is clean.

    Addition 8b. "Whole" means chunk quantization and the tail chunk included — a window that
    tidies the body while leaving a five-frame final chunk has not balanced anything. Often the
    answer is `w_max` itself with more overlap, and the window steps down the lattice only when
    no clean schedule exists at the rung above.

    Returns `(window, (v, passes), tail_or_None)`, or `None` when nothing at or above the floor
    admits — in which case the default schedule stands, because a lever that cannot be honoured
    must not silently deliver something else.
    """
    window = w_max
    while window >= MIN_WINDOW:
        chunk, tail = spans(frames, window, cap)
        main = strict_schedule(chunk, window)
        if main is not None:
            if not tail:
                return window, main, None
            tail_sched = strict_schedule(tail, min(window, _pad(tail)))
            if tail_sched is not None:
                return window, main, tail_sched
        window = _lattice_floor(window - 1)
    return None


def _ladder(out_w, out_h, max_tile=None):
    """Every grid this plane can be cut into, coarsest first, each at its minimal tile.

    **A plane smaller than twice the overlap has no ladder at all**, and that is a real case
    rather than a degenerate one: a 128 px still against a 128 px tile overlap. `grid_ladder`
    correctly refuses to offer rungs there — below `2 x overlap` the stride collapses, three
    tiles can cover one pixel, and the blend arithmetic goes negative — so the honest answer is
    the single untiled rung, which is also the only sane way to process a plane that small.
    """
    ladder = tiles.grid_ladder(out_w, out_h, LAP, max_tile=max_tile)
    if ladder:
        return ladder
    # **Marked, because this rung is not on the ladder — it is the absence of one.** A real 1x1
    # rung carries a tile the vendored tiler was asked for and the oracle prices as such; this
    # one is synthesized for a plane too small to have rungs at all, and its "tile" is a number
    # invented here to make the arithmetic work. Reporting that number as a tile size is what
    # had a 128 px still running the tiler to cut itself into one piece (F-2026-08-18-28).
    rung = dict(tiles.tile_grid(out_w, out_h, max(out_w, out_h) + LAP, LAP))
    rung["synthesised_untiled"] = True
    return [rung]


def output_plane(src_w, src_h, target_short_edge):
    """The model plane R: the source scaled to the requested short edge, padded to a 16 grid.

    `DivisiblePad(16)` is the vendored transform, and it runs before anything is priced — so R
    is what every phase line is a function of, and a plane that "is" 1080p is 1088 to the model.
    """
    scale = target_short_edge / float(min(src_w, src_h))
    width, height = int(round(src_w * scale)), int(round(src_h * scale))
    return -(-width // 16) * 16, -(-height // 16) * 16


def encode_price(tile_mp, r_mp, window):
    return ENC_A + ENC_SLAB * tile_mp + ENC_W * r_mp * window


def dit_price(r_mp, window):
    return DIT_A + DIT_R * r_mp + DIT_W * r_mp * window


def decode_price(tile_mp):
    return DEC_A + DEC_C * tile_mp


def postprocess_price(r_mp, window):
    return POST_A + POST_C * r_mp * window


def window_for(usable, r_mp):
    """Step 1 inverted: the largest lattice window whose DiT price fits `usable`.

    This is the closed form the whole release turns on — `w = (usable − A − A_r·R) / (C·R)`,
    floored to the lattice. It is also the only place the window is decided; every other step
    takes it as given, and the correction path re-enters here with one constant changed.
    """
    return _lattice_floor((usable - DIT_A - DIT_R * r_mp) / (DIT_W * r_mp))


def _check_geometry(src, frames, target):
    """Refuse a shape no answer could be about. **Raised, not answered** (F-2026-08-18-20).

    `fits(frames=0)` used to return `fits: True, best_window: 1, host_chunk: 0`, and
    `frames=-5` a window of −3 with negative prices — a confident answer to a question nobody
    asked. The worker never saw it because `estimator` clamps on the way in; CF's embedded copy
    and `scripts/fits.py` call straight through, and this is shipped as *the* CF-side answer.

    An exception rather than `fits: False`, because a negative frame count is a bug in the
    caller and not a statement about capacity. Routing that stops with a message names the
    defect; routing that quietly skips an endpoint hides it in a latency graph a week later.
    """
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
        raise ValueError(
            "frames must be a positive integer — a still is 1 — got {!r}".format(frames))
    width, height = src
    if min(width, height) < 1 or min(target, 1) < 1:
        raise ValueError(
            "source {}x{} at target {} is not a plane anything can be planned for".format(
                width, height, target))


def _terminal(reason, **extra):
    """A refusal computed **before a GPU-second is spent**, carrying what the caller can do.

    §6: the two options are the caller's to choose between and neither is this module's to
    take. A sub-floor plan is never returned silently — that is the leak this shape closes.
    """
    body = {"action": "terminal", "reason": reason,
            "options": ["reduce target resolution", "flagged sub-floor run"],
            "registry_version": REGISTRY_VERSION}
    body.update(extra)
    return body


def plan(src, frames, target, vram_free_gb=None, host_ram_gb=None, usable_gb=None,
         gpu_name=None,
         tile_quality="default", schedule="max_window"):
    """The six steps, in the order quality demands. Returns a plan, or a terminal state.

    `src` is `(width, height)` of the source, `target` the requested short edge. Give either a
    card reading (`vram_free_gb`, from which the reserve is subtracted) or a budget already net
    of it (`usable_gb`, which is how CF asks about an endpoint it is not standing on).

    `tile_quality` is the caller's decode-seam lever: `"default"` honours the time knee,
    `"high"` waives it and buys the coarsest grid memory alone allows.

    `schedule` is the caller's tail lever (addition 8b). **What `"balanced"` promises is exactly
    one thing: no pass below decision 6's floor.** The mechanism is the planner's to choose,
    quality-first, and the prose here used to describe the wrong one — "steps the window down the
    lattice" — which was a plausible reading of `balanced_window`'s loop and not what it does
    (CF clarification, 2026-08-20, ratified from a live witness).

    What it actually does, and why that is the better mechanism: the window stays at the card's
    maximum and the **final pass's temporal overlap grows** until that pass reaches the floor —
    witnessed as `[85, 21] ov16` where `max_window` ran `[85, 6] ov1` on the same job. Stepping
    the window down would tax *every* frame's context to repair the tail; growing the last pass's
    overlap spends only on the pass that was short. The lattice step below remains as the
    fallback for when no overlap makes the schedule clean, which is why the loop exists at all.

    Corollary worth stating where a reader will meet it: **the bigger the card, the worse
    `max_window`'s runt on a short clip** — a larger `w_max` leaves a smaller remainder — so
    `balanced` matters most on the strongest hardware, which is the opposite of the intuition
    that a big card needs fewer levers.

    Neither is imposed — the default favours the body, balanced the tail, and which matters more
    is per-job judgment only the caller has.
    """
    _check_geometry(src, frames, target)
    usable = usable_gb if usable_gb is not None else max(0.0, (vram_free_gb or 0.0)
                                                         - VRAM_RESERVE)
    out_w, out_h = output_plane(src[0], src[1], target)
    r_mp = out_w * out_h / 1e6

    # ── step 1 — window, closed-form ────────────────────────────────────────────────────────
    # **Stills branch before the floor is consulted, not after.** `MIN_WINDOW` is a *temporal*
    # quality floor; a single image has no temporal dimension for it to floor, and applying it
    # anyway is what refused two perfectly plannable still jobs with a video's argument.
    if frames == 1:
        # **A still still has to fit.** The temporal floor does not apply to one frame, and for a
        # while nothing else did either: the branch set the window and walked straight past the
        # budget check that the video branch gets from its closed form. An 8192-px still prices
        # inside an A40 and a 12288-px one does not, and both were being planned.
        window = 1
        if dit_price(r_mp, 1) > usable:
            return _terminal(
                "a single frame at {} prices the sampler at {:.2f} GiB against {:.1f} usable — "
                "the window is already 1 and there is nothing left to give up".format(
                    "{}x{}".format(out_w, out_h), dit_price(r_mp, 1), usable),
                best_window=1)
    else:
        # **THE ASYMMETRY WITH THE STILL BRANCH ABOVE IS REAL AND LOAD-BEARING.** That branch
        # checks its own budget -- `if dit_price(r_mp, 1) > usable` -- because a still's window
        # is fixed at 1 and there is no closed form to trust. This branch has no such check: it
        # relies entirely on `window_for` returning a window whose sampler price fits, and on
        # `_pad(frames)` and the floor terminal below only ever lowering it from there.
        #
        # **Swept on 2026-08-28 and it held: four source geometries x seven frame counts x six
        # targets x seven budgets = 1,176 combinations, zero returned plans whose
        # `prices["dit_sample"]` exceeded their budget.** Written as a measurement rather than as
        # "this cannot happen", because the second is a claim someone has to re-establish and the
        # first is one they can re-run.
        #
        # **What would make it happen again**, so a later reader knows where to look rather than
        # re-deriving it: a `window_for` that returns anything but the largest FITTING window; a
        # lattice change that rounds the window UP after pricing; or any new path that reaches
        # the plan body while bypassing the floor terminal below. The last of those is not
        # hypothetical -- `allow_below_quality_floor` was exactly that path, and while it existed
        # this branch returned a plan at w1 priced 21.44 GiB against 18.0 usable. It was
        # withdrawn at 2f1b26b on CF's rule, and the hole went with it rather than being patched,
        # because a guard for a branch the tree cannot reach makes a bad state survivable instead
        # of impossible -- which this repository has already ruled against once, over the VAE
        # collision in `bake_weights.py`.
        window = min(window_for(usable, r_mp), _pad(frames))
        floor = window_floor(frames)
        if window < floor:
            return _terminal(
                "no window at or above the {}-frame quality floor fits {:.1f} GiB usable: the "
                "closed form gives {}, and below the floor the model stops beating a plain "
                "enlargement".format(floor, usable, max(0, window)),
                best_window=window)
    # **The host cap is computed before the window is final**, because the balanced lever may
    # step the window down and can only judge that against the spans it would actually produce —
    # chunk quantization and tail included. Step 5 consumes the same number further down.
    cap = None
    rung = RESIDENT
    if host_ram_gb:
        # **The three-term model, not a division** (ratified 2026-08-20). What stood here was
        # `(host_ram - HOST_RESERVE) / per_frame`, which charged 4.0 GiB for
        # everything-that-isn't-frames against a measured constant of 18 to 29 — and the chunk is
        # *chosen from* that budget, so the error did not shave a margin, it manufactured chunks
        # that could not fit the slice they were sized for.
        #
        # **And the ladder decides where the model lives while it is being spent** (amendment 9,
        # Build D). The cap that follows is the chosen rung's cap, so everything downstream —
        # quantization, the tail, the balanced lever — sizes against the container that will
        # actually exist rather than against one the job was never going to run in.
        rung, cap = residency_rung(src[0] * src[1], out_w * out_h, frames, window, host_ram_gb,
                                   still=(frames == 1), gpu_name=gpu_name)

    balanced = None
    if schedule == "balanced" and frames > 1 and _pad(frames) >= MIN_WINDOW:
        balanced = balanced_window(frames, window, cap)
        if balanced:
            window = balanced[0]

    dit = round(dit_price(r_mp, window), 2)

    # Both tiled phases cut the same plane R, so they walk one ladder. Coarsest first, each
    # rung carrying its minimal tile — within a grid the seam count is fixed and the tile is
    # pure cost, so the smallest tile reaching a grid is the cheapest way to buy it.
    ladder = _ladder(out_w, out_h)

    # ── step 2 — decode: the coarsest grid that fits, under the time knee ───────────────────
    decode = None
    for grid in ladder:
        tile_mp = grid["tile_pixels"] / 1e6
        price = decode_price(tile_mp)
        if (tile_quality == "high" or tile_mp <= KNEE_MP) and price <= usable \
                and grid["blended_fraction"] <= MAX_BLEND_DECODE:
            decode = (grid, round(price, 2))
            break
    if decode is None:
        return _terminal("no decode grid fits {:.1f} GiB usable under the {:.0%} blend floor"
                         .format(usable, MAX_BLEND_DECODE))

    # ── step 3 — encode: untiled whenever it fits, else the coarsest grid that does ─────────
    untiled = round(encode_price(r_mp, r_mp, window), 2)
    if untiled <= usable:
        encode = ("untiled", None, untiled)
    else:
        encode = None
        for grid in ladder:
            price = encode_price(grid["tile_pixels"] / 1e6, r_mp, window)
            if price <= usable:
                encode = ("{}x{}".format(grid["nx"], grid["ny"]), grid["tile"], round(price, 2))
                break
        if encode is None:
            return _terminal("no encode grid fits {:.1f} GiB usable".format(usable))

    # ── step 4 — postprocess: assert, never decide ──────────────────────────────────────────
    # It has no lever of its own — it is priced on the window step 1 already chose — so the
    # only thing to do here is check. An assert that fires is a *discovery*: the postprocess
    # line and the DiT line disagree about what the window costs, and that is a discrepancy
    # entry rather than a configuration to route around.
    post = round(postprocess_price(r_mp, window), 2)
    if post > usable:
        return _terminal(
            "postprocess prices {:.2f} GiB against {:.1f} usable at the window DiT accepted — "
            "the two lines disagree and that is a registry discrepancy, not a plan"
            .format(post, usable), postprocess_gb=post)

    # ── step 5 — chunk from the host model, including the tail term ─────────────────────────
    # **The chunk counts in windows, and the schedule is simulated over the span that runs**
    # (ruled decision 7, CF 2026-08-18 — F-2026-08-18-19). Two halves of one correction:
    #
    # *The chunk is quantized.* `chunk = ⌊cap/w⌋·w`, so every full chunk slices into exact
    # windows with zero overlap and no runt batch ever reaches the model. An unquantized chunk
    # sent whatever was left over as a final short batch — 6 to 10 frames on a host-limited
    # clip, far below the 21-frame temporal floor the whole release is built around.
    #
    # *The remainder is its own chunk.* `tail = N mod chunk`, run at `min(w, pad(tail))` —
    # decision 6's logic at the chunk boundary: the tail cannot be given context the RAM wall
    # has already taken.
    #
    # *And the schedule follows the chunk.* The vendored loop batches per chunk, so a pass list
    # computed over the clip describes a schedule that never runs — the runt guard enforced on
    # one span while another was executed. When the host holds the clip, the two spans are the
    # same and nothing here applies.
    if cap is not None and cap < min(window, frames):
        # **The rung travels with the refusal, and rung 3 names the card as its remedy.** The
        # sentence is unchanged — it was already the true thing to say, and a message the kit has
        # certified is not repriced by a feature — but a caller walking a tier list needs to know
        # whether a bigger *machine* is the answer or whether nothing on this shape fits at all.
        return _terminal(
            "host RAM allows a {}-frame chunk but the window is {} — the chunk can never be "
            "smaller than the window it has to hold".format(cap, window),
            chunk=cap, window=window, residency=rung,
            **({"remedy": "larger_host"} if rung == ROUTE_UP else {}))
    chunk, tail = spans(frames, window, cap) if frames > 1 else (frames, 0)
    if frames == 1:
        overlap, passes = 0, [1]
    elif balanced:
        overlap, passes = balanced[1]
    else:
        overlap, passes = best_overlap(chunk, window)

    prices = {"vae_encode": encode[2], "dit_sample": dit,
              "vae_decode": decode[1], "postprocess": post}
    binder = max(prices, key=prices.get)
    answer = {
        "action": "plan",
        "w": window, "chunk": chunk, "v": overlap, "passes": passes,
        "enc": encode[0], "enc_tile": encode[1],
        "dec": "{}x{}".format(decode[0]["nx"], decode[0]["ny"]),
        # **The synthesised rung reports no tile** (F-2026-08-18-28) — see `_ladder`. A genuine
        # 1x1 ladder rung keeps its tile, which is what the oracle prices and what the caller
        # asked for by choosing a coarse `tile_quality`.
        "dec_tile": None if decode[0].get("synthesised_untiled") else decode[0]["tile"],
        "dec_blend": round(decode[0]["blended_fraction"], 4),
        "prices": prices, "binder": binder,
        "usable": round(usable, 2), "out": "{}x{}".format(out_w, out_h),
        "ideal_window": ideal_window(frames),
        "tile_quality": tile_quality,
        "blocks_to_swap": 36,
        # **Always present, at every rung.** A field that appears only when something unusual
        # happened makes its absence ambiguous — "resident" and "this build has no ladder" would
        # read identically to a reader of an old record, and the corpus outlives the build.
        "residency": rung,
        "anchored": usable <= ANCHORED_MAX_USABLE,
        "registry_version": REGISTRY_VERSION,
    }
    # **Recorded only when it was honoured.** A lever that could not be satisfied — no window at
    # or above the floor gives a clean schedule — leaves the default standing, and saying
    # "balanced" over that would be reporting a policy the run did not follow.
    if balanced:
        answer["schedule"] = "balanced"
    # The tail chunk is reported as its own thing, because it *is* its own thing: a shorter span
    # at a narrower window, with its own schedule. A reader who sees only `chunk` and `passes`
    # would be looking at the full chunks and missing the one that is different.
    if tail:
        answer["chunks"] = frames // chunk + 1
        answer["tail_chunk"] = tail
        answer["tail_passes"] = (balanced[2][1] if balanced and balanced[2]
                                 else best_overlap(tail, min(window, _pad(tail)))[1])
    elif chunk < frames:
        answer["chunks"] = frames // chunk
    if rung == EVICTED:
        # **Evictions and reloads are different counts, and only the reloads cost seconds.** The
        # model is scheduled out once per chunk — including the last, whose postprocess needs the
        # room as much as any other — but it is only brought *back* for a chunk that follows one,
        # so a single-chunk job on rung 2 evicts once and reloads never. Conflating the two would
        # bill every rung-2 still for a re-materialisation that does not happen.
        answer["evictions"] = answer.get("chunks", 1)
        answer["reloads"] = max(0, answer.get("chunks", 1) - 1)
    return answer


def lowest_in_spec(src, frames, target, host_ram_gb=None, tile_quality="default",
                   gpu_name=None):
    """The cheapest configuration that is still **in spec**, for a card that cannot be read.

    §6's blind case. No VRAM reading means nothing can be claimed to fit, and the honest answer
    is neither "the floor rung" (a window of one — below the quality floor, unflagged, which is
    the leak this shape exists to close) nor a confident plan. It is the lowest-priced plan that
    still satisfies every floor: the window at `MIN_WINDOW`, and the *finest* grid on each plane
    that stays inside the blend cap — finest, because within spec the small tile is the cheap one
    and there is no budget here to spend on a coarse one.

    The answer is flagged `blind`, and the caller is expected to say so out loud: a plan chosen
    without a reading is a different kind of object from a plan chosen against one.
    """
    _check_geometry(src, frames, target)
    out_w, out_h = output_plane(src[0], src[1], target)
    r_mp = out_w * out_h / 1e6
    window = 1 if frames == 1 else min(MIN_WINDOW, _pad(frames))

    # **The feasibility guard `plan()` has, which this path did not** (F-2026-08-18-16). `plan()`
    # terminals on the closed form before it ever walks a ladder, so an absurd target costs it
    # nothing; this branch went straight to the geometry, whose sweep is linear in the plane
    # edge — a 1e9 target is ~1.25e8 iterations. The asymmetry between two entry points to the
    # same module was the defect, not the target itself: no maximum belongs in validation, which
    # is CF's ruled product choice, and none is invented here either. What is refused is a plane
    # no budget the constants were ever measured against could hold.
    if dit_price(r_mp, window) > ANCHORED_MAX_USABLE:
        return _terminal(
            "a {}x{} plane prices the sampler at {:.1f} GiB at the {}-frame floor, past the "
            "{:.1f} GiB the constants were measured against — and the card cannot be read, so "
            "nothing here can be claimed to fit".format(
                out_w, out_h, dit_price(r_mp, window), window, ANCHORED_MAX_USABLE))

    # And the ladder is bounded by the coarsest tile any anchored budget could decode, so its
    # walk is a function of the constants rather than of the caller's target.
    ladder = _ladder(out_w, out_h, max_tile=MAX_DECODABLE_TILE)
    in_spec = [g for g in ladder if g["blended_fraction"] <= MAX_BLEND_DECODE] or ladder[-1:]
    grid = in_spec[-1]
    tile_mp = grid["tile_pixels"] / 1e6

    prices = {
        "vae_encode": round(encode_price(tile_mp, r_mp, window), 2),
        "dit_sample": round(dit_price(r_mp, window), 2),
        "vae_decode": round(decode_price(tile_mp), 2),
        "postprocess": round(postprocess_price(r_mp, window), 2),
    }
    chunk = frames
    if host_ram_gb:
        chunk = max(window, host_chunk_cap(src[0] * src[1], out_w * out_h, frames, window,
                                           host_ram_gb, still=(frames == 1),
                                           gpu_name=gpu_name))
    overlap, passes = best_overlap(frames, window) if frames > 1 else (0, [1])
    name = "{}x{}".format(grid["nx"], grid["ny"])
    return {
        "action": "plan", "blind": True,
        "w": window, "chunk": chunk, "v": overlap, "passes": passes,
        "enc": name, "enc_tile": grid["tile"], "dec": name, "dec_tile": grid["tile"],
        "dec_blend": round(grid["blended_fraction"], 4),
        "prices": prices, "binder": max(prices, key=prices.get),
        "usable": None, "out": "{}x{}".format(out_w, out_h),
        "ideal_window": ideal_window(frames), "tile_quality": tile_quality,
        "blocks_to_swap": 36, "anchored": False,
        "registry_version": REGISTRY_VERSION,
    }


def rationale_line(answer):
    """The four priced lines and the binding phase, in one sentence a human can re-derive.

    Printed on every run and carried in the rationale so that CF, the gate, or anyone holding
    the registry can check the plan by hand — which is the whole argument for a formula-hearted
    planner over a table.
    """
    if answer.get("action") != "plan":
        return "terminal: {}".format(answer.get("reason"))
    prices = answer["prices"]
    return ("w{} · dit {:.2f}{} · enc {:.2f} {} · dec {:.2f} @{} ({:.1%} blended) · post {:.2f} "
            "· of {} usable{} · registry v{}".format(
                answer["w"], prices["dit_sample"],
                " (binder)" if answer["binder"] == "dit_sample" else "",
                prices["vae_encode"], answer["enc"],
                prices["vae_decode"], answer["dec"], answer.get("dec_blend") or 0.0,
                prices["postprocess"],
                "an unreadable card" if answer.get("usable") is None
                else "{:.2f} GiB".format(answer["usable"]),
                # **Silent on rung 1, and that is deliberate.** Rung 1 is "run exactly as today",
                # so today's sentence is what a rung-1 run should print — byte for byte, against
                # the kit's frozen expectations and against every rationale already in the
                # corpus. A clause that appears on every line would make the ladder's arrival
                # look like a change to jobs it did not change.
                ("" if answer.get("residency", RESIDENT) == RESIDENT else
                 " · model EVICTED for the peak phase ({} eviction(s), {} reload(s))".format(
                     answer.get("evictions", 1), answer.get("reloads", 0)))
                + (" · BLIND, lowest in-spec plan" if answer.get("blind") else ""),
                answer["registry_version"]))


def fits(src, frames, target, usable_gb, host_ram_gb=None, tile_quality="default",
         gpu_name=None):
    """**The caller-side predicate.** CF walks its endpoint list cheapest-first and stops at the
    first answer that clears the quality bar CF sets.

    A *predicate*, not a ladder, and the distinction is the contract: this says what a given
    budget can honestly do, and whether degraded-but-cheap beats full-quality-but-pricier for a
    given customer is CF's policy and never this module's. An A40 honestly "fits" a 4K job at
    w33 — that is a true answer, and only the caller knows what to do with it.

    `anchored: false` means the budget sits outside the span the constants were measured
    against, so the answer is an extrapolation wearing its label.
    """
    # **`gpu_name` reaches the plan, or the predicate prices a different card than the worker
    # will** (F-2026-08-20-48). It was accepted here, used by the `balanced_window` branch
    # below, and dropped on the way in — so `fits(..., gpu_name="NVIDIA A40")` answered with the
    # *default* host constant of 29.1 GiB while `plan(..., gpu_name="NVIDIA A40")` used the
    # measured 18.4. Measured effect on the low tier's own shape (1080p -> 4K, 600f, 46.57 GiB
    # slice): the predicate reported a 37-frame host chunk against the plan's 185. Five times
    # out, on the card every sub-4K measurement in this repo was taken on.
    answer = plan(src, frames, target, usable_gb=usable_gb, host_ram_gb=host_ram_gb,
                  tile_quality=tile_quality, gpu_name=gpu_name)
    ideal = ideal_window(frames)
    anchored = usable_gb <= ANCHORED_MAX_USABLE
    if answer["action"] != "plan":
        # **`fits` stays a boolean and `residency` carries the third state** (amendment 9). CF's
        # router already branches on `fits`; a field that changed type under it would break every
        # caller to express something a new field says without breaking anyone. `route_up` here
        # means the host slice cannot hold this job's peak even with the model scheduled away —
        # the remedy is a bigger machine, which is exactly what a tier walk needs to hear.
        return {"fits": False, "reason": answer["reason"], "ideal_window": ideal,
                "anchored": anchored, "residency": answer.get("residency", ROUTE_UP),
                "registry_version": REGISTRY_VERSION}
    reply = {"fits": True, "best_window": answer["w"], "ideal_window": ideal,
             "binding_phase": answer["binder"], "prices": answer["prices"],
             "host_chunk": answer["chunk"], "anchored": anchored,
             "residency": answer["residency"],
             "registry_version": REGISTRY_VERSION}
    if frames > 1:
        # **Both stories, so CF's router can advise before the job is sent** (addition 8b, CF:
        # "case by case a user can decide"). `shortest_pass` is what the default policy would
        # actually run — the number that says whether this clip has a weak tail at all — and
        # `balanced_window` is what the other lever would cost to remove it. A router with one
        # of the two can only guess at the trade the caller is being asked to make.
        reply["shortest_pass"] = min(answer["passes"] + (answer.get("tail_passes") or []))
        if reply["shortest_pass"] >= MIN_WINDOW:
            # Nothing to trade: the default already leaves no sub-floor pass.
            reply["balanced_window"] = answer["w"]
        else:
            out_w, out_h = output_plane(src[0], src[1], target)
            cap = None
            if host_ram_gb:
                # At the rung the plan actually chose: a rung-2 job asked about its rung-1 cap
                # would be told a tail is unfixable that the container it will run in can fix.
                cap = host_chunk_cap(src[0] * src[1], out_w * out_h, frames,
                                     answer.get("batch_size") or 1, host_ram_gb,
                                     still=(frames == 1), gpu_name=gpu_name,
                                     residency=answer["residency"])
            pick = balanced_window(frames, answer["w"], cap)
            # `None` is a real answer: no window at or above the floor cleans this shot up, so
            # the tail is a fact about the clip rather than a choice anyone is declining.
            reply["balanced_window"] = pick[0] if pick else None
    return reply


# ── correction: the same function, one constant updated ─────────────────────────────────────

def fragmentation_suspected(gap_gb, tried_gb):
    """§5.1 as amended 2026-08-17: **both** an absolute gap and a share of the request.

    The ratio alone has a degenerate case and B2 walked straight into it — a 20 MiB allocation
    failing on a full card with 25 MiB reserved-unallocated is a real limit wearing a
    fragmentation signature, and one pointless retry is what the ratio bought. Cost-only, never
    wrong output, which is why it survived this long.

    Exported as a function rather than left as a pair of constants so that everything asking
    this question — the live OOM path, the dry-run, and the acceptance kit — asks the one
    implementation instead of three copies of a rule that has already been amended once.
    """
    gap, tried = float(gap_gb or 0.0), float(tried_gb or 0.0)
    return gap >= FRAGMENTATION_GAP_FLOOR_GB and tried > 0 and \
        gap / tried >= FRAGMENTATION_GAP_SHARE


def classify_oom(shortfall):
    """`retry_same_config_fragmentation` or `confirmed` — and confirmed means immediately."""
    if fragmentation_suspected(shortfall.get("reserved_unallocated_gb"),
                               shortfall.get("failed_allocation_gb")):
        return "retry_same_config_fragmentation"
    return "confirmed"


#: Which single plan number a failing phase moves. **One lever, then the whole chunk from phase
#: 1** (ruled 2026-08-18): there is no mid-pipeline recovery, so what the failing phase decides
#: is only which number changed before the restart — never where execution resumes.
PHASE_LEVER = {"dit_sample": "window", "vae_decode": "decode_grid",
               "vae_encode": "encode_grid", "postprocess": "chunk", "host": "chunk"}


def correct(src, frames, target, usable_gb, phase, needed_gb, failed,
            host_ram_gb=None, tile_quality="default"):
    """Re-plan after a confirmed OOM: **the same six steps with one constant updated.**

    An OOM is a measurement. `needed_gb` is what the failing phase turned out to require at the
    configuration that failed, so the correction is not a guess at a safer rung — it is the one
    line the card just corrected, put back into the same arithmetic.

    **Exactly one plan number moves, and every other lever is held at what failed** — not
    re-planned. That distinction is the whole rule: a configuration that just OOMed in the
    sampler has told us nothing about its decode grid, and re-deriving the grid from the closed
    form would quietly move a second lever (often *upward*, since the failed run may have been
    deliberately coarser than the formula's own pick). Held levers are evidence; only the
    indicted one is arithmetic.

    Execution restarts the failed chunk from phase 1 under whatever comes back — completed
    chunks stay written, and there is no mid-pipeline resume.

    `failed` is the plan that died: `w`, `chunk`, `enc`/`enc_tile`, `dec`/`dec_tile`, `prices`.
    Returns a fresh plan carrying `lever`, or a terminal state when the lever is exhausted —
    never a sub-floor configuration, and never a second lever moved to rescue the first.
    """
    out_w, out_h = output_plane(src[0], src[1], target)
    r_mp = out_w * out_h / 1e6
    lever = PHASE_LEVER.get(phase, "window")
    ladder = _ladder(out_w, out_h)
    chunk = failed.get("chunk") or frames

    if lever == "window":
        # §5.3 rule 2: the DiT need is linear in the window, so the miss prices the correction
        # directly — `w ≤ lattice_floor(w_failed · usable/needed)`. No search, no ladder walk,
        # no margin rung: the measurement says where the window has to be, and the lattice says
        # which of those the sampler will actually run.
        window = _lattice_floor(failed["w"] * usable_gb / float(needed_gb))
        if window >= failed["w"]:
            # The arithmetic has to move: a "correction" that returns the window which just
            # failed is an infinite walk wearing a formula.
            window = _lattice_floor(failed["w"] - 1)
        if window >= failed["w"]:
            # **w = 1 has nothing below it, and this is where a still burned money**
            # (F-2026-08-18-17). `_lattice_floor(0)` is 1, so the step above handed back the
            # window that had just OOMed; the sub-floor terminal below was gated on `frames > 1`,
            # so a video was trapped by its own floor and a still fell through — into a retry
            # loop with no stop, on a billed worker, re-running an identical plan for ever.
            return _terminal(
                "the sampler needed {:.2f} GiB at w{} against {:.2f} usable, and w{} is the "
                "smallest window there is — the lever cannot move, so there is nothing to "
                "attempt".format(float(needed_gb), failed["w"], usable_gb, failed["w"]),
                window_cap=window, lever=lever)
        floor = window_floor(frames)
        if window < floor:
            # **`window_floor(frames)`, not `MIN_WINDOW`.** For a clip shorter than 21 frames the
            # floor is the clip (ratified by CF 2026-08-18), and quoting the constant here shipped
            # a number that was wrong for exactly the shots the ruling was about. The *outcome*
            # was already right under both readings — a short clip's initial window is its floor,
            # so any correction is sub-floor either way — which is why this only ever showed up
            # in the sentence.
            return _terminal(
                "the sampler needed {:.2f} GiB at w{} against {:.2f} usable, which caps the "
                "window at {} — below this shot's {}-frame floor, so there is nothing quieter "
                "to fall back to".format(float(needed_gb), failed["w"], usable_gb, window, floor),
                window_cap=window, lever=lever)
        return _held(failed, src, frames, target, usable_gb, r_mp, chunk, lever,
                     window=window, needed_gb=needed_gb)

    if lever == "chunk":
        # Host and postprocess move the chunk toward the window and **never below it**: the
        # window is a quality floor the chunk may approach and may not cross. A chunk that has
        # already reached the window has no lever left, and that is terminal rather than a
        # silent sub-floor run.
        shrunk = int(chunk * usable_gb / float(needed_gb)) if needed_gb else chunk - failed["w"]
        shrunk = min(shrunk, chunk - 1)
        if shrunk < failed["w"]:
            return _terminal(
                "the chunk would have to fall to {} frames to fit {:.2f} GiB, but the window is "
                "{} and a chunk cannot be smaller than the window it holds"
                .format(max(shrunk, 0), usable_gb, failed["w"]), lever=lever)
        return _held(failed, src, frames, target, usable_gb, r_mp, shrunk, lever,
                     needed_gb=needed_gb)

    # Both grid levers: **that plane's coefficient inflated once by needed/P₀**, then the next
    # grid down that fits under the inflated line. Inflated once and only once — the measurement
    # corrects this phase's coefficient for this card and this job, and compounding it across a
    # second failure would be modelling the OOM rather than the phase.
    side = "dec" if lever == "decode_grid" else "enc"
    priced = "vae_decode" if side == "dec" else "vae_encode"
    p0 = float((failed.get("prices") or {}).get(priced) or 0.0)
    inflation = max(1.0, float(needed_gb) / p0) if (p0 > 0 and needed_gb) else 1.0

    below = _grids_below(ladder, failed.get(side), out_w, out_h, failed.get(side + "_tile"))
    for grid in below:
        tile_mp = grid["tile_pixels"] / 1e6
        if side == "dec":
            if grid["blended_fraction"] > MAX_BLEND_DECODE:
                continue
            price = decode_price(tile_mp)
        else:
            price = encode_price(tile_mp, r_mp, failed["w"])
        if price * inflation <= usable_gb:
            return _held(failed, src, frames, target, usable_gb, r_mp, chunk, lever,
                         needed_gb=needed_gb, grid_side=side, grid=grid,
                         grid_price=round(price, 2))

    return _terminal(
        "the {} ladder is exhausted below {} at {:.2f} GiB usable — every remaining grid either "
        "misses the corrected budget or blends past the {:.0%} floor".format(
            lever.replace("_", " "), failed.get(side) or "untiled", usable_gb,
            MAX_BLEND_DECODE), lever=lever)


def _grids_below(ladder, current, out_w, out_h, current_tile):
    """The ladder rungs whose **working set** is strictly smaller than the one that failed.

    **On the working set, not on the tile edge** (F-2026-08-18-21). An untiled encode has no
    tile number, so the old reading admitted the whole ladder — including its 1x1 rung, whose
    tile is the entire plane. The correction for an untiled encode that OOMed at 79.34 GiB
    therefore answered with a 1x1 grid priced at 79.34 GiB: the identical working set, wearing
    a grid name, and one full billed re-run before the ladder moved at all.

    The failed grid is named by the *config that ran*, which may carry a tile the ladder never
    offered — a forced probe, or a rung whose minimal tile differs — so the comparison is on
    pixels the phase actually holds, which lands in the right place either way.
    """
    if current in (None, "untiled") and not current_tile:
        held = out_w * out_h
    else:
        tile = current_tile or max(out_w, out_h)
        held = min(tile, out_w) * min(tile, out_h)
    return [g for g in ladder if g["tile_pixels"] < held]


def _held(failed, src, frames, target, usable_gb, r_mp, chunk, lever,
          window=None, needed_gb=None, grid_side=None, grid=None, grid_price=None):
    """The failed plan with **one** number changed and everything else held where it was.

    Prices are recomputed rather than carried, because a price is a function of the levers and a
    plan whose numbers do not follow from its own configuration is exactly the half-updated
    rationale this repo has been bitten by before. What is *held* is the configuration; what is
    *derived* is every figure describing it.
    """
    window = failed["w"] if window is None else window
    enc, enc_tile = failed.get("enc") or "untiled", failed.get("enc_tile")
    dec, dec_tile = failed.get("dec"), failed.get("dec_tile")
    dec_blend = failed.get("dec_blend")

    if grid_side == "dec":
        dec, dec_tile = "{}x{}".format(grid["nx"], grid["ny"]), grid["tile"]
        dec_blend = round(grid["blended_fraction"], 4)
    elif grid_side == "enc":
        enc, enc_tile = "{}x{}".format(grid["nx"], grid["ny"]), grid["tile"]

    enc_mp = r_mp if enc_tile is None else (min(enc_tile, _plane(src, target)[0])
                                            * min(enc_tile, _plane(src, target)[1]) / 1e6)
    prices = {
        "vae_encode": grid_price if grid_side == "enc" else round(
            encode_price(enc_mp, r_mp, window), 2),
        "dit_sample": round(dit_price(r_mp, window), 2),
        "vae_decode": grid_price if grid_side == "dec" else round(
            decode_price(_tile_mp(dec_tile, src, target, r_mp)), 2),
        "postprocess": round(postprocess_price(r_mp, window), 2),
    }
    overlap, passes = best_overlap(frames, window) if frames > 1 else (0, [1])
    out_w, out_h = _plane(src, target)
    answer = {
        "action": "plan", "w": window, "chunk": max(chunk, window if frames > 1 else 1),
        "v": overlap, "passes": passes,
        "enc": enc, "enc_tile": enc_tile, "dec": dec, "dec_tile": dec_tile,
        "dec_blend": dec_blend,
        "prices": prices, "binder": max(prices, key=prices.get),
        "usable": round(usable_gb, 2), "out": "{}x{}".format(out_w, out_h),
        "ideal_window": ideal_window(frames),
        "tile_quality": failed.get("tile_quality", "default"),
        "blocks_to_swap": 36,
        "anchored": usable_gb <= ANCHORED_MAX_USABLE,
        "registry_version": REGISTRY_VERSION,
        "lever": lever,
    }
    if needed_gb:
        answer["corrected_from_gb"] = round(float(needed_gb), 2)
    return answer


def _plane(src, target):
    return output_plane(src[0], src[1], target)


def _tile_mp(tile, src, target, r_mp):
    """A tile's working set on the model plane, or the whole plane when nothing is tiled."""
    if not tile:
        return r_mp
    out_w, out_h = _plane(src, target)
    return min(tile, out_w) * min(tile, out_h) / 1e6
