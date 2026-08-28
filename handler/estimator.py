"""Deciding whether the work fits, and what to spend to make it fit.

**This is the primary defence; the retry is only the backstop.** An OOM after the spend is the
worst outcome available — RunPod has billed the seconds, CF passes them to the customer, and the
customer receives nothing. So the standing rule is: where a choice is between a slower job and a
job closer to an OOM, take the slower job, every time, without asking.

There is no resolution guard rail and CF does not want one. A cap refuses work that would have
succeeded, which is a failure this model row has already produced once. What replaces it is this:
decide per job whether the work fits, and shape it so it does.

## Calibrated from measurement, or not at all

CF retired a megapixel cost model on this model because **tiling moves execution time by more
than any pixel formula predicts**. So this module does not contain a formula for memory or for
time. It contains a ladder of configurations and a table of what each one actually cost, and the
table is filled by running jobs.

**The table ships empty, and an empty table means the floor.** That is the honest degradation:
with nothing measured, the estimator cannot claim a faster rung fits, so it takes the slowest and
safest one and says `calibrated: false` in its report. The alternative — inventing a formula to
fill the gap — is precisely what CF retired, and it would fail in the direction that costs a paid
job its result.

## The ladder

Rungs run fastest to safest. Each is a complete configuration, so "step to a materially more
conservative configuration" is one index rather than a negotiation between five knobs.

The levers, and why they are ordered this way (`docs/decisions.md` 0.1):

  `chunk_size` first, because upstream's default of 0 means *load every frame at once* and that
  default is the OOM. Bounding frames resident is the lever with the largest effect on a long
  clip and the smallest effect on quality — chunks are processed with temporal overlap, so the
  cost is compute at the seams rather than a worse picture.

  `vae_*_tiled` next: encode and decode tile independently, which is why "did it tile" is two
  booleans rather than one.

  `blocks_to_swap` next — 0 to 36 on the 7B. This is the lever the handoff reached for and could
  not name: CPU offload as a *dial* rather than the all-or-nothing switch the image worker uses.
  It converts VRAM into time continuously, which is exactly the trade this worker is told to make.

  whole-model offload last, because it is the bluntest and slowest.
"""

import json
import os
import re

import solver
from errors import (CAPACITY_EXCEEDED, DEADLINE_EXCEEDED, INVALID_FIELD_VALUE,
                    Remedy, WorkerError)
# The 4n+1 lattice the model requires of `batch_size`, imported rather than restated: a second copy
# of a constraint is a copy that drifts. `pipeline`'s module scope is light — os, sys and errors —
# so this does not drag torch or cv2 into a planner that CI runs without them.
from pipeline import BATCH_SIZES
import planner

CALIBRATION_PATH = os.environ.get(
    "CALIBRATION_PATH", os.path.join(os.path.dirname(__file__), "calibration.json")
)

#: VRAM held back from planning: allocator fragmentation, the CUDA context, cuDNN workspaces.
#: **One definition, in `planner`.** It moved from 2.0 to 1.0 on two direct measurements of the
#: true reserve (R3 1.27 GiB, E1b 1.03), and three modules each holding their own copy of that
#: number is three places to forget when the next measurement moves it again.
VRAM_RESERVE_GB = planner.VRAM_RESERVE

#: Fastest to safest. `name` is what the response and the manifest report, so CF can see which
#: rung a job ran at without reading this file.
RUNGS = (
    {
        "name": "fast",
        "chunk_size": 97, "batch_size": 21, "temporal_overlap": 2,
        "vae_encode_tiled": False, "vae_decode_tiled": False,
        "vae_encode_tile_size": 1024, "vae_decode_tile_size": 1024,
        "vae_encode_tile_overlap": 128, "vae_decode_tile_overlap": 128,
        "blocks_to_swap": 0, "swap_io_components": False,
        "dit_offload_device": "none", "vae_offload_device": "none",
        "tensor_offload_device": "cpu",
    },
    {
        # **`fast` with a chunk that fits, and nothing else changed.** The ladder's other rungs
        # move chunk size and block-swapping together, which conflates two different trades:
        # the chunk *bounds* memory, while swapping *converts* memory into time. Video wants the
        # first without the second, and until this rung existed there was no way to ask for it.
        #
        # The consequence was concrete. A 1500-frame clip asks `fast` for a 97-frame chunk —
        # about 210 GB at 1080p by the measured per-frame slope — so it fell all the way to the
        # floor and took 6.4 hours where this configuration projects to roughly 86 minutes.
        #
        # `chunk_size` is 10 rather than a rounder number because **10 is what was measured**:
        # the ten-frame clip that ran at `fast` produced exactly one chunk of ten, so its
        # 34.89 GB peak describes this rung directly rather than by extrapolation. Every other
        # value here is `fast`'s, unchanged, for the same reason — a knob that differs from the
        # measured run is a knob the measurement does not cover.
        "name": "resident",
        "chunk_size": 10, "batch_size": 21, "temporal_overlap": 2,
        "vae_encode_tiled": False, "vae_decode_tiled": False,
        "vae_encode_tile_size": 1024, "vae_decode_tile_size": 1024,
        "vae_encode_tile_overlap": 128, "vae_decode_tile_overlap": 128,
        "blocks_to_swap": 0, "swap_io_components": False,
        "dit_offload_device": "none", "vae_offload_device": "none",
        "tensor_offload_device": "cpu",
    },
    {
        "name": "balanced",
        "chunk_size": 49, "batch_size": 9, "temporal_overlap": 2,
        "vae_encode_tiled": False, "vae_decode_tiled": True,
        "vae_encode_tile_size": 1024, "vae_decode_tile_size": 1024,
        "vae_encode_tile_overlap": 128, "vae_decode_tile_overlap": 128,
        "blocks_to_swap": 0, "swap_io_components": False,
        "dit_offload_device": "cpu", "vae_offload_device": "cpu",
        "tensor_offload_device": "cpu",
    },
    {
        "name": "tiled",
        "chunk_size": 25, "batch_size": 5, "temporal_overlap": 2,
        "vae_encode_tiled": True, "vae_decode_tiled": True,
        "vae_encode_tile_size": 768, "vae_decode_tile_size": 768,
        "vae_encode_tile_overlap": 128, "vae_decode_tile_overlap": 128,
        "blocks_to_swap": 12, "swap_io_components": False,
        "dit_offload_device": "cpu", "vae_offload_device": "cpu",
        "tensor_offload_device": "cpu",
    },
    {
        "name": "swapped",
        "chunk_size": 9, "batch_size": 5, "temporal_overlap": 1,
        "vae_encode_tiled": True, "vae_decode_tiled": True,
        "vae_encode_tile_size": 512, "vae_decode_tile_size": 512,
        "vae_encode_tile_overlap": 128, "vae_decode_tile_overlap": 128,
        "blocks_to_swap": 24, "swap_io_components": True,
        "dit_offload_device": "cpu", "vae_offload_device": "cpu",
        "tensor_offload_device": "cpu",
    },
    {
        # The floor. Everything this worker can spend to stay off an OOM is spent here, so a
        # failure at this rung is the only honest basis for "a larger card would work" — and CF
        # asks for the shortfall measured at *this* configuration, not at the one that happened
        # to fail. A figure from a faster rung overstates the card CF would have to buy, and
        # that is a purchasing decision, so the direction of the error matters.
        "name": "floor",
        "chunk_size": 5, "batch_size": 1, "temporal_overlap": 0,
        "vae_encode_tiled": True, "vae_decode_tiled": True,
        "vae_encode_tile_size": 384, "vae_decode_tile_size": 384,
        "vae_encode_tile_overlap": 128, "vae_decode_tile_overlap": 128,
        "blocks_to_swap": 36, "swap_io_components": True,
        "dit_offload_device": "cpu", "vae_offload_device": "cpu",
        "tensor_offload_device": "cpu",
    },
)

FLOOR_INDEX = len(RUNGS) - 1
RUNG_NAMES = tuple(r["name"] for r in RUNGS)


def load_calibration(path=None):
    """Measured runs. Absent or unreadable is a normal state, not an error — it is the state
    this repo ships in, and it means 'take the floor' rather than 'guess'."""
    path = path or CALIBRATION_PATH
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    return data.get("runs", []) if isinstance(data, dict) else (data or [])


def short_edge_covering(source_width, source_height, target_width, target_height):
    """The short-edge target whose output covers `target_width` x `target_height` in both axes.

    **So the exact size is only ever reached by shrinking.** Asking the model for the requested
    short edge is not enough: the long edge follows the *source's* aspect, and where the caller's
    aspect is wider, the derived long edge lands short and the final fit would have to enlarge it
    — inventing pixels after the model has finished, which is the one thing the caller paid a GPU
    to avoid. Scaling by the larger of the two ratios covers both, and the fit is a downscale of
    at most the aspect difference.
    """
    scale = max(target_width / float(source_width), target_height / float(source_height))
    return max(1, int(round(min(source_width, source_height) * scale)))


def output_dimensions(source_width, source_height, target_short_edge_px):
    """`target_short_edge_px` names the output's SHORT edge — not a scale, not a long edge.

    It fixes one dimension; the other follows from the source's aspect. Verified by CF on stills
    across both orientations (749×500 and 500×750 both scaled 3.84× at a target of 1920) and
    stated in the vendored source's own help text. Both orientations are handled by the same two
    lines here, which is the property worth keeping rather than special-casing.

    A target below the source's short edge is a downscale. CF permits it and warns, so it is
    computed exactly the same way and never refused.
    """
    short = min(source_width, source_height)
    scale = target_short_edge_px / float(short)
    width = int(round(source_width * scale))
    height = int(round(source_height * scale))
    # The model pads to a multiple of 32 internally and crops back, so this is about the
    # container rather than the model: an odd dimension is not encodable as yuv420p.
    return width - (width % 2), height - (height % 2)


#: What a row's tiling is when the row does not say. **Every one of the 39 rows in the shipped
#: table is absent here, and every one of them was planned at `default`**: `tile_quality` is a
#: release-2 lever, the first job ever to ask for `high` was the 8K hero shot of 2026-08-20, and
#: that job's master never landed — so it banked no row (F-2026-08-20-39). Reading absent as
#: `default` is therefore a statement about this table's history rather than a guess, it is
#: asserted as such by rung 1, and it stops being load-bearing the moment rows start carrying the
#: field, which `run_one` now banks.
#:
#: The alternative — absent means unknown, so degrade — would relabel every prediction the table
#: has ever made, including the 8K row a certified verdict already reads as measured. A rule that
#: changes certified output in order to describe rows it has no doubt about is not caution.
DEFAULT_TILE_QUALITY = "default"


def _row_tile_quality(row):
    return row.get("tile_quality") or DEFAULT_TILE_QUALITY


def _timing_rows(calibration, output_pixels, window, unbatched=None):
    """Rows to borrow a per-frame time from when the configuration has no rung name.

    **Time and memory are different questions and this is where they part.** A borrowed peak is an
    OOM; a borrowed rate is a wider ETA, which the deadline margin already covers. So this is
    permissive where `_matching_runs` is strict: any row within 2x on pixels, preferring those at a
    comparable window, and the caller still prefers rows measured on the same card.
    """
    # **Selected on pixels alone, deliberately.** An earlier version preferred rows at a
    # comparable window, which made the rate depend on the plan: two jobs 12.5% apart in size
    # picked different window bands and their predictions diverged 3.2x. The per-frame rate is
    # then scaled by the pixel ratio by the caller, which is the correction that belongs here;
    # the window's effect on speed is real but second-order beside it, and folding it in through
    # row selection made the prediction unstable rather than sharper.
    del window
    rows = [r for r in calibration
            if r.get("seconds_per_frame") and r.get("output_pixels")
            and 0.5 <= r["output_pixels"] / float(output_pixels) <= 2.0]

    # **A row that ran without temporal batching may never price one that will, or the reverse**
    # (F-2026-08-20-40, third face). The reason is one sentence: a window of 1 pays the whole
    # per-frame setup on every frame, where a window of N amortises it across N. That is a
    # different quantity, not a slower version of the same one.
    #
    # **This subsumes the still/video split and is why it replaced it.** The rule was `frames ==
    # 1`, which caught all sixteen still rows — every one of them runs at window 1 — and missed
    # the three *video* rows that also ran unbatched. One of those is an A40 48-frame 4K row at
    # `rung: floor`, `window: 1`, **34.438 s/frame**, sitting beside honest rows of 12.8, 13.8,
    # 14.5 and 16.4 for the same card, size and day. Because the caller takes `max()` over
    # comparable rows, that single row priced every A40 4K video job at roughly 2.4x its measured
    # rate, wearing `prediction_basis: measured`.
    #
    # Keyed on `window` and not on the rung name: `floor` is how this configuration is usually
    # reached, but the thing that costs is the window, and a forced `batch_size=1` at any rung
    # costs exactly the same.
    #
    # Selection still does not depend on the window's *value* — that was tried and removed,
    # because two jobs 12.5% apart in size picked different bands and their predictions diverged
    # 3.2x. This is a partition into batched and unbatched, which is a difference in kind.
    if unbatched is None:
        return rows
    return [r for r in rows if (r.get("window") == 1) == bool(unbatched)]


#: RunPod's own hard ceiling on `executionTimeout`, in milliseconds — seven days. Not this
#: worker's number and not a policy: past it the platform kills the container whatever anyone
#: intended, so a job that cannot finish inside it cannot finish at all. `validation.py` already
#: refuses an `execution_timeout_ms` above it for the same reason.
PLATFORM_EXECUTION_CEILING_MS = 604_800_000


def fastest_seconds_per_frame(calibration, output_pixels):
    """The quickest per-frame rate ever measured here, scaled to this size. `None` if unmeasured.

    A *lower bound* on how long a frame can take, which is the only direction that can support a
    refusal: if a job cannot finish even at the fastest rate anyone has ever observed, no
    configuration will rescue it.
    """
    rows = [r for r in (calibration or [])
            if r.get("seconds_per_frame") and r.get("output_pixels")]
    if not rows:
        return None
    return min(r["seconds_per_frame"] * (output_pixels / float(r["output_pixels"]))
               for r in rows)


def refuse_frames_no_deadline_admits(job, calibration, output_pixels):
    """Refuse a frame count that no deadline could accommodate — **arithmetic, not a constant.**

    **The bound is the platform's ceiling, not the caller's deadline** (F-2026-08-18-15). The
    caller's own deadline is `refuse_if_the_deadline_cannot_be_met`'s business, with its safety
    factor and its margin for an approximate rate; duplicating it here would give one job two
    deadline refusals wearing different error codes. What this answers is the stronger question:
    is there *any* deadline this could finish inside? Seven days is RunPod's, so past it there
    is not. `estimated_frames` comes from container metadata — duration x fps, both
    attacker-controlled on an untrusted source — and nothing upstream bounds it. A claim of a
    billion frames costs real CPU and real memory to plan, and `plan_only` reaches it without
    spending a GPU-second.

    What this deliberately is not is a maximum frame count. `validation.py` records CF
    withdrawing exactly that kind of invented input rule, and a constant here would repeat it:
    a number nobody measured, refusing work that would have succeeded. What replaces it is the
    arithmetic that was always available — at the fastest rate ever measured, does this many
    frames fit in the time the job actually has? A source claiming more than that is not
    describing a job anyone can run.

    Silent where it cannot know: no rate measured, no frame count, no refusal.
    """
    frames = job.get("estimated_frames")
    if not frames or frames < 1:
        return
    per_frame = fastest_seconds_per_frame(calibration, output_pixels)
    if not per_frame:
        return
    budget_s = PLATFORM_EXECUTION_CEILING_MS / 1000.0
    fastest_s = frames * per_frame
    if fastest_s <= budget_s:
        return
    raise WorkerError(
        INVALID_FIELD_VALUE,
        "the source reports {:,} frames, which at the fastest rate ever measured here "
        "({:.3f} s/frame at this size) is {:,.0f} s of work against a {:,.0f} s {} — it cannot "
        "finish, so there is no configuration to plan. Check the source's duration and frame "
        "rate: container metadata claiming a duration the file does not have produces exactly "
        "this.".format(frames, per_frame, fastest_s, budget_s,
                       "platform execution ceiling"),
        shortfall={"estimated_frames": frames,
                   "fastest_seconds_per_frame": round(per_frame, 4),
                   "seconds_at_that_rate": round(fastest_s, 1),
                   "budget_seconds": round(budget_s, 1),
                   "budget_source": "runpod execution ceiling"},
    )


#: How many pass lengths a rationale will carry verbatim. Above it the list is summarised.
#: **The cap lives here, at the serialization seam, and never in `planner.plan`'s return** — the
#: kit freezes the plan dict's shape, so a change there is a repricing rather than a fix.
MAX_REPORTED_PASSES = 64


def _summarise_passes(passes):
    """The schedule, or a description of it once printing it stops informing anyone.

    **A schedule is not a payload** (F-2026-08-18-15). `passes` is serialized verbatim into the
    response envelope and the manifest, and a legitimate two-hour clip already ships thousands of
    elements — every one of them the same number, since only the tail ever differs. A caller
    reading that learns nothing the summary does not tell them, and a worker writing it spends
    real bytes on every poll.

    Kept verbatim below the cap so that short schedules — which is every case anyone reads by
    hand, and every case the kit freezes — are unchanged.
    """
    if not passes or len(passes) <= MAX_REPORTED_PASSES:
        return passes
    return {
        "count": len(passes),
        "first": passes[0],
        "last": passes[-1],
        "distinct": sorted(set(passes)),
        "note": "summarised above {} passes; every pass is the window except the tail".format(
            MAX_REPORTED_PASSES),
    }


def plan(job, snapshot, calibration=None, force_rung=None):
    """Choose a configuration before any work starts. Returns `(plan, rationale)`.

    **The heart is `planner`, and this function is the wiring around it.** Six steps over
    registry v1.2's per-phase lines decide the window, both grids, the postprocess assert and
    the chunk; everything here does is translate geometry in and a runnable configuration out,
    then attach the timing prediction the deadline guard needs.

    What used to live here — the rung ladder as a decider, the pooled frontier, the raw-row
    matching against `calibration.json` — is gone (F-14). Those were the mechanism for *not*
    having formulas, and the campaign that produced the registry ended the need for them. The
    table survives for **time only**: the registry models memory and says nothing about wall
    clock, and a deadline check that disappears is indistinguishable from one that passes.

    `job` carries `target_short_edge_px`, `source_width`, `source_height` and `estimated_frames`
    — the last from the source's duration and rate, used for **planning and the ETA only**.
    Every frame count this worker reports comes from the decode.
    """
    calibration = load_calibration() if calibration is None else calibration
    width, height = output_dimensions(
        job["source_width"], job["source_height"], job["target_short_edge_px"]
    )
    output_pixels = width * height
    usable = _usable_vram(snapshot)
    src = (int(job["source_width"]), int(job["source_height"]))
    target = int(job["target_short_edge_px"])
    # **A still is one frame, and that is a fact about the source rather than a guess.** Where
    # the caller's shape does not say, an unknown frame count plans as a single pass over what
    # the duration implies; `frames` is never allowed to be zero, which would divide the chunk
    # arithmetic by nothing.
    frames = int(job.get("estimated_frames") or 0)
    if job.get("still"):
        frames = 1
    frames = max(1, frames)
    host_ram_gb = snapshot.get("host_ram_gb")
    tile_quality = job.get("tile_quality") or "default"
    schedule = job.get("schedule") or "max_window"

    job_for_config = dict(job, output_width=width, output_height=height,
                          output_pixels=output_pixels,
                          source_pixels=src[0] * src[1])

    if force_rung is not None:
        # **`RUNGS` survives as the forced-configuration facility and nothing else.** It is how
        # the gate runs calibration at chosen coordinates, and a forced run is a deliberate
        # instruction to find out rather than a configuration anyone is claiming fits.
        index = force_rung
        chosen = dict(RUNGS[index])
        chosen["target_short_edge_px"] = job["target_short_edge_px"]
        answer = None
        why = "forced to rung '{}' — pinned for calibration, not chosen from the formulas".format(
            RUNGS[index]["name"])
    elif not (snapshot.get("vram_free_gb") or snapshot.get("vram_total_gb")):
        # §6's blind case: no reading means nothing can be claimed to fit, so the answer is the
        # lowest-priced plan that is still in spec — flagged, never sub-floor, never `solved`.
        index = FLOOR_INDEX
        answer = planner.lowest_in_spec(src, frames, target, host_ram_gb=host_ram_gb,
                                        tile_quality=tile_quality,
                                        gpu_name=snapshot.get("gpu_name"))
        chosen = solver.config_of_plan(answer, job_for_config)
        chosen["name"] = "blind"
        why = ("free VRAM could not be read, so no configuration can be claimed to fit. This is "
               "the lowest-priced plan that is still in spec — window {}, grids {} — and it is "
               "flagged rather than presented as a choice".format(answer["w"], answer["dec"]))
    else:
        index = None
        # **The card class reaches the host model** (ratified 2026-08-20): the materialised
        # constant is ~11 GiB larger on the big cards than on the A40, and an unnamed card takes
        # the larger figure — a card nobody has measured is not an argument for optimism.
        answer = planner.plan(src, frames, target, usable_gb=usable, host_ram_gb=host_ram_gb,
                              tile_quality=tile_quality, schedule=schedule,
                              gpu_name=snapshot.get("gpu_name"))
        if answer["action"] != "plan":
            # **The terminal state, computed before a GPU-second is spent.** The two options are
            # the caller's to choose between and neither is this function's to take; a sub-floor
            # plan is never returned silently.
            raise WorkerError(
                CAPACITY_EXCEEDED,
                _refusal_text(answer["reason"]),
                remedy=Remedy.LARGER_GPU,
                shortfall={
                    "usable_vram_gb": None if usable is None else round(usable, 2),
                    # **This shot's floor, not the constant.** The message beside it already
                    # quoted `window_floor(frames)`, so a ten-frame clip was refused with prose
                    # saying 13 and a machine-readable field saying 21 — and the field is the
                    # half CF parses. A refusal that contradicts itself is worse than either
                    # number alone.
                    "quality_floor_frames": planner.window_floor(frames),
                    "registry_version": planner.REGISTRY_VERSION,
                    "options": _terminal_options(job, snapshot, width, height),
                },
            )
        chosen = solver.config_of_plan(answer, job_for_config)
        chosen["name"] = "planned"
        why = planner.rationale_line(answer)

    rationale = {
        # An internal label. It reads `planned` on every ordinary job, `blind` where the card
        # could not be read and the rung's own name only where a caller forced one.
        "rung": chosen["name"],
        "rung_index": index,
        "reason": why,
        # **"Calibrated" now means what the build guarantees.** It used to mean "the table has
        # rows", which was a state the worker could be in or out of; the constants are baked in,
        # so the state cannot not-hold.
        "calibrated": True,
        "registry_version": planner.REGISTRY_VERSION,
        "output_width": width,
        "output_height": height,
        "output_pixels": output_pixels,
        "usable_vram_gb": None if usable is None else round(usable, 2),
        "predicted_peak_vram_gb": None,
        "predicted_seconds": None,
        "tile_quality": tile_quality,
        "schedule": schedule,
    }

    if answer is not None:
        # **Everything the plan decided, not a subset of it.** Reconstructing the plan from the
        # report is the entire job of the report on a run whose purpose is measurement, and a
        # rationale carrying the window but not the grid, the overlap or the prices describes a
        # plan nobody can re-derive.
        rationale.update({
            "window": answer["w"],
            "temporal_window": answer["w"],
            "ideal_window": answer["ideal_window"],
            "chunk_size": answer["chunk"],
            "temporal_overlap": answer["v"],
            "passes": _summarise_passes(answer["passes"]),
            # **The chunk span, and the tail that is not like it** (ruled decision 7). When the
            # host forces chunking, the chunk is quantized to whole windows and the remainder
            # runs as its own final chunk at its own padded window — a genuinely different
            # configuration for the last few seconds of the clip, and one a reader seeing only
            # `chunk` and `passes` would never know about.
            "chunks": answer.get("chunks"),
            "tail_chunk": answer.get("tail_chunk"),
            "tail_passes": _summarise_passes(answer.get("tail_passes")),
            # Reported only when the lever was honoured — see `planner.plan`.
            "schedule_applied": answer.get("schedule") or "max_window",
            "shortest_pass": min(answer["passes"] + (answer.get("tail_passes") or [])),
            "decode_grid": answer["dec"], "decode_tile": answer["dec_tile"],
            "encode_grid": answer["enc"], "encode_tile": answer["enc_tile"],
            "decode_blend": answer.get("dec_blend"),
            "phase_prices_gb": answer["prices"],
            "binding_phase": answer["binder"],
            "predicted_peak_vram_gb": round(max(answer["prices"].values()), 2),
            "anchored": answer["anchored"],
            "blind": bool(answer.get("blind")),
            "blocks_to_swap": answer["blocks_to_swap"],
            # **The rung, and the two counts that are not the same count** (amendment 9). The
            # executor reads `residency` from this dict to decide whether to schedule an
            # eviction at all, so a rationale that dropped it would leave rung 2 decided and
            # never performed — a plan sized for a container nobody arranged.
            "residency": answer.get("residency", planner.RESIDENT),
            "evictions": answer.get("evictions", 0),
            "reloads": answer.get("reloads", 0),
            "host_gb": round(solver.host_gb(chosen, src[0] * src[1], output_pixels), 2),
            "rationale": planner.rationale_line(answer),
        })
        if not answer["anchored"]:
            rationale["span_warning"] = (
                "{:.1f} GiB usable is outside the span the registry constants were measured "
                "against (up to {:.1f}); this plan is an extrapolation and says so"
                .format(usable or 0.0, planner.ANCHORED_MAX_USABLE))

    _attach_timing(rationale, calibration, chosen, job, snapshot, output_pixels)
    # After the rate, because it adds to the total the rate produced — and never inside it, so a
    # run that has no timing row at all still reports what its reloads will cost.
    _attach_reload_cost(rationale, snapshot)
    return chosen, rationale


#: **What one re-materialisation costs, per card class** — rung 2's own time term (amendment 9:
#: "the reload cost is measured per card class from the corpus (cache-warm reads, not assumed)
#: and enters the time quote as its own term").
#:
#: **EMPTY, AND THAT IS THE HONEST STATE OF THE CORPUS.** The amendment asks for cache-warm reads
#: per class; the corpus holds exactly one strip measurement of any kind — `strip_seconds: 77.7`
#: on the B200 row of 2026-08-20T14:43, and its own note decomposes it as import 2.3 + prepare
#: 75.4. That number is *cold*: a container's first checkpoint read, off disk, with no page cache
#: behind it. A rung-2 reload is warm by construction — the 16 GiB of file cache the anon/file
#: caveat measured is the checkpoint still sitting in memory — so the cold figure is an upper
#: bound on the warm one and not a measurement of it.
#:
#: Carrying the bound rather than nothing, because the two errors are not symmetric and this
#: number feeds the deadline preflight: over-quoting delays a job CF can re-route, under-quoting
#: lets a doomed one run and be hard-killed at `executionTimeout` with every second billed. Same
#: asymmetry, same direction, as `DEADLINE_SAFETY_FACTOR` below.
#:
#: A row here is what retires the bound, per class. One cold-vs-warm pair on any card fills it.
MODEL_RELOAD_SECONDS = {}

#: The one strip measurement the corpus holds, used as a ceiling until a warm read exists.
#: Cross-card as well as cold, which is two admissions in one number — the reload is dominated by
#: a fixed 16.4 GiB checkpoint read, so a class figure is unlikely to exceed it, but "unlikely" is
#: not "measured" and the rationale says `bounded` rather than `measured` because of it.
MODEL_RELOAD_BOUND_S = 77.7


def model_reload_seconds(gpu_name=None):
    """`(seconds, basis)` for one re-materialisation on this card class."""
    for name, value in MODEL_RELOAD_SECONDS.items():
        if name in (gpu_name or ""):
            return value, "measured"
    return MODEL_RELOAD_BOUND_S, "bounded"


def _attach_reload_cost(rationale, snapshot):
    """Rung 2's reloads, priced into the quote as their own line rather than folded into a rate.

    Its own term, because amendment 6e's lesson was exactly this shape one axis over: 365 s of
    pure strip hidden inside a "per-frame" rate made two runs look 20% apart when their phases
    matched to seconds. A reload is a fixed cost that happens a countable number of times, and a
    quote that smears it across frames cannot be checked by hand against the registry.
    """
    reloads = int(rationale.get("reloads") or 0)
    if reloads <= 0:
        return
    seconds, basis = model_reload_seconds(snapshot.get("gpu_name"))
    rationale["reload_seconds_each"] = seconds
    rationale["reload_basis"] = basis
    rationale["reload_seconds_total"] = round(reloads * seconds, 1)
    if rationale.get("predicted_seconds") is not None:
        rationale["predicted_seconds"] = round(
            rationale["predicted_seconds"] + reloads * seconds, 1)


def _attach_timing(rationale, calibration, chosen, job, snapshot, output_pixels):
    """The ETA and the deadline guard's input — **time only, and from the table only.**

    The registry models memory; it says nothing about wall clock, and the host-tail time model
    is an open item. So `calibration.json` keeps exactly one job: predicting seconds. It no
    longer decides anything about what is run, which is the separation F-14 asked for — a peak
    borrowed from another row is an OOM, while a rate borrowed from another row is a labelled
    estimate, and only one of those is worth the risk.
    """
    if not calibration:
        return
    window = rationale.get("temporal_window")
    # The plan's own window decides which kind of row may price it. A still always plans 1; a
    # video that plans 1 is the floor rung, and costs the same way for the same reason.
    comparable = _timing_rows(calibration, output_pixels, window,
                              unbatched=(window == 1))
    if not comparable:
        per_frame = _approximate_seconds_per_frame(
            calibration, chosen["name"], output_pixels, job.get("estimated_frames"))
        if per_frame is not None:
            rationale["seconds_per_frame"] = round(per_frame, 4)
            rationale["prediction_basis"] = "approximate"
            if job.get("estimated_frames"):
                rationale["predicted_seconds"] = round(per_frame * job["estimated_frames"], 1)
        return

    # **Time is matched on the card and memory never was.** A 1.5x slower card silently
    # inheriting another's rate accepts a job it cannot finish and is hard-killed at
    # `executionTimeout` with every second billed — which is the failure the deadline factor
    # exists to prevent, and which is calibrated per card.
    running_on = snapshot.get("gpu_name")
    same_card = [r for r in comparable if running_on and r.get("gpu_name") == running_on]
    timing_rows = same_card or comparable
    rows = [r for r in timing_rows if r.get("output_pixels")]
    if not rows:
        return

    # **And matched on the tiling, for the same reason** (F-2026-08-20-40). `tile_quality` moves
    # the decode grid, not the geometry: at 8K the default grid is 6x4 tiles at 1392 px and
    # `high` is 3x2 at 2648, which is seven decode passes of ~900 s each against ~120 s. The
    # entire 2x wall lives there. A default-tiling row priced that job at 4147.4 s against 8255 s
    # actually spent — **and wore `prediction_basis: "measured"` while doing it**, the strongest
    # label §9 allows, on a configuration nothing in the table had ever run.
    #
    # §9's span rule already governs this: a coefficient prices only configurations within span of
    # the plane it was measured on, and a new plane class is admissible under a pooled fallback
    # *flagged as such*. The flag was the missing half. So the rows are preferred by tiling like
    # they are by card, and where none match the prediction still happens — an ETA is better than
    # no ETA — but it is labelled `borrowed` and says what it borrowed from.
    job_tiling = rationale.get("tile_quality") or DEFAULT_TILE_QUALITY
    same_tiling = [r for r in rows if _row_tile_quality(r) == job_tiling]
    borrowed_tiling = not same_tiling
    if same_tiling:
        rows = same_tiling

    per_frame = max(r["seconds_per_frame"] * (output_pixels / float(r["output_pixels"]))
                    for r in rows)
    rationale["seconds_per_frame"] = round(per_frame, 4)
    # **`borrowed` outranks `measured` in the labelling, because it is the weaker claim.** A rate
    # taken from another tiling is not a measurement of this configuration in any sense, and the
    # deadline guard reads this field to decide how much rope a prediction gets — the one that
    # was 2x wrong must not be trusted like the one that was not.
    # **A rate from another card is not a measurement of this one either** (F-2026-08-20-40,
    # fourth face). `timing_from_another_card` has flagged this since the field existed, and the
    # basis went on saying `measured` beside it — which is the same vocabulary violation the
    # tiling case was ruled on, one axis over. Formulas §9: `measured` claims only what THIS run
    # measured on the job in front of it.
    #
    # It surfaced on a 48-class card that is not an A40: no rows for it, so every row in the
    # table became comparable, and the plan reported a borrowed number as a measured one.
    rationale["prediction_basis"] = (
        "borrowed" if (borrowed_tiling or not same_card) else "measured")
    if borrowed_tiling:
        rationale["timing_from_another_tiling"] = {
            "running_at": job_tiling,
            "rows_measured_at": sorted({_row_tile_quality(r) for r in rows}),
            "why_it_matters": ("tile_quality moves the decode grid, and decode is where a long "
                               "job's wall clock lives; a high-tiling 8K job measured 2x its "
                               "default-tiling prediction (F-2026-08-20-40)"),
        }
    if not same_card:
        rationale["timing_from_another_card"] = {
            "running_on": running_on,
            "rows_measured_on": sorted({r["gpu_name"] for r in rows if r.get("gpu_name")}),
        }
    if job.get("estimated_frames"):
        rationale["predicted_seconds"] = round(per_frame * job["estimated_frames"], 1)


#: How far past the remaining deadline an *approximate* prediction has to reach before it refuses.
#: A measured prediction refuses on any overrun; a scaled one has to be wrong by more than a factor
#: of two to be wrong about this, and the cost of the two errors is not symmetric — refusing a job
#: that would have finished costs the customer their result, while letting a doomed one run costs
#: the seconds it was going to cost anyway.
APPROXIMATE_DEADLINE_MARGIN = 2.0

#: **The prediction is multiplied by this before it is compared to the deadline** (CF, 2026-08-15).
#:
#: Not a fudge factor: the estimator is *known* to err under, which is the one direction that
#: matters here. It was 25% under on a large job before the pixel-scaling fix and ~10% under after
#: it — a 929-frame run measured 18.23 s/frame against 14.54 predicted, and nothing refuses a job
#: this much under. Under-prediction is exactly what lets the refusal pass work it should have
#: turned away.
#:
#: 1.5 because the asymmetry decides it, on CF's standing rule: a false refusal costs a round trip
#: CF can see and act on, while a miss costs a full-price nothing — a hard kill with no master, no
#: error and every second billed. 1.5 would have caught that 929-frame run, which is the only
#: empirical test available.
#:
#: **Two margins, neither depending on the other.** CF sizes its ceiling generously and
#: independently; this is a second line of defence that should rarely engage. So a refusal that
#: fires *often* is evidence about CF's ceiling being tight, not about 1.5 being wrong — and it is
#: not a guarantee, because the check still fails open (no measurement for a job's shape means no
#: check at all, and "no refusal" is not "it fits").
DEADLINE_SAFETY_FACTOR = 1.5


def _approximate_seconds_per_frame(calibration, rung_name, output_pixels, estimated_frames=None):
    """Per-frame time for a rung with nothing comparable measured, scaled by pixel count.

    Inference time on this model is close to linear in output pixels — the measured 4K and 8K runs
    at `swapped` differ by 4x in pixels and 4.9x in time — so a run at any size gives a usable
    figure for a run at another. It is not good enough to *plan* with, which is why the rung choice
    still uses `_matching_runs` and its 2x window, and it is not recorded as a measurement. It is
    good enough to answer "is this job hours away from a one-minute deadline", which is the only
    question the refusal asks.

    The **median** rather than the max: the max is the conservative choice for capacity, where
    being wrong means an OOM, and the reckless one for a refusal, where being wrong means denying
    a job that would have finished.

    **And a still is not a slow video.** This scaled by pixels and ignored frame count entirely,
    while `_matching_runs` directly above it has always known frames matter. A one-frame row's
    per-frame time is mostly fixed cost -- the model load a sequence amortises over hundreds of
    frames -- so scaling it by pixels answers a different question from the one being asked.

    Measured on 2026-08-15: a 121-frame 4K job at `fast` drew the median of three rows, two of
    them single stills, and predicted **77.0 s/frame** where the A40 does that work at 14.59 and a
    Blackwell does it faster. About 5x over, and the deadline guard refused a job that would have
    finished in twenty minutes -- which is exactly the failure CF named when it asked this worker
    not to harden the factor into a guarantee.

    So a sequence is estimated from sequences and a still from stills, and only where neither
    exists does it fall back to everything rather than to nothing. A rough number is the point of
    this function; a rough number drawn from the wrong kind of run is not.
    """
    rows = [run for run in calibration
            if run.get("rung") == rung_name
            and run.get("seconds_per_frame") and run.get("output_pixels")]
    if estimated_frames:
        sequence = estimated_frames > 1
        alike = [run for run in rows
                 if run.get("frames") and (run["frames"] > 1) == sequence]
        rows = alike or rows
    scaled = sorted(
        run["seconds_per_frame"] * (output_pixels / float(run["output_pixels"]))
        for run in rows)
    if not scaled:
        return None
    return scaled[len(scaled) // 2]




def _usable_vram(snapshot):
    """Budget net of the reserve, or `None` when the card cannot be read at all.

    **Falls back to `vram_total_gb`, because the blind guard beside it already does**
    (F-2026-08-18-28). A snapshot carrying a total but no free reading — a driver that answers
    one query and not the other — cleared the blind branch on the total and then arrived here,
    where `free is None` produced `usable = None`, which `planner.plan` reads as a budget of
    zero and refuses. A card with 80 GiB reported was being told it had none. Latent, because
    every snapshot seen so far carries both; the disagreement between two guards reading the
    same fields is the defect, not the driver.
    """
    free = snapshot.get("vram_free_gb") or snapshot.get("vram_total_gb")
    return None if free is None else max(0.0, free - VRAM_RESERVE_GB)




#: The clause the predicate's own floor refusal already ends with (`planner.py:446`). Matched
#: rather than assumed: the terminal branch has more than one reason shape — a still that cannot
#: fit at window 1 never mentions a floor at all — so the boilerplate below is still owed to those.
_FLOOR_CLAUSE = "the model stops beating a plain enlargement"


def _refusal_text(reason):
    """The predicate's reason, finished off without saying its last sentence twice.

    **F-2026-08-19-30.** The assembler appended its own floor sentence to every terminal reason,
    and the floor-refusal reason already ends with that sentence — so the 8K refusal test read
    back: "…below the floor the model stops beating a plain enlargement. Below that window the
    model stops beating a plain enlargement, so there is nothing quieter to fall back to."
    Cosmetic, but this text is the customer-facing half of a refusal and CF quotes it verbatim.

    The remaining half of the boilerplate — that there is nothing quieter to fall back to — is
    not a restatement and is what makes the refusal terminal rather than a suggestion, so it is
    kept in both branches.
    """
    if _FLOOR_CLAUSE in reason:
        return "{}, so there is nothing quieter to fall back to.".format(reason)
    return ("{}. Below that window the model stops beating a plain enlargement, so there is "
            "nothing quieter to fall back to.".format(reason))


def _terminal_options(job, snapshot, width, height):
    """§6: the two choices a caller has when nothing at or above the floor fits.

    **Reported, never taken.** Both give up something that belongs to the caller -- the delivered
    resolution, or the guarantee that the model beat a plain enlargement -- and a worker that
    picks one while a job is running has made a product decision on their behalf. That is the one
    decision this design says the solver reports instead of making.

    The smaller target is computed, not suggested: the largest short edge whose best plan the card
    is predicted to hold, stepped down the same 32 px grid a caller would think in and snapped to
    an even number, since `yuv420p` cannot encode an odd dimension.
    """
    floor = planner.window_floor(max(1, int(job.get("estimated_frames") or 1)))
    options = [{
        "option": "run_below_quality_floor",
        "how": "resend with allow_below_quality_floor=true",
        "cost": "the temporal window falls below {} frames, where the model no longer beats a "
                "plain enlargement; the job is flagged in its manifest".format(floor),
    }]

    source_w = job["source_width"]
    source_h = job["source_height"]
    short_edge = int(job["target_short_edge_px"])
    while short_edge > 64:
        short_edge = (short_edge - 32) // 2 * 2
        out_w, out_h = output_dimensions(source_w, source_h, short_edge)
        answer = planner.plan(
            (source_w, source_h), max(1, int(job.get("estimated_frames") or 1)), short_edge,
            usable_gb=_usable_vram(snapshot), host_ram_gb=snapshot.get("host_ram_gb"),
            gpu_name=snapshot.get("gpu_name"))
        if answer["action"] == "plan":
            options.insert(0, {
                "option": "reduce_target_resolution",
                "how": "resend with target_short_edge_px={}".format(short_edge),
                "cost": "delivers {}x{} instead of {}x{}, at a window of {} frames".format(
                    out_w, out_h, width, height, answer["w"]),
            })
            break
    return options


def refuse_if_the_deadline_cannot_be_met(rationale, budget_ms, elapsed_s, frames):
    """Refuse a job that cannot finish in the time it was given, before the GPU is spent.

    **`execution_timeout_ms` is a hard kill.** RunPod ends the container: no master is written, no
    error is returned, every second is billed, and the caller receives `TIMED_OUT` with nothing
    attached. The worker never gets to say anything because it is not asked — it is terminated.
    This is the same trade as `capacity_exceeded`, on the other axis.

    **Measured against elapsed-since-handler-entry, never a wall clock.** CF sends a cap rather
    than a timestamp deliberately: `executionTimeout` bounds time *being processed* while queue
    time lives under `ttl`, a separate clock — so an absolute deadline would be measuring the
    wrong one and would be wrong by however long the job sat in the queue. CF sends the cap, this
    owns the stopwatch, and the two agree by construction.

    **Declines to guess.** No deadline, no prediction, or no frame count means no refusal — the
    same rule the capacity refusal follows, for the same reason: refusing a job that would have
    succeeded costs the customer their result.

    The estimate covers **inference only**; fetch is already spent and the upload follows. So the
    comparison is against the remaining budget with the write left out, which errs toward letting
    a borderline job run. CF sizes the cap generously and an unhit timeout is free, so the
    borderline case is rare by construction.
    """
    if not budget_ms:
        return
    predicted = rationale.get("predicted_seconds")
    if not predicted or not frames:
        return
    remaining = (budget_ms / 1000.0) - elapsed_s
    # An approximate prediction is scaled from runs at another size rather than measured at this
    # one, so it refuses only on a margin it cannot plausibly have invented. The two multipliers
    # compose rather than compete: the safety factor raises what the job is assumed to need, the
    # approximate margin raises what a *guess* is allowed to overrun by, and an approximate
    # prediction still gets more rope than a measured one (1.33x remaining against 0.67x).
    # **`borrowed` gets the same rope as `approximate`, and for a sharper reason.** Both are
    # predictions the table cannot stand behind at this configuration; the borrowed-tiling one is
    # the case that was actually observed 2x wrong (F-2026-08-20-40), which is precisely the
    # "wrong by more than a factor of two" the approximate margin was sized for. Refusing a job on
    # a number that has been that wrong costs a customer a result they would have got.
    approximate = rationale.get("prediction_basis") in ("approximate", "borrowed")
    required = predicted * DEADLINE_SAFETY_FACTOR
    allowed = remaining * (APPROXIMATE_DEADLINE_MARGIN if approximate else 1.0)

    # **Recorded whether it fires or not, which is the half that was missing.** A refusal is
    # visible by definition; a job that passed with four seconds to spare looked identical to one
    # that passed with four hours. CF tunes the factor from what actually happens, and until this
    # existed the only observations available were the failures.
    rationale["deadline"] = {
        "predicted_seconds": round(predicted, 1),
        "safety_factor": DEADLINE_SAFETY_FACTOR,
        "required_seconds": round(required, 1),
        "remaining_seconds": round(remaining, 1),
        "allowance_seconds": round(allowed, 1),
        "headroom": round(allowed / required, 3) if required else None,
        "basis": rationale.get("prediction_basis"),
    }
    if required <= allowed:
        return
    raise WorkerError(
        DEADLINE_EXCEEDED,
        "this job needs about {:.0f}s of inference at rung '{}' ({} frames at {:.1f}s each, "
        "{}), {:.0f}s once the {:.1f}x safety factor is applied, and {:.0f}s of its {:.0f}s "
        "deadline remain after {:.0f}s already spent fetching and probing. Refused before the GPU "
        "is spent: on this deadline the container would be killed partway with the whole cost "
        "billed and nothing delivered. The factor exists because this estimator errs *under* — "
        "measured 10-25% under on long jobs — so an unfactored comparison passes work it should "
        "turn away."
        .format(predicted, rationale.get("rung"), frames,
                predicted / float(frames),
                {"approximate": "scaled from measurements at other sizes",
                 "borrowed": "borrowed from rows at another tile_quality, which moves the "
                             "decode grid and is where a long job's wall clock lives"}
                .get(rationale.get("prediction_basis"), "measured"),
                required, DEADLINE_SAFETY_FACTOR, remaining, budget_ms / 1000.0, elapsed_s),
        remedy=Remedy.LONGER_DEADLINE,
        shortfall={"predicted_seconds": round(predicted, 1),
                   # **The factored figure alongside the raw one, so the factor is tunable from
                   # what fires rather than arguable.** CF asked for all three — prediction,
                   # requirement and limit — precisely so that a refusal can be judged without
                   # re-deriving the arithmetic from a log.
                   "safety_factor": DEADLINE_SAFETY_FACTOR,
                   "required_seconds": round(required, 1),
                   "prediction_basis": rationale.get("prediction_basis"),
                   # **The inputs, because the output could not be reproduced from them.** A
                   # refusal on hardware predicted 63.5 s/frame where the same rung, table and
                   # target reproduce 12.1 locally, and the report carried no way to tell which
                   # of the two differed. Naming the pixel count and the per-frame figure the
                   # prediction was built from makes the next one self-explaining.
                   "output_pixels": rationale.get("output_pixels"),
                   "seconds_per_frame": rationale.get("seconds_per_frame"),
                   "remaining_seconds": round(remaining, 1),
                   "elapsed_seconds": round(elapsed_s, 1),
                   "execution_timeout_ms": budget_ms,
                   # What to resend. Named rather than left as arithmetic, because a caller that
                   # has to double blindly is being told less than this worker knows. CF resubmits
                   # against this figure rather than doubling (CF, 2026-08-15), so it has to be a
                   # number that passes: the *factored* requirement plus what is already spent.
                   # It used to be `predicted * 1.5` by coincidence of the same constant; now the
                   # factor is named once and this reads it, so the two cannot drift apart.
                   "suggested_execution_timeout_ms": int((elapsed_s + required) * 1000),
                   "frames": frames,
                   "rung": rationale.get("rung")},
    )


_ALLOC_PATTERN = re.compile(r"Tried to allocate ([\d.]+) ([KMG])iB")

#: `... 312.00 MiB is reserved by PyTorch but unallocated ...` — the allocator gap, straight from
#: the message. **This is the figure that decides whether a retry is worth its minutes.** A gap
#: that is a meaningful share of the shortfall says the memory existed and could not be handed
#: out in one piece, which a fresh allocator may fix; a gap that is noise against the shortfall
#: says the configuration does not fit, and retrying it buys a second identical failure at the
#: price of a full run.
_RESERVED_PATTERN = re.compile(r"([\d.]+) ([KMG])iB is reserved by PyTorch but unallocated")

#: Above this share of the shortfall, an OOM is treated as suspected fragmentation and earns one
#: retry. Policy, not measurement — named here so the first run that contradicts it can move it.
#: Re-exported from `planner`, which owns the amended rule. Kept as a name because the
#: acceptance kit imports it, and because one threshold with two spellings is how the
#: amendment came to be half-applied in the first place.
FRAGMENTATION_GAP_SHARE = planner.FRAGMENTATION_GAP_SHARE
FRAGMENTATION_GAP_FLOOR_GB = planner.FRAGMENTATION_GAP_FLOOR_GB
_CAPACITY_PATTERN = re.compile(r"GPU [\d]+ has a total capa?c?i?t?y? of ([\d.]+) ([KMG])iB")


def diagnose_oom(exception, snapshot):
    """Read the shortfall out of the exception, not out of a log.

    The caught OOM knows the allocation that failed, and PyTorch's message carries it. That is
    better information than any log gives and it is free of the platform's collection lag —
    worker logs on RunPod have been observed carrying timestamps 2 h 17 m apart for adjacent
    statements in one function.

    Returns a dict for `cf_error.shortfall`, or None where the exception says nothing usable.
    **Absent is better than invented**: CF sizes a hardware purchase off this figure.
    """
    return _diagnose_oom(exception, snapshot)


def reset_peak_vram():
    """Zero the peak counter so the next attempt's peak is that attempt's, not the job's.

    Called before **every** attempt. Without the reset a retry at a safer rung would inherit the
    peak of the attempt that just OOMed, and the calibration table would learn that the
    conservative rung needs as much memory as the one that failed — the exact opposite of what
    happened.

    Imported inside the function like every other torch touch here, because `handler` must import
    with the model chain still lazy or the rung-1 suite stops running in CI.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
    except Exception:  # noqa: BLE001 — a counter we cannot reset is not worth failing a job over
        pass


def release_gpu_memory():
    """Give the allocator back everything not currently held, before an in-place retry.

    **The reason an in-place retry is worth attempting at all.** PyTorch's caching allocator keeps
    freed blocks for reuse, so a chunk can fail to find a contiguous span while the total free
    memory would have been ample — which is the shape of a mid-run OOM. Measured evidence that
    this is real rather than theoretical: 43.71 GB has succeeded on a card where 43.01 GB failed,
    and the same `gpu_name` has reported 44.42, 44.43 and 47.4 GB of total memory.

    `gc.collect()` first, because the tensors from the failed chunk are usually only reachable
    from the traceback that is about to be discarded — emptying the cache before they are
    collected releases nothing that matters.

    Imported inside the function like every other torch touch here, so `handler` imports with the
    model chain still lazy and the rung-1 suite keeps running in CI.
    """
    import gc

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            # **The strongest reset available without leaving the process.** §5.1 asks the
            # fragmentation retry to run in a fresh process; a fresh process cannot be taken
            # mid-job here, because the same clause requires the frames already written to be
            # kept and they live in this process's writer. So the allocator is taken as close to
            # new as it can be from inside: synchronise so nothing is still in flight, empty the
            # cache, collect the inter-process handles that `empty_cache` alone leaves behind,
            # and zero the accumulated counters so the next attempt's statistics are its own.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_accumulated_memory_stats(0)
            torch.cuda.reset_peak_memory_stats(0)
    except Exception:  # noqa: BLE001 — memory we cannot release is not worth failing a job over
        pass


def observed_peak_vram_gb():
    """What the attempt actually used, or `None` off-GPU.

    **This is the measurement the estimator is made of.** `RUNGS` carries no peak figures and
    `_fastest_that_fits` reads them from calibration records; nothing else in this worker writes
    one. Until this lands in a manifest, every job runs the floor rung — the protection works and
    is never allowed to relax, which reads as the estimator being broken rather than uncalibrated.

    Recorded on **success**, where it was previously only read on failure (`diagnose_oom`). A run
    that OOMed tells you a rung does not fit; only a run that succeeded tells you what a rung
    costs.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 2)
    except Exception:  # noqa: BLE001
        return None
    return None


def _diagnose_oom(exception, snapshot):
    text = str(exception)
    tried_gb = _to_gb(_ALLOC_PATTERN.search(text))
    if tried_gb is None:
        return None

    peak_gb = None
    try:
        import torch
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    except Exception:  # noqa: BLE001 — a figure we cannot read is one we do not report
        peak_gb = None

    shortfall = {
        "failed_allocation_gb": round(tried_gb, 2),
        "had_vram_gb": snapshot.get("vram_total_gb"),
        "free_vram_at_start_gb": snapshot.get("vram_free_gb"),
    }
    # **The card's own statement of its size, at the moment it ran out.** `_CAPACITY_PATTERN` had
    # been compiled and never read — dead code, and the wrong half to delete: `had_vram_gb` comes
    # from a snapshot taken before the job began and can be stale or, on a MIG slice, describe a
    # different partition. This figure comes out of the allocator's own message and is therefore
    # a reading of the thing that actually failed.
    capacity = _CAPACITY_PATTERN.search(text)
    if capacity:
        shortfall["reported_capacity_gb"] = round(_to_gb(capacity), 2)
    if peak_gb is not None:
        shortfall["peak_allocated_gb"] = round(peak_gb, 2)
        # What the run was reaching for when it died. A lower bound on what it needed: the
        # allocator may have been about to ask for more.
        shortfall["needed_at_least_gb"] = round(peak_gb + tried_gb, 2)

    # **The allocator gap, and what it implies about retrying.** Both are read out of the message
    # rather than inferred, and both are recorded even when they point the boring way: a run that
    # failed with no gap worth mentioning is evidence that the 2.0 GB reserve is adequate for that
    # shape, which is exactly as useful as the opposite.
    gap_gb = _to_gb(_RESERVED_PATTERN.search(text))
    if gap_gb is not None:
        shortfall["reserved_unallocated_gb"] = round(gap_gb, 2)
        # **The amended rule, asked of the one implementation** (F-2026-08-18-13). The ratio
        # alone lived here for a build after the 2026-08-17 amendment reached the docs and the
        # kit: B2 failed a 20 MiB allocation on a full card with a 25 MiB gap, the ratio called
        # that fragmentation, and the job bought one pointless retry. The absolute floor is the
        # half that was missing, and it is asked of `planner` rather than recomputed here so
        # that the live path, the dry-run and the kit share a rule that has already been
        # amended once.
        shortfall["fragmentation_suspected"] = planner.fragmentation_suspected(gap_gb, tried_gb)
    return shortfall


def retry_is_worth_it(shortfall):
    """Whether an OOM earns one same-configuration retry. `True` when the message cannot say.

    **The default is to retry, because the message is the only thing that can rule it out.** An
    absent gap figure means an unfamiliar message, not a proven hard limit, and refusing to retry
    on a message we failed to parse would convert a parsing gap into a lost job.
    """
    if not shortfall:
        return True
    return shortfall.get("fragmentation_suspected", True)


def _to_gb(match):
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2)
    return value * {"K": 1 / (1024 ** 2), "M": 1 / 1024, "G": 1.0}[unit]


def is_oom(exception):
    """CUDA OOM, however it is spelled. `torch.cuda.OutOfMemoryError` subclasses RuntimeError and
    is not importable without torch, so this matches on the message as well as the type."""
    name = type(exception).__name__
    if name in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    text = str(exception).lower()
    return "out of memory" in text or "cuda oom" in text
