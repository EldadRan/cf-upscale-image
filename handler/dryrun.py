"""`--dry-run-walk <case.json>`: ask the shipped walk what it would do next. No GPU, no model.

Takes the shape `walk_oracle.walk()` takes -- `{N, src, out, usable, current, shortfall}` -- and
prints the worker's chosen next move as JSON: `action`, and for an attempt `w, enc, dec, v, chunk,
lever`.

**It calls the production walk.** `solver.next_after_oom` is the function the mid-stream ratchet
calls after a confirmed OOM. Nothing here re-implements a decision. The only thing computed
locally is §5.1's gap test, because the live path derives that while parsing an exception's text
and a case file states the two numbers directly; it asks `planner.fragmentation_suspected`, which
is the same function the live path asks, so there is one rule rather than two.

The one thing reconstructed rather than borrowed is the *failed configuration*: a case names it as
a window and two grid names (`w 109, untiled, 2x1`) and the walk needs a config dict.
`_config_from_case` rebuilds it through the same grid ladder the planner walks, so `"2x1"` means
what it means in `handler/tiles.py`.
"""

import json

import estimator
import planner
import solver
import tiles

#: Used when a case does not name one. Large enough that `_chunk_for` returns the whole clip,
#: which is what §2.1 says happens on this pool.
DEFAULT_HOST_RAM_GB = 1511.0


def _grid_named(name, width, height, overlap=128):
    """The ladder rung a case means by `"2x1"`, or `None` for untiled."""
    if name in (None, "untiled", "1x1"):
        return None
    for grid in tiles.grid_ladder(width, height, overlap):
        if "{}x{}".format(grid["nx"], grid["ny"]) == name:
            return grid
    raise ValueError("this plane has no {!r} rung on the grid ladder".format(name))


def _config_from_case(case, current):
    """The configuration that failed, as a config dict the walk can take."""
    src_w, src_h = case["src"]
    out_w, out_h = case["out"]
    lap = 128
    dec = _grid_named(current.get("dec"), out_w, out_h, lap)
    # **Both planes are R.** The encode grid used to be resolved against the source, which is
    # what the vendored tiler's docstring describes and what the registry's own fit contradicts:
    # R3 ran a 2176 tile on a 3840x2160 output and measured 47.64 GiB, where pricing that tile
    # on the model plane predicts 47.31 and pricing it on the 1920x1080 source predicts 23.9.
    enc = _grid_named(current.get("enc"), out_w, out_h, lap)

    config = solver.base_configuration(min(out_w, out_h))
    config.update(
        batch_size=int(current["w"]),
        chunk_size=int(current.get("chunk") or case["N"]),
        blocks_to_swap=int(current.get("b", solver.PROBE_BLOCKS_TO_SWAP)),
        swap_io_components=bool(current.get("swap_io")),
        vae_decode_tiled=dec is not None, vae_encode_tiled=enc is not None,
        vae_decode_tile_overlap=lap, vae_encode_tile_overlap=lap)
    config["dit_offload_device"] = "cpu" if config["blocks_to_swap"] else "none"
    if dec:
        config["vae_decode_tile_size"] = dec["tile"]
    if enc:
        config["vae_encode_tile_size"] = enc["tile"]
    return config


def _job_and_snapshot(case):
    src_w, src_h = case["src"]
    out_w, out_h = case["out"]
    job = {
        "estimated_frames": int(case["N"]),
        "source_width": src_w, "source_height": src_h,
        "output_width": out_w, "output_height": out_h,
        "source_pixels": src_w * src_h, "output_pixels": out_w * out_h,
        "target_short_edge_px": min(out_w, out_h),
    }
    # A case names `usable`, the far side of `free - VRAM_RESERVE_GB`; this puts the reserve back
    # so `solver._usable` subtracts it again and arrives where the case meant.
    free = float(case["usable"]) + solver.VRAM_RESERVE_GB
    snapshot = {"vram_free_gb": free, "vram_total_gb": free,
                "host_ram_gb": float(case.get("host_ram_gb") or DEFAULT_HOST_RAM_GB)}
    return job, snapshot


def walk(case):
    """The worker's next move, in the case file's vocabulary."""
    job, snapshot = _job_and_snapshot(case)
    current = case["current"]
    shortfall = dict(case["shortfall"])
    phase = shortfall.get("phase")
    config = _config_from_case(case, current)

    # §5.1's gap rule. `estimator.retry_is_worth_it` reads `fragmentation_suspected`, which the
    # live path sets while parsing the OOM text; a case states the two numbers instead, so the
    # same comparison is made here against `estimator`'s own threshold.
    if "fragmentation_suspected" not in shortfall:
        # **The one implementation, asked rather than re-derived.** This used to recompute the
        # rule from `FRAGMENTATION_GAP_SHARE` alone, which meant the dry-run and the live path
        # disagreed for the whole build in which the 1.0 GiB absolute floor reached the docs and
        # not the code (F-2026-08-18-13). A rule that has been amended once will be amended
        # again, and three copies is three chances to miss it.
        shortfall["fragmentation_suspected"] = planner.fragmentation_suspected(
            shortfall.get("reserved_unallocated_gb"), shortfall.get("failed_allocation_gb"))

    if estimator.retry_is_worth_it(shortfall):
        return {"action": "retry"}

    row, _why = solver.next_after_oom(
        job, snapshot, config, current.get("P"),
        phase=phase, shortfall=shortfall,
        # Probing runs do not relax `b`; `pin` is what turns relaxation off live.
        relax_swap=bool(case.get("relax_swap")))

    if row is None:
        return {"action": "terminal"}

    chosen = row["config"]
    return {
        "action": "attempt",
        "w": solver.window_of(chosen),
        "enc": row["encode_grid"] if chosen["vae_encode_tiled"] else "untiled",
        "dec": row["decode_grid"] if chosen["vae_decode_tiled"] else "untiled",
        "v": chosen["temporal_overlap"],
        "chunk": chosen["chunk_size"],
        "lever": planner.PHASE_LEVER.get(phase, "window"),
    }


def main(path):
    with open(path) as handle:
        case = json.load(handle)
    print(json.dumps(walk(case), indent=1, sort_keys=True))
    return 0
