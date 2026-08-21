"""Choosing a configuration lever by lever, instead of picking one of six bundles.

**What this replaces.** `RUNGS` was six hard-coded configurations, each carrying five knobs at
once, and `more_conservative` stepped between them. Three things were wrong with that and all
three were expensive:

  *It could not answer a question about one lever.* Changing tiling meant changing rung, which
  also changed the window -- the dominant quality lever -- so every tiling measurement was
  confounded with the thing it was competing against (`decisions.md` 4.41).

  *It overshot.* One step could take the window from 49 to 9, a five-fold cut, when the failure
  wanted a smaller decode tile and nothing else (4.39).

  *It had a ceiling nobody chose.* The window came from a list of *measured* values, so a 180 GB
  card stopped at 49 because that was the largest window anyone had run -- not because anything
  said it would not fit.

**The shape of the replacement.** A configuration is a point in lever space. `solve` walks the
4n+1 lattice down from the shot length, and at each window applies relief in *quality-cost order*
until the prediction fits or the levers are spent. `relieve` steps exactly one lever, chosen by
the phase that actually failed.

**The lever order is the whole design, and it is not the ladder's.**

    1. blocks_to_swap        time only -- NO quality cost at all
    2. swap_io_components    time only
    3. decode tiling         seams, on the output
    4. smaller decode tile   more seams, monotonically
    5. encode tiling         seams, on the input (smaller frame on an upscale, so cheaper)
    6. smaller encode tile   more seams
    7. chunk below the clip  hard cuts -- the only unblended seams in the system
    8. window                the picture itself

Block-swapping first is the headline. It buys memory with time and nothing else, and the ladder
had it arriving at rung four, *after* tiling was already on.

**Why the window is last and the chunk is second-to-last.** A chunk boundary discards its context
frames rather than blending them (`generation_phases.py` blends only *within* a chunk), so
lowering the chunk introduces the only hard cuts a job has. And the chunk costs nothing in VRAM --
measured, 31.37 GB at chunk 21 and at chunk 192 with the window held -- so it should never be
touched for memory at all. It is on this list only because it bounds *host* RAM.

**On being wrong.** The prediction will be wrong; it is fitted from a handful of points. That is
survivable because an OOM now costs one chunk rather than the job, so this plans at the edge on
purpose. What it must not do is be *confidently* wrong, so every prediction carries how far it
extrapolated and `solve` refuses the edge when it is guessing.
"""

import planner
import tiles

#: Held back from planning. **The single source is `planner`** — the reserve moved from 2.0 to
#: 1.0 on two direct measurements of the true reserve (R3 1.27, E1b 1.03), and a second copy of
#: that number here is a second thing to forget to change.
VRAM_RESERVE_GB = planner.VRAM_RESERVE

#: Held back from the host. The chunk lives here -- input frames as float32 and the output canvas
#: -- and a chunk of the whole clip is free in VRAM but emphatically not in RAM: the 192-frame 4K
#: run held about 13.3 GB there. Nothing measured it at the time, which is why the bound exists.
HOST_RESERVE_GB = 4.0

#: Bytes per pixel the host holds per frame, measured from the shapes the vendored reader and the
#: tile accumulator actually allocate: input frames arrive float32 RGB (12 B/px of *source*) and
#: the canvas is fp16 RGB (6 B/px of *output*).
HOST_BYTES_PER_SOURCE_PIXEL = 12
HOST_BYTES_PER_OUTPUT_PIXEL = 6





def lattice(limit):
    """The 4n+1 window sizes the model accepts, up to `limit`, largest first.

    **Generated, never listed.** The old ceiling was the largest window anyone had measured, so a
    card with room for more could not ask for it.
    """
    return [n for n in range(limit if limit % 4 == 1 else (limit - 1) // 4 * 4 + 1, 0, -4)]


def base_configuration(target_short_edge_px):
    """Everything off, everything large. The best configuration, if it fits.

    Untiled is the vendored default and the right starting point: tiling is never free, on either
    the seam axis or the time axis. `tensor_offload_device` is the one thing on from the start --
    it is what keeps peak flat as a clip lengthens, and it costs a few per cent of wall time.
    """
    return {
        "target_short_edge_px": target_short_edge_px,
        "batch_size": 1, "chunk_size": 1, "temporal_overlap": 2,
        "vae_encode_tiled": False, "vae_encode_tile_size": 1024, "vae_encode_tile_overlap": 128,
        "vae_decode_tiled": False, "vae_decode_tile_size": 1024, "vae_decode_tile_overlap": 128,
        "blocks_to_swap": 0, "swap_io_components": False,
        "dit_offload_device": "none", "vae_offload_device": "none",
        "tensor_offload_device": "cpu",
    }


def signature(config):
    """What makes two configurations comparable for memory.

    **Tile *size* is deliberately absent, and used to be here.** With it, a row measured at a 1024
    px decode tile said nothing about 768, so the search offered seven tiling options and the model
    could price exactly one -- the tiling axis existed in `candidates` and was dead in `predict`.

    Tile size does not need its own row, because peak in a tiled phase is set by the **tile area**,
    not the frame: a 768 tile is (768/1024)^2 = 0.56 of a 1024 one. That is arithmetic, so it
    belongs in the scaling rather than in the key. What stays in the key is everything that changes
    the *shape* of the computation rather than its size: whether each side tiles at all, and how the
    model is split across devices.

    The chunk is absent for a different reason -- it is measured to cost nothing in VRAM, so a row
    taken at one chunk describes any other.
    """
    # **Which sides tile, and nothing else.** Tile size and block-swapping were both removed from
    # this key, each because keying on them hid options the arithmetic prices perfectly well:
    # `working_area` knows that tiling a side shrinks its working set, `_fixed_gb` knows that
    # swapping moves weights off the card.
    #
    # **Removing the last two was tried and does not work, and that is a finding.** With one pool
    # and one coefficient the options become mutually comparable -- and the model stops
    # reproducing its own anchors: the window-49 measurement of 77.41 GB came back as 92.28, and
    # the window-21 measurement of 31.37 GB as 11.67. A single cost-per-unit-area does not fit
    # this data, which means the rows disagree about it, which means either the area model is
    # incomplete or some rows are wrong.
    #
    # So the key stays, and the consequence is stated rather than hidden: predictions are anchored
    # on measurement *within* a tiling family and are **not comparable across families**. Fixing
    # that needs per-phase peaks -- which phase is the ceiling and what each costs -- and those
    # arrive from the first run on an image carrying `handler/phasewatch.py`.
    return (bool(config["vae_encode_tiled"]), bool(config["vae_decode_tiled"]))


def host_gb(config, source_pixels, output_pixels):
    """What the chunk will hold in system RAM.

    The chunk is free in VRAM and is not free here: it is the input frames and the output canvas,
    both full size, both resident for the whole pass.
    """
    per_frame = (source_pixels * HOST_BYTES_PER_SOURCE_PIXEL
                 + output_pixels * HOST_BYTES_PER_OUTPUT_PIXEL)
    return config["chunk_size"] * per_frame / float(1024 ** 3)


# ── the memory model ────────────────────────────────────────────────────────────────────────────




# ── relief ──────────────────────────────────────────────────────────────────────────────────────

# ── the solve ───────────────────────────────────────────────────────────────────────────────────

#: Re-exported from `planner`, which owns it. A second spelling of a floor is a second thing to
#: forget when a ruling moves it — and one just did (short clips, decision 6).
MIN_WINDOW = planner.MIN_WINDOW

MAX_BLEND_DECODE = planner.MAX_BLEND_DECODE




def window_of(config):
    """The frames the model attends to at once: the smaller of the batch and the chunk."""
    return min(config["batch_size"], config["chunk_size"])


def _chunk_for(frames, window, source_pixels, output_pixels, host_budget_gb):
    """The whole clip, unless the host cannot hold it.

    **The chunk should be the clip.** A chunk boundary is the only unblended seam in the system --
    context frames condition the model and are then discarded -- and the chunk is measured to cost
    nothing in VRAM. The one thing that bounds it is host RAM, where it costs a great deal.
    """
    per_frame = (source_pixels * HOST_BYTES_PER_SOURCE_PIXEL
                 + output_pixels * HOST_BYTES_PER_OUTPUT_PIXEL) / float(1024 ** 3)
    if host_budget_gb is None or per_frame <= 0:
        return frames
    affordable = int(host_budget_gb / per_frame)
    # Never below the window: a chunk under the batch *is* the window, silently.
    return max(window, min(frames, affordable))


def _usable(snapshot):
    free = snapshot.get("vram_free_gb") or snapshot.get("vram_total_gb")
    return max(0.0, (free or 0.0) - VRAM_RESERVE_GB)


# ── the frontier ────────────────────────────────────────────────────────────────────────────────
#
# **What replaced the six-bundle ladder, and then replaced the tile ladder too.** `candidates()`
# above enumerates the whole lever space and prices every point; that is the right shape for a
# search and the wrong shape for a campaign, because most of those 25,872 points are the same
# picture reached a worse way. The frontier is the subset worth arguing about: one row per
# (window, decode grid, encode grid), each already carrying the overlap its own schedule earns.

#: Blocks swapped while probing. **At the top of the ladder the weights term is near zero, so an
#: OOM indicts the activation prediction rather than the swap choice** -- which is the whole point
#: of a probe. Production relaxes it to the least the measured peak allows, buying speed back with
#: proven headroom rather than with a guess.
PROBE_BLOCKS_TO_SWAP = 36


def quality_floor(frames):
    """The floor that applies to *this* clip. `MIN_WINDOW`, or the whole shot when it is shorter.

    **A five-frame clip cannot have a twenty-one-frame window**, and refusing it for not having
    one would refuse work that is not failing. `MIN_WINDOW` is a measured statement about temporal
    context at 4K: below 21 frames the model stops beating a plain enlargement *when 21 frames
    were available*. Where the shot is shorter than the floor, the shot is the floor.
    """
    return min(MIN_WINDOW, window_ceiling(frames))



def window_ceiling(frames):
    """The largest window worth asking for: the smallest 4n+1 **at or above** the shot.

    **Above, not below.** Matching the lattice downward is what the vendored tip recommends and
    what manufactures a runt tail: 192 frames at a window of 189 leaves three frames for a second
    pass, which the loop then pads to five with reflections. 193 takes the clip in one pass and
    pads once, at the end, by a single frame. The model does not care that the window exceeds the
    footage -- the batch is clipped to what exists and padded to the lattice either way.
    """
    frames = max(1, int(frames))
    return frames if frames % 4 == 1 else ((frames - 1) // 4 + 1) * 4 + 1


# ── the pooled model, §9's pricing ──────────────────────────────────────────────────────────────
#
# **Stated as constants because it is a hypothesis, not a fit.** The planner's §9 prices the whole
# frontier from two anchors and one area ratio, at b = 36 where the weights term is near zero:
#
#     P(w, grid) = var(w) x A(grid) / A(untiled)
#
# It is used in preference to `predict()` for probe planning for one reason: `predict()` refuses to
# compare across tiling families, and the frontier is entirely made of such comparisons. Refusing
# is the more honest answer and it is not an answer you can rank a ladder with.
#
# **What is wrong with it, recorded here rather than discovered later.** The two anchors differ in
# three variables at once -- window, block-swap (24 vs 0) and working area (7.9x) -- so the raw
# slope of 1.644 GB/frame charges the window for a 10.54 GB weights difference and an area
# difference that are not the window's doing. Removing only the weights gives 1.268. Both numbers
# are here; `POOLED_SLOPE_GB_PER_FRAME` selects which the frontier believes, and the first probe
# is what settles it.




#: Which rows a measured under-prediction is evidence about, by the phase that produced it. §5.3's
#: third rule. `None` -- an unattributed failure -- has to inflate everything, because a correction
#: whose origin is unknown cannot be scoped.
#:
#: **A sampler miss scopes to the tiling family, not to the failed window column.** The correction
#: attaches to the family's per-frame coefficient, and every window in the family shares that
#: coefficient -- so evidence taken at one window is evidence about all of them. Scoping it to the
#: column made the correction invisible to exactly the rows the walk was about to choose: rows at
#: a narrower window were admitted on their uncorrected price and the §5.3 bound never reached
#: them. The window itself is now handled by the cap below rather than by inflation.
_LOCALITY = {
    "vae_encode": "the same tiling family",
    "vae_decode": "the same tiling family",
    "dit_sample": "the same tiling family",
    "postprocess": "the same window",
    None: "every row",
}


def _lattice_floor(value):
    """The largest 4n+1 at or below `value`."""
    return max(1, 4 * int((max(1.0, float(value)) - 1) // 4) + 1)


def _grid_name(width, height, tile, overlap):
    grid = tiles.tile_grid(width, height, tile, overlap)
    return "{}x{}".format(grid["nx"], grid["ny"])


#: The one place the walk's currency is defined. **Every price below comes out of `planner`'s
#: registry lines and nothing else** — §5.3 rule 4, two tables never meet in one inequality.
#: The pooled frontier that used to price this walk is gone with the rest of the pre-model
#: mechanism (F-14): it ranked rows in a currency the per-phase anchors do not share, which is
#: how an encode failure could be "relieved" by a decode grid.


def plan_of_config(config, job):
    """The `planner` view of a configuration that is about to run, or has just failed.

    The walk needs the failed attempt priced in the *same* table as its candidates, and a config
    dict is the runtime's vocabulary rather than the planner's. This translates once, here, so
    there is one crossing between the two rather than one per comparison.
    """
    src_w, src_h = int(job["source_width"]), int(job["source_height"])
    # **The model plane, asked of `planner`** (F-2026-08-18-28). `job["output_*"]` is
    # `estimator.output_dimensions` — the *delivered* canvas, rounded to even because yuv420p
    # cannot encode an odd dimension. The registry's lines are functions of R, the plane after
    # `DivisiblePad(16)`, and the two differ wherever the canvas is not already a multiple of 16:
    # 1000x1000 at 1080 delivers 1080x1080 and models 1088x1088; 973x512 at 4320 delivers
    # 8210x4320 and models 8224x4320. Repricing the walk on the delivered plane put the
    # correction in a slightly different currency from the plan it was correcting, which is
    # §5.3 rule 4's whole prohibition.
    out_w, out_h = planner.output_plane(src_w, src_h, job["target_short_edge_px"])
    r_mp = out_w * out_h / 1e6
    window = window_of(config)

    dec_tile = config.get("vae_decode_tile_size") if config.get("vae_decode_tiled") else None
    enc_tile = config.get("vae_encode_tile_size") if config.get("vae_encode_tiled") else None
    dec_mp = planner._tile_mp(dec_tile, (src_w, src_h), job["target_short_edge_px"], r_mp)
    enc_mp = planner._tile_mp(enc_tile, (src_w, src_h), job["target_short_edge_px"], r_mp)

    return {
        "w": window,
        "chunk": config.get("chunk_size") or window,
        "enc": _grid_name(out_w, out_h, enc_tile, 128) if enc_tile else "untiled",
        "enc_tile": enc_tile,
        "dec": _grid_name(out_w, out_h, dec_tile, 128) if dec_tile else "untiled",
        "dec_tile": dec_tile,
        "tile_quality": config.get("tile_quality", "default"),
        "prices": {
            "vae_encode": round(planner.encode_price(enc_mp, r_mp, window), 2),
            "dit_sample": round(planner.dit_price(r_mp, window), 2),
            "vae_decode": round(planner.decode_price(dec_mp), 2),
            "postprocess": round(planner.postprocess_price(r_mp, window), 2),
        },
    }


def config_of_plan(answer, job, template=None):
    """A runnable configuration from a `planner` answer.

    `template` is the configuration this one replaces, so free levers the walk does not price —
    `swap_io_components` above all — are **carried rather than re-derived**. Without that, a walk
    that spends the free lever and then re-plans silently gives it back, and the next step spends
    it again: apply, discard, apply, discard, which is exactly what the window-109 replay showed
    over eight steps.
    """
    config = dict(template) if template else base_configuration(job["target_short_edge_px"])
    window = answer["w"]
    config.update(
        batch_size=window,
        chunk_size=max(answer.get("chunk") or window, window),
        temporal_overlap=answer.get("v", 0),
        blocks_to_swap=answer.get("blocks_to_swap", PROBE_BLOCKS_TO_SWAP),
        vae_decode_tiled=answer.get("dec_tile") is not None,
        vae_encode_tiled=answer.get("enc_tile") is not None,
        vae_decode_tile_overlap=planner.LAP,
        vae_encode_tile_overlap=planner.LAP,
    )
    if answer.get("dec_tile"):
        config["vae_decode_tile_size"] = answer["dec_tile"]
    if answer.get("enc_tile"):
        config["vae_encode_tile_size"] = answer["enc_tile"]
    config["dit_offload_device"] = "cpu" if config["blocks_to_swap"] else "none"
    config["target_short_edge_px"] = job["target_short_edge_px"]
    # **Every configuration carries a name**, because a dozen call sites read `plan["name"]` for
    # a banner, a ledger row or a warning, and a plan without one turns a re-plan into a KeyError
    # at exactly the moment something has already gone wrong.
    config.setdefault("name", "planned")
    return config


def row_of_plan(answer, job, template=None):
    """The `(row)` half of the walk's return: a config plus the figures callers report."""
    return {
        "config": config_of_plan(answer, job, template),
        "window": answer["w"],
        "chunk": answer.get("chunk"),
        "overlap": answer.get("v", 0),
        "passes": answer.get("passes"),
        "decode_grid": answer.get("dec") or "untiled",
        "decode_tile": answer.get("dec_tile"),
        "encode_grid": answer.get("enc") or "untiled",
        "encode_tile": answer.get("enc_tile"),
        "predicted_peak_vram_gb": round(max(answer["prices"].values()), 2),
        "prices": answer["prices"],
        "binder": answer.get("binder"),
        "registry_version": answer.get("registry_version"),
        "lever": answer.get("lever"),
    }


def next_after_oom(job, snapshot, failed_config, failed_prediction_gb,
                   phase=None, shortfall=None, relax_swap=False):
    """The configuration to attempt after a **confirmed** OOM. `(row, why)`, row `None` at the end.

    **One lever, priced from the measurement, then the whole chunk from phase 1.** An OOM is not
    a cue to search: PyTorch reports what was allocated and what the failed request wanted, and
    their sum is a direct lower bound on the true peak of *that phase* at *that configuration*.
    The phase names which registry line was wrong, the bound says by how much, and
    `planner.correct` puts exactly that one correction back through the same six steps.

    Everything the failing phase did not implicate is **held**, not re-planned. A sampler miss
    says nothing about the decode grid, and re-deriving the grid from the closed form would move
    a second lever — sometimes upward, since a probe may be running deliberately coarser than the
    formula's own pick. That was the confusion the phase tap was added to end, and holding is how
    it stays ended.

    `relax_swap` is accepted for signature compatibility and does nothing: `b = 36` everywhere
    until the b<36 relaxation term is measured (ruled 2026-08-18, one two-b run outstanding).
    """
    usable = _usable(snapshot)
    src = (int(job["source_width"]), int(job["source_height"]))
    target = job["target_short_edge_px"]
    frames = int(job.get("estimated_frames") or 0) or None
    host_ram = snapshot.get("host_ram_gb")
    shortfall = shortfall or {}
    needed = shortfall.get("needed_at_least_gb")

    failed = plan_of_config(failed_config, job) if failed_config is not None else None
    advisory = failed_prediction_gb
    repriced = max(failed["prices"].values()) if failed else None

    if failed is None or frames is None:
        return None, {
            "bound_gb": round(usable, 2), "phase": phase,
            "basis": "nothing to correct: no failed configuration or no frame count",
            "exited_sideways": False, "carried_free_levers": None,
            "candidates_under_bound": 0, "advisory_prediction_gb": advisory,
            "repriced_prediction_gb": repriced, "window_cap": None,
            "registry_version": planner.REGISTRY_VERSION,
        }

    # **The bound, measured rather than inferred.** Where the message carried no figure the
    # phase's own repriced peak is the floor of what it needed — it failed at that price, so it
    # needs at least that much — which keeps the correction in one currency either way.
    needed = float(needed) if needed else max(repriced or 0.0, usable) * 1.0001
    answer = planner.correct(src, frames, target, usable, phase, needed, failed,
                             host_ram_gb=host_ram,
                             tile_quality=failed.get("tile_quality", "default"))

    lever = planner.PHASE_LEVER.get(phase, "window")
    basis = "measured from the message: {} corrected against {:.2f} GiB needed at w{}".format(
        lever.replace("_", " "), needed, failed["w"])
    why = {
        "bound_gb": round(usable, 2), "phase": phase, "lever": lever, "basis": basis,
        "exited_sideways": lever in ("decode_grid", "encode_grid"),
        "carried_free_levers": ("swap_io_components"
                                if failed_config.get("swap_io_components") else None),
        "advisory_prediction_gb": advisory, "repriced_prediction_gb": repriced,
        "window_cap": answer.get("w") if lever == "window" else None,
        "correction_applies_to": _LOCALITY.get(phase, "every row"),
        "correction_factor": round(needed / repriced, 3) if repriced else 1.0,
        "registry_version": planner.REGISTRY_VERSION,
    }
    if answer.get("action") != "plan":
        why["candidates_under_bound"] = 0
        why["reason"] = answer.get("reason")
        why["options"] = answer.get("options")
        return None, why

    why["candidates_under_bound"] = 1
    why["rationale"] = planner.rationale_line(answer)
    return row_of_plan(answer, job, template=failed_config), why


def preflight(config, frames=None):
    """§4's asserts. Returns `(normalised_config, warnings)`; raises `ValueError` on the fatal one.

    **These are the mistakes that are invisible rather than loud**, which is why they are checked
    rather than trusted:

      1. `batch > chunk` is silently the chunk. Four runs once measured the same nine-frame window
         while believing they were measuring four different batches. Normalised here, and said.
      2. The **effective** window is what the model attends to and what a row must record --
         `min(batch, chunk)` raised to the lattice, because a batch off the lattice is padded up
         before it reaches the model. A row keyed on the requested batch is keyed on a number that
         never happened.
      3. Sub-8px changes to a tile or an overlap are no-ops: the grid is laid out in latent space
         and both are floor-divided by 8. Warned rather than corrected, because silently moving a
         caller's number is how a measurement stops being of what was asked for.
      4. `prepend_frames > 0` is fatal. Both places the vendored code strips prepended frames are
         bypassed on this worker's path, so enabling it ships reflected frames to the client as
         content. A hard error until the strip is wired.
    """
    config = dict(config)
    warnings = []

    if config["batch_size"] > config["chunk_size"]:
        warnings.append(
            "batch_size {} exceeds chunk_size {}; the temporal window is min(batch, chunk) = {}, "
            "so the batch has been normalised to it".format(
                config["batch_size"], config["chunk_size"], config["chunk_size"]))
        config["batch_size"] = config["chunk_size"]

    window = window_of(config)
    config["effective_window"] = window if window % 4 == 1 else ((window - 1) // 4 + 1) * 4 + 1

    # **Block-swapping needs somewhere to swap to, and the vendored code refuses rather than
    # ignores.** `blocks_to_swap=36` with `dit_offload_device='none'` raises before a single frame
    # is read -- it is not a silent no-op, it is a lost run. Every path that *builds* a
    # configuration already pairs the two; the forced-lever path sets one field at a time by
    # design and so could set the flag without its device. Measured: 190.9 s of paid GPU and a
    # refusal, on the first probe of the campaign this file exists to serve.
    #
    # Normalised rather than refused, because the caller who asked for 36 blocks asked for block
    # swapping, and "you forgot the device" is not a question worth a round trip.
    if int(config.get("blocks_to_swap") or 0) > 0 and config.get("dit_offload_device") in (
            None, "", "none"):
        config["dit_offload_device"] = "cpu"
        warnings.append(
            "blocks_to_swap={} requires dit_offload_device; it was 'none', which the vendored "
            "code refuses outright, so it has been set to 'cpu'".format(config["blocks_to_swap"]))

    for side in ("encode", "decode"):
        if not config.get("vae_{}_tiled".format(side)):
            continue
        for what in ("tile_size", "tile_overlap"):
            key = "vae_{}_{}".format(side, what)
            value = int(config.get(key) or 0)
            if value % tiles.SCALE_FACTOR:
                warnings.append(
                    "{} = {} is not a multiple of {}; the grid is laid out in latent space so it "
                    "acts as {}".format(key, value, tiles.SCALE_FACTOR,
                                        value - value % tiles.SCALE_FACTOR))

    if int(config.get("prepend_frames") or 0) > 0:
        raise ValueError(
            "prepend_frames is {} and this worker does not strip prepended frames: both vendored "
            "removal sites are bypassed on our path, so the reflected frames would be delivered "
            "as content. Refused until the strip is wired.".format(config["prepend_frames"]))

    if frames and config["chunk_size"] > frames:
        config["chunk_size"] = frames
    return config, warnings
