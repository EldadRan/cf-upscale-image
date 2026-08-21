"""Where the VAE tiler puts its tiles, and what that costs.

**Mirrors `attn_video_vae.tiled_encode` / `tiled_decode` at the pinned commit**, not the obvious
formula. Three details separate the two and every one changes the answer:

  1. the grid is laid out in **latent space** (spatial scale factor 8). Tile and overlap are
     floor-divided by 8 before anything else, so a change to either smaller than 8px does nothing;
  2. edge tiles are **clipped**, not shifted inward, so `tiles x T^2` overstates the work;
  3. a trailing tile whose remaining extent is no larger than the overlap is **skipped** -- the
     same rule the temporal batch loop uses to drop a runt batch.

Verified against the only published figure that did not come from this repo: Numz's walkthrough
tiles a 2742x4096 image at 1024/128 and reports fifteen tiles.

**This lives in the image because the solver needs it.** Choosing a tile is a planning decision --
peak VRAM in a tiled phase is set by the tile, not by the frame -- so the geometry cannot sit in a
laptop script. `scripts/simulate_schedule.py` imports it from here rather than keeping a copy,
because two implementations of the same arithmetic is one that silently disagrees.

**Encode and decode are different frames.** Encode tiles the input, decode tiles the output. At
1080p in and 4K out the same tile size gives a 2x2 grid on one side and 5x3 on the other, so a
single number applied to both is two different decisions wearing one name.
"""

import math

SCALE_FACTOR = 8


def tile_spans(total, tile, overlap):
    """Where the tiles actually land on one axis, in output pixels.

    Returns `(spans, effective_overlap, stride)`. `spans` are half-open `(start, end)` pairs,
    clipped at the far edge, which is why the last one is usually short.
    """
    if tile <= overlap:
        raise ValueError("overlap must be smaller than the tile")
    lat = -(-total // SCALE_FACTOR)
    tile_lat = max(1, tile // SCALE_FACTOR)
    over_lat = max(0, min(overlap // SCALE_FACTOR, tile_lat - 1))
    stride = max(1, tile_lat - over_lat)

    spans, start = [], 0
    while start < lat:
        end = min(start + tile_lat, lat)
        if start > 0 and (end - start) <= over_lat:
            break
        spans.append((start * SCALE_FACTOR, min(end * SCALE_FACTOR, total)))
        if end >= lat:
            break
        start += stride
    return (spans or [(0, total)]), over_lat * SCALE_FACTOR, stride * SCALE_FACTOR


def _multiply_covered(spans):
    """Length covered by two or more spans — the blend, measured rather than assumed."""
    return sum(max(0, spans[i - 1][1] - spans[i][0]) for i in range(1, len(spans)))


def tile_grid(width, height, tile, overlap):
    """The grid the vendored tiler will lay down, and what it costs.

    `blended_fraction` is computed by inclusion-exclusion rather than by adding the two axes,
    which would double-count the corners where a vertical and a horizontal overlap cross.
    """
    xs, eff_overlap, stride = tile_spans(width, tile, overlap)
    ys, _, _ = tile_spans(height, tile, overlap)
    native = float(width * height)

    processed = sum((x1 - x0) * (y1 - y0) for (y0, y1) in ys for (x0, x1) in xs)
    lap_x, lap_y = _multiply_covered(xs), _multiply_covered(ys)
    blended = (lap_x * height + lap_y * width - lap_x * lap_y) / native

    return {
        "tile": tile, "overlap": overlap,
        "effective_overlap": eff_overlap, "stride": stride,
        "nx": len(xs), "ny": len(ys), "tiles": len(xs) * len(ys),
        "spans_x": xs, "spans_y": ys,
        "processed_pixels": processed,
        "overhead": processed / native,
        "seam_px": (len(xs) - 1) * height + (len(ys) - 1) * width,
        "blended_fraction": blended,
        # Proxy for peak in a tiled phase: the working set is one full tile, not one frame.
        # This is the axis a measured phase-3 peak will calibrate, and the only one that argues
        # for a small tile.
        "tile_pixels": min(tile, width) * min(tile, height),
    }


def largest_tile_that_fits(width, height, overlap, max_tile_pixels, step=8):
    """The biggest legal tile whose working set stays inside a budget.

    **This is the whole tiling policy, once the sawtooth turned out not to exist.** Overhead and
    seam both improve monotonically with tile size, so there is nothing to trade off and nothing
    to search: take the largest tile the memory allows. `max_tile_pixels` is the caller's, because
    this file has no business guessing it.
    """
    best = None
    upper = max(width, height)
    for tile in range(step, upper + step, step):
        if tile <= overlap:
            continue
        grid = tile_grid(width, height, tile, overlap)
        if grid["tile_pixels"] > max_tile_pixels:
            break
        best = grid
    return best


#: Sizes worth printing. Conventional powers of two plus the exact-fit values, and **nothing here
#: is a constraint** — the vendored code accepts any tile, floors it to a multiple of 8, and gets
#: on with it. The ladder our rungs use (1024 / 768 / 512 / 384) is inherited convention.
TILE_SIZES = (2048, 1536, 1024, 768, 512, 384, 256)


def survey_tiles(width, height, overlap, tile_sizes=TILE_SIZES):
    """Every listed tile size at one overlap, largest first — which is also best first."""
    return [tile_grid(width, height, t, overlap) for t in sorted(tile_sizes, reverse=True)
            if t > overlap]


def grid_ladder(width, height, overlap, step=8, max_tile=None):
    """Every distinct grid this plane can be cut into, coarsest first, each at its **minimal** tile.

    **Grid-first, not tile-first.** A tile ladder of round numbers is a guess at the geometry: at
    4K a 1024 px tile and a 1056 px tile both give 4x3, but the larger one wastes nothing and the
    smaller one is simply a worse way to reach the same grid. What the memory model cares about is
    the tile working set and what quality cares about is the seam count, and both follow from the
    grid -- so the grid is the axis, and each rung carries the **smallest** tile that still
    achieves it.

    Smallest, because within one grid the seam count is fixed and the tile is pure cost: every
    larger tile holds a bigger working set for an identical picture. So the minimal tile is the
    cheapest way to buy that seam count, and a ladder of round numbers -- 1024, 768, 512 -- is
    just a ladder that overpays. At 4K a 1056 px tile and a 1024 px tile are both 4x3; only one of
    them is the tightest fit.

    The 1x1 rung is the untiled case, included so callers walk one ladder rather than two.
    """
    ladder = []
    upper = min(max(width, height), max_tile or max(width, height))
    # **A tile no larger than twice the overlap is degenerate and is not a rung.** Below that the
    # stride falls to or under the overlap, three tiles can cover one pixel, and the two-tile
    # cross-fade the vendored ramp implements stops describing what happens. `blended_fraction`
    # says so loudly -- its inclusion-exclusion goes *negative* there, which is how this was
    # found: those rows were passing a `blend <= 0.24` cap by being less than zero.
    floor_tile = 2 * overlap
    # Walked downward, so grids get finer as we go and each new one appends. A tile that lands on
    # the grid already at the end of the ladder replaces it -- being smaller, it is the better way
    # to reach the same grid -- which leaves every rung holding its minimal tile.
    for tile in range(upper - upper % step, step - 1, -step):
        if tile <= floor_tile:
            break
        grid = tile_grid(width, height, tile, overlap)
        key = (grid["nx"], grid["ny"])
        if ladder and (ladder[-1]["nx"], ladder[-1]["ny"]) == key:
            ladder[-1] = grid
        elif not any((g["nx"], g["ny"]) == key for g in ladder):
            ladder.append(grid)
    return ladder
