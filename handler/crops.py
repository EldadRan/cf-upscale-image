"""The `crop` derive: 1:1 evidence about what the model invented.

**Neither `poster` nor `proxy` can show fabrication.** A downscale and a full-frame still both
hide it, and this model at 3.84× redrew a QR code's modules and haloed letterforms. What shows it
is a 1:1 crop at native output resolution with the same region of the source beside it.

Agreed with CF (`bf4471c`). The rules below are the platform's, and four of them are corrections
CF made to the original proposal — each one closing a way this artefact could have looked fine
while being useless:

  **Lossless WebP.** `poster` is a thumbnail and lossy suits it; a crop is *evidence*. Lossy
  compression puts artefacts into an artefact-detection image.

  **The source half is nearest-neighbour only.** Any smooth interpolation invents plausible
  pixels of its own, which makes the model's invention look less anomalous — the comparison
  flattering exactly the thing it exists to expose.

  **One file, not two.** A comparison someone has to assemble by hand is one nobody makes, and
  CF's delivery is per-object ticketed, so two files means two tickets to view one comparison.

  **Regions must not overlap.** Highest-energy selection will otherwise return three views of one
  text block and report three samples.

And one that governs delivery rather than this module: **a crop is original-only.** CF's delivery
ladder resizes on read, and a ticket naming a ladder step is refused at mint for this role
(`CF_storage` 1.33.0) — without that refusal a reviewer would be handed interpolated detail while
judging whether detail was invented. The test for membership is whether a resize changes what the
object *asserts*, not whether it still looks acceptable, and a crop is the strictest case there is.

**Coordinates go in the manifest, never the filename.** Names stay deterministic so recovery by
listing works, and a reviewer still needs to find the region in the master.

## What this does not cover, stated rather than assumed

**Every crop is one frame, so nothing here sees a temporal seam** — a chunk boundary, a flicker
between batches, a discontinuity where two chunks were blended. That is the failure mode a video
model has that a still model does not, and neither `poster` nor `crop` addresses it. Recorded by
CF as a limit rather than solved.
"""

import numpy as np

#: Output-resolution size of each crop. Square, so orientation does not change what a reviewer
#: is comparing between jobs.
CROP_PX = 512

DEFAULT_COUNT = 3
MAX_COUNT = 8
SELECT_MODES = ("detail", "centre", "spread")


def _luma(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _energy_map(gray):
    """High-frequency energy, as a 4-neighbour Laplacian.

    One pass over one frame — the right cost for choosing where to look, and CF's reason for
    selecting on the **source**: that is where invention is judged. Choosing on the *output*
    would select the regions the model made busiest, which is the same thing as selecting for
    whatever it invented and calling that evidence.
    """
    gray = gray.astype(np.float32)
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        4.0 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    )
    return np.abs(lap)


def _overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def select_regions(source_rgb, count, scale, select="detail", alpha=None):
    """Regions in **source** coordinates, non-overlapping, best first.

    `scale` is output ÷ source, so a `CROP_PX` window in the output is `CROP_PX / scale` in the
    source. On a downscale that window is smaller than the crop it produces, which is correct:
    the crop is always `CROP_PX` of *output*, and the source half is whatever fed it.

    **`alpha` is not optional decoration on a cutout.** A layer separated out of a larger image
    keeps real colour in the RGB behind its matte — measured on three Qwen layers, the fully
    transparent regions average 68/255 rather than black — so `detail` scoring, which reads only
    luma, happily picks the busiest *masked-out* background. Run on a real layer that is 90%
    transparent it chose a window with **7% subject coverage**: a crop of something the caller
    will never see, offered as evidence of what the upscale did.

    Where an alpha channel exists the energy is weighted by it, so the regions come from the part
    that survives compositing.
    """
    if select not in SELECT_MODES:
        raise ValueError("unknown select mode {!r}".format(select))
    count = max(1, min(int(count), MAX_COUNT))
    height, width = source_rgb.shape[:2]

    window = max(8, int(round(CROP_PX / float(scale))))
    window = min(window, width, height)
    if window <= 0:
        return []

    if select == "centre":
        candidates = [((width - window) // 2, (height - window) // 2)]
    else:
        step = max(1, window // 2)
        candidates = [(x, y)
                      for y in range(0, max(1, height - window + 1), step)
                      for x in range(0, max(1, width - window + 1), step)]

    if select == "detail":
        energy = _energy_map(_luma(source_rgb))
        if alpha is not None:
            # Weighted, not masked. A hard mask would score a window that clips the subject's edge
            # the same as one centred on it, and the edge is exactly what a cutout is judged on.
            weight = np.asarray(alpha).astype(np.float32) / 255.0
            if weight.shape == energy.shape:
                energy = energy * weight
        # Integral image, so scoring every candidate window is one subtraction each rather than
        # a slice-and-sum. It matters: a 4K frame at step 256 is a few thousand candidates.
        integral = energy.cumsum(0).cumsum(1)
        integral = np.pad(integral, ((1, 0), (1, 0)))

        def score(position):
            x, y = position
            x2, y2 = min(x + window, width), min(y + window, height)
            return float(integral[y2, x2] - integral[y, x2]
                         - integral[y2, x] + integral[y, x])

        candidates.sort(key=score, reverse=True)
    elif select == "spread":
        candidates.sort(key=lambda p: (p[1], p[0]))
        if len(candidates) > count:
            stride = len(candidates) / float(count)
            candidates = [candidates[int(i * stride)] for i in range(count)]

    chosen = []
    for x, y in candidates:
        region = (x, y, window, window)
        # **Non-overlap, enforced rather than hoped for.** Without this the three highest-energy
        # windows on a page of text are three views of the same paragraph, reported as three
        # samples.
        if any(_overlaps(region, taken) for taken in chosen):
            continue
        # **A window with nothing in it is not a sample.** On a cutout, non-overlap decides the
        # later regions rather than scoring does — two 320 px windows fill an 864x480 layer, so
        # the second is whatever is left, and on a 90%-transparent layer that is empty backdrop.
        # Weighting the energy cannot fix a choice that was forced; declining to emit it can.
        # Fewer crops with a warning beats padding the count with pictures of nothing.
        if alpha is not None and not _holds_subject(alpha, region):
            continue
        chosen.append(region)
        if len(chosen) == count:
            break
    return chosen


#: A crop has to be at least this much subject to be worth returning. Low on purpose: the point
#: is to reject windows that are entirely masked-out backdrop, not to demand a well-composed one.
MIN_SUBJECT_FRACTION = 0.02


def _holds_subject(alpha, region):
    """Whether a source-coordinate region contains any of the matte at all."""
    x, y, w, h = region
    window = np.asarray(alpha)[y:y + h, x:x + w]
    return window.size > 0 and float((window > 10).mean()) >= MIN_SUBJECT_FRACTION


def _nearest_upscale(block, out_h, out_w):
    """Nearest-neighbour only. **Smooth interpolation would invent plausible pixels of its own**,
    flattering the model's invention by making it look less anomalous than it is."""
    src_h, src_w = block.shape[:2]
    rows = (np.arange(out_h) * src_h // max(1, out_h)).clip(0, src_h - 1)
    cols = (np.arange(out_w) * src_w // max(1, out_w)).clip(0, src_w - 1)
    return block[rows][:, cols]


def render_pair(source_rgb, output_rgb, region, scale):
    """Source region (nearest-upscaled) beside the output region (native). Returns an array.

    Left is the source, right is the output — a fixed order, so a reviewer comparing two jobs is
    never comparing two conventions.

    **The output rectangle is derived from the source rectangle, never taken as a fixed size.**
    An earlier version cut a fixed `CROP_PX` square out of the output while the source window had
    been clamped smaller by the frame edge, so the two halves showed *different areas* with the
    source stretched to fit — a comparison that looks entirely plausible and is worthless. The
    two rectangles must describe the same region of the picture or the artefact asserts something
    it has not shown.

    So: map source → output by `scale`, clip that to the output frame, and map whatever survives
    back to source coordinates. Both halves then cover the same region by construction, whatever
    the clipping did.
    """
    x, y, w, h = region
    out_h_total, out_w_total = output_rgb.shape[:2]

    ox, oy = int(round(x * scale)), int(round(y * scale))
    ow, oh = int(round(w * scale)), int(round(h * scale))
    # Clip to the output frame, keeping the rectangle inside it rather than shrinking from the
    # far edge — a crop at the frame border should still be full size where the frame allows.
    ox = max(0, min(ox, max(0, out_w_total - ow)))
    oy = max(0, min(oy, max(0, out_h_total - oh)))
    ow = min(ow, out_w_total - ox)
    oh = min(oh, out_h_total - oy)
    out_block = output_rgb[oy:oy + oh, ox:ox + ow]

    # **Sample the source on the output's own global grid**, rather than resampling the extracted
    # sub-block on a grid of its own. The two are equivalent only at integer scale; at CF's
    # measured 3.84× they disagree on which source pixel each output pixel comes from, which
    # shows up as a fifth of the comparison differing at zero displacement. The region was right
    # and the sampling was not — a distinction invisible on a photograph and obvious on noise.
    #
    # Still nearest-neighbour only, which is the rule that matters: no value here is interpolated.
    src_h, src_w = source_rgb.shape[:2]
    rows = (np.arange(oy, oy + oh) * src_h // out_h_total).clip(0, src_h - 1)
    cols = (np.arange(ox, ox + ow) * src_w // out_w_total).clip(0, src_w - 1)
    left = source_rgb[rows][:, cols]

    # Report the source rectangle the sampling actually covers, so the manifest's coordinates
    # locate the region in the source as well as in the master.
    x, y = int(cols[0]), int(rows[0])
    w, h = int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)

    target_h, target_w = out_block.shape[:2]

    # A one-pixel divider, so the seam is unambiguous when the two halves happen to match.
    divider = np.full((target_h, 1, 3), 255, dtype=output_rgb.dtype)
    return np.concatenate([left, divider, out_block], axis=1), {
        "source_x": int(x), "source_y": int(y),
        "source_w": int(w), "source_h": int(h),
        "output_x": int(ox), "output_y": int(oy),
        "output_w": int(target_w), "output_h": int(target_h),
    }


def write_lossless_webp(array, path, xmp=None):
    """**Lossless**, because this is evidence. Lossy compression would put artefacts into an
    artefact-detection image, which is the one place they cannot be tolerated.

    `xmp` is the delivered file's identity (`derives._xmp_packet`) and rides in a metadata chunk,
    so stamping a crop with whose job it came from costs the evidence nothing.
    """
    from PIL import Image

    options = {"format": "WEBP", "lossless": True, "quality": 100, "method": 4}
    if xmp:
        options["xmp"] = xmp
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path, **options)
    return path
