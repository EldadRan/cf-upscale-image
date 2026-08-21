"""Producing the artefacts `derive[]` asks for.

**It is CF's list, not this worker's**: you produce what is asked and nothing else. The
vocabulary, the nesting and the strictness are the platform's (`CF_media_worker` §5).

Two rules govern everything here:

  **Derives are of the output, not the source** — at that file's resolution, **never upscaled**.
  A poster of a 720p output is 720p.

  **`derived[]` nests inside the output it came from**, and is absent when nothing was asked for
  or nothing succeeded. A parentless derive is not expressible, and inventing a place for one
  breaks the nesting every consumer reads.

**The master is written before any of this is attempted**, and that is the order rather than a
preference: a failed derive is recoverable from the master, while a master that was never written
costs hours of GPU to reproduce. So a derive that fails leaves the expensive artefact intact, and
`report()` says which artefacts landed so a recovery knows what it is completing rather than
redoing.
"""

import os

import crops
import encoder
import keys


def poster_frame_index(at_fraction, frame_count):
    """`round(at_fraction × (frame_count − 1))`.

    0 is the first frame, 1 the last, and **a single-frame video degenerates to 0 rather than
    erroring** — which is also the whole of the image path's poster behaviour, reached without a
    branch worth naming.
    """
    return int(round(at_fraction * max(0, frame_count - 1)))


def build(spec, master_path, source_path, workdir, frame_count, scale, warn=None):
    """Produce every requested role. Returns a list of entries, one per artefact written.

    Each entry carries the local `path` and the fields CF's `derived[]` needs; the caller
    uploads them and swaps `path` for the key. **A role that fails is reported through `warn` and
    omitted — it never takes the master down with it, and it never takes the other roles down
    either.**

    That sentence was in this docstring for two months describing behaviour the code did not have.
    There was no `try` here at all, and the caller caught only `WorkerError` — so an ffmpeg exit,
    a PIL error or an `OSError` from any role escaped as an `internal` failure for a job whose
    master was **already written and uploaded**. CF would have received a refusal for work that
    had succeeded, over a poster. And because the loop had no per-role isolation, one failing role
    discarded every role after it.

    Nothing ever failed here, which is why nobody found it: a guarantee that is never exercised
    looks identical to one that works.
    """
    entries = []
    for item in spec:
        role = item["role"]
        try:
            if role == "poster":
                entries.append(_poster(item, master_path, workdir, frame_count))
            elif role == "proxy":
                entries.append(_proxy(item, master_path, workdir))
            elif role == "crop":
                entries.extend(_crops(item, master_path, source_path, workdir, frame_count, scale,
                                      warn=warn))
        except Exception as exc:  # noqa: BLE001 — the master outranks every derive, see above
            if warn is not None:
                warn("derive '{}' failed and was omitted: {}: {}".format(
                    role, type(exc).__name__, str(exc)[:200]))
    return entries


def _poster(item, master_path, workdir, frame_count):
    """The display role. **Lossy WebP, and that is CF's requirement rather than a choice here.**

    Handoff §4 names it: "poster — round(at_fraction x (frame_count - 1)) of the file this call
    produced, WebP." It is inherited from CF_media_worker, whose response example already emits
    `poster.webp` with `content_type: image/webp`, and the handoff tells this worker to follow
    that worker tightly on derives — so the format came along with the role.

    The reasoning, confirmed by CF on 2026-08-15: a poster is a display thumbnail, lossy WebP is
    meaningfully smaller than PNG at the same perceived quality on photographic content, and one
    format across both workers means one delivery path.

    **`derive[]` is CF's list and no caller chooses**, so this is not a default that could be
    overridden -- there is nothing to support. Recorded here because it was proposed for change on
    2026-08-15 by someone who had read the still master's PNG rule and generalised it one role too
    far. The still master is ours to decide; this is not. Contrast `_crops`, where CF required the
    opposite for a stated reason.
    """
    name = keys.DERIVE["poster"]
    path = os.path.join(workdir, name)
    index = poster_frame_index(item.get("at_fraction", 0.25), frame_count)
    encoder.extract_poster(master_path, path, index, fps=None)
    return {
        "role": "poster", "name": name, "path": path,
        "content_type": keys.content_type(name),
        "frame_index": index,
        "bytes": os.path.getsize(path),
    }


def _proxy(item, master_path, workdir):
    name = keys.DERIVE["proxy"]
    path = os.path.join(workdir, name)
    encoder.encode_proxy(master_path, path, max_duration_s=item.get("max_duration_s"))
    return {
        "role": "proxy", "name": name, "path": path,
        "content_type": keys.content_type(name),
        "bytes": os.path.getsize(path),
    }


def _crops(item, master_path, source_path, workdir, frame_count, scale, warn=None):
    """The evidence role. One file per crop, source beside output, lossless.

    **Lossless because CF required it, and for a reason worth keeping** (2026-08-15): "poster is a
    thumbnail and lossy WebP suits it; a crop is evidence, and a lossy encode adds artefacts to an
    artefact-detection image." The two roles take opposite formats deliberately, so neither can be
    read as the house style.


    Both frames come out of the finished files rather than being held from the streaming loop —
    two single-frame decodes, at the same cost class as a poster, against holding full frames in
    memory on a worker whose entire design is about not doing that.
    """
    import numpy as np
    from PIL import Image

    requested = int(item.get("count", crops.DEFAULT_COUNT))
    select = item.get("select", "detail")
    index = poster_frame_index(item.get("at_fraction", 0.25), frame_count)

    source_png = encoder.extract_frame_png(
        source_path, os.path.join(workdir, "_crop_src.png"), index)
    master_png = encoder.extract_frame_png(
        master_path, os.path.join(workdir, "_crop_out.png"), index)
    source_rgb = np.asarray(Image.open(source_png).convert("RGB"))
    output_rgb = np.asarray(Image.open(master_png).convert("RGB"))

    # The matte decides where the evidence is, when there is one.
    source_rgba = np.asarray(Image.open(source_png))
    layer_alpha = source_rgba[..., 3] if source_rgba.ndim == 3 and source_rgba.shape[2] == 4 \
        else None
    regions = crops.select_regions(source_rgb, requested, scale, select=select,
                                   alpha=layer_alpha)

    # **No silent caps.** `count` is bounded by geometry as well as by MAX_COUNT: non-overlapping
    # windows have to fit in the frame, and a small source cannot hold many. Producing two where
    # three were asked for is correct; reporting three, or saying nothing, is not.
    if warn is not None and len(regions) < min(requested, crops.MAX_COUNT):
        warn("crop: produced {} of {} requested — non-overlapping {}px regions do not fit in a "
             "{}×{} source".format(len(regions), requested,
                                   int(round(crops.CROP_PX / float(scale))),
                                   source_rgb.shape[1], source_rgb.shape[0]))

    entries = []
    for ordinal, region in enumerate(regions):
        pair, coordinates = crops.render_pair(source_rgb, output_rgb, region, scale)
        name = keys.crop_name(ordinal)
        path = os.path.join(workdir, name)
        crops.write_lossless_webp(pair, path)
        entries.append({
            "role": "crop", "name": name, "path": path,
            "content_type": keys.content_type(name),
            "ordinal": ordinal,
            "frame_index": index,
            "select": select,
            # Coordinates ride here and never in the filename: names stay deterministic so
            # recovery by listing works, and a reviewer still needs to find the region in the
            # master. CF's call.
            "region": coordinates,
            # A crop must never be delivered through a resizing ladder step — interpolated detail
            # handed to someone judging whether detail was invented (`CF_storage` 1.33.0). Stated
            # on the artefact so the property travels with it rather than living only in a spec.
            "original_only": True,
            "bytes": os.path.getsize(path),
        })

    for temporary in (source_png, master_png):
        try:
            os.remove(temporary)
        except OSError:
            pass
    return entries
