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

import json
import os
import struct

import crops
import encoder
import keys


#: **The identity a delivered derive carries** (`api.md` §6, ruled item 4; `contract.md` §3 is the
#: rule it answers to). Until this existed, `identity_tags` appeared in this module zero times: the
#: master was stamped and every artefact beside it was anonymous, so a crop or a proxy that left
#: its prefix — copied into a review folder, attached to a ticket, downloaded and renamed — could
#: not say whose job produced it.
#:
#: **Identity only, exactly as the master's tags are.** These files are delivered. Timings,
#: hardware, tiling configuration and anything resembling a credential stay in the manifest and
#: the diagnostics bundle. A recovery aid and never a source of truth.
#:
#: **The role is stamped as well as the id**, which the master does not need and a derive does: a
#: `poster` and a `crop` from the same request differ in what they are for, and a file found alone
#: cannot be told apart by its pixels.
IDENTITY_XMP_NAMESPACE = "https://cf/ns/upscale/1.0/"


def _xmp_packet(tags):
    """The `cf_*` identity as an XMP packet.

    **XMP rather than EXIF because the keys are ours.** EXIF has a fixed tag table and no room for
    a name this project invented; XMP carries arbitrary properties under a namespace, which is
    what a set of `cf_`-prefixed facts needs. WebP carries XMP natively, so the same packet serves
    both roles that write one.
    """
    properties = "".join(
        ' cf:{}="{}"'.format(key, _xml_escape(str(value)))
        for key, value in sorted(tags.items()) if value is not None)
    return (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:cf="{}"{}/>'
        '</rdf:RDF></x:xmpmeta>'
        '<?xpacket end="w"?>'.format(IDENTITY_XMP_NAMESPACE, properties)).encode("utf-8")


def _xml_escape(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


#: The VP8X flag bit that says "this file carries an XMP chunk". Bit 2 of the flags byte, per the
#: WebP container specification.
_VP8X_XMP_FLAG = 0x04


def _webp_chunks(blob):
    """Walk a RIFF/WEBP container. Chunks are padded to an even length; the pad is not payload."""
    offset = 12
    while offset + 8 <= len(blob):
        fourcc = blob[offset:offset + 4]
        size = struct.unpack("<I", blob[offset + 4:offset + 8])[0]
        yield fourcc, blob[offset + 8:offset + 8 + size]
        offset += 8 + size + (size & 1)


def _webp_chunks_with_end(blob):
    """Every chunk, and the offset the walk finished at — so a caller can tell whether it read
    the whole file or merely as much of it as was there."""
    chunks, offset = [], 12
    while offset + 8 <= len(blob):
        fourcc = blob[offset:offset + 4]
        size = struct.unpack("<I", blob[offset + 4:offset + 8])[0]
        payload = blob[offset + 8:offset + 8 + size]
        if len(payload) != size:
            return chunks, offset
        chunks.append((fourcc, payload))
        offset += 8 + size + (size & 1)
    return chunks, offset


def _webp_chunk(fourcc, payload):
    return fourcc + struct.pack("<I", len(payload)) + payload + (
        b"\x00" if len(payload) & 1 else b"")


def stamp_webp(path, tags, width, height):
    """Add the identity to a WebP **without re-encoding it**, in place.

    **ffmpeg's WebP muxer silently discards `-metadata`, with a zero exit code and no warning** —
    measured, not assumed. That is the same failure the MP4 path already carries a comment about,
    where `+use_metadata_tags` is what makes the master's tags exist at all; here there is no flag
    that helps, because the muxer never writes a metadata chunk.

    So the container is rewritten rather than the picture. A simple-format WebP holds one image
    chunk and nothing else and has nowhere to put metadata; the extended format (`VP8X`) does. The
    image chunk is carried across byte-for-byte and only the container around it changes, which is
    the property that matters: **a poster is a delivered artefact, and re-encoding one to attach a
    label would trade the customer's pixels for our bookkeeping.** Re-saving through PIL would do
    exactly that, which is why it is not what happens here.

    Returns True if the file now carries the identity.
    """
    with open(path, "rb") as handle:
        blob = handle.read()
    if blob[:4] != b"RIFF" or blob[8:12] != b"WEBP":
        raise ValueError("not a WebP container: {}".format(path))

    # **The walk has to consume the file exactly.** `_webp_chunks` stops when fewer than eight
    # bytes remain and reports what it read; it cannot tell a complete container from a truncated
    # one. Rewriting a truncated file would emit *corrected* chunk lengths — a container that
    # every parser now calls well-formed while the image payload inside it is short, which is a
    # worse outcome than the unstamped file it started as.
    #
    # **This is checked here rather than left to whoever opened the image first.** It was, once:
    # the caller happened to decode the file through PIL and that rejected a truncation on the
    # way past. A guard that lives in another function's incidental behaviour is a guard that
    # leaves when that function does.
    declared = struct.unpack("<I", blob[4:8])[0] if len(blob) >= 8 else 0
    if declared + 8 != len(blob):
        raise ValueError("WebP size field says {} bytes, file holds {}: {}".format(
            declared + 8, len(blob), path))
    existing, consumed = _webp_chunks_with_end(blob)
    if consumed != len(blob):
        raise ValueError("WebP chunk walk ended at {} of {} bytes: {}".format(
            consumed, len(blob), path))
    # A second XMP chunk would be undefined; an existing VP8X keeps its canvas and its other
    # flags, because it may be announcing an alpha channel this code knows nothing about.
    carried = [(f, p) for f, p in existing if f not in (b"VP8X", b"XMP ")]
    previous = [p for f, p in existing if f == b"VP8X"]
    if previous:
        header = bytes([previous[0][0] | _VP8X_XMP_FLAG]) + previous[0][1:]
    else:
        # Flags, three reserved bytes, then canvas width-1 and height-1 as 24-bit little-endian.
        header = (bytes([_VP8X_XMP_FLAG]) + b"\x00\x00\x00"
                  + struct.pack("<I", width - 1)[:3] + struct.pack("<I", height - 1)[:3])

    body = _webp_chunk(b"VP8X", header)
    for fourcc, payload in carried:
        body += _webp_chunk(fourcc, payload)
    # XMP goes last, which is where the container specification puts it.
    body += _webp_chunk(b"XMP ", _xmp_packet(tags))
    stamped = b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body

    # **Written beside the file and moved onto it, never over it.** `open(path, "wb")` truncates
    # before the first byte is written, so a failed write — a full disk, an I/O error, the cgroup
    # SIGKILL this worker has a documented history of — would leave a zero-byte poster where a
    # delivered one had been, and the caller above would report it as intact and upload it. The
    # rename is atomic on the same filesystem, so the artefact is either the original or the
    # stamped one and never a truncation of either.
    temporary = path + ".stamping"
    try:
        with open(temporary, "wb") as handle:
            handle.write(stamped)
        os.replace(temporary, path)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt or a SystemExit between the two must
        # not leave the scratch file behind either.
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return True


def verify(path):
    """Read the identity back **out of the delivered bytes**. False when it is not there.

    **The field beside it used to be a literal `True`.** Two of the three roles reported that they
    had been stamped without anything having looked: PIL ignores a save option it does not
    implement, and the MP4 muxer drops keys it does not recognise with a zero exit code — which
    are the two failures the code around this already carries comments about. A run on a runtime
    whose Pillow predates WebP XMP would have written an anonymous crop and filed a record saying
    it had not.

    That is the class this module's own docstring names: a mechanism absent from every file while
    every check around it passes. The kit catches it on the machine running the kit; this catches
    it in the image, on the job, which is where it would actually happen.

    **Never raises.** A verification that could fail a job would be worse than the defect.
    """
    try:
        if path.lower().endswith(".webp"):
            from PIL import Image  # noqa: PLC0415 — leaf import, same as the stamping path
            with Image.open(path) as image:
                packet = image.info.get("xmp") or b""
            return b"cf_request_id" in packet
        import subprocess  # noqa: PLC0415 — only the container-tagged roles reach this
        probed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags", "-print_format", "json",
             path], capture_output=True, timeout=30)
        if probed.returncode != 0:
            return False
        tags = json.loads(probed.stdout).get("format", {}).get("tags", {})
        return bool(tags.get("cf_request_id"))
    except Exception:  # noqa: BLE001 — an unverifiable stamp is reported as unstamped
        return False


def _stamp(path, tags, warn=None):
    """Stamp a delivered WebP, and **never lose the artefact over the label.**

    The same order the module already keeps between a derive and the master: an unstamped poster
    is a poster, and a poster that failed to be written is nothing. So a stamping failure warns
    and leaves the file exactly as it was.

    **It warns rather than passing quietly**, because a stamp that silently does nothing is
    indistinguishable from one that works — and this project has already paid for a metadata
    mechanism that was absent from every file while every check around it passed.
    """
    try:
        from PIL import Image  # noqa: PLC0415 — the canvas is not in the container's header
        with Image.open(path) as image:
            size = image.size
        stamp_webp(path, tags, size[0], size[1])
        return verify(path)
    except Exception as exc:  # noqa: BLE001 — the artefact outranks its label
        if warn is not None:
            warn("derive identity not stamped on {} ({}: {}); the file is delivered and intact, "
                 "but it cannot say whose job it came from".format(
                     os.path.basename(path), type(exc).__name__, str(exc)[:200]))
        return False


def poster_frame_index(at_fraction, frame_count):
    """`round(at_fraction × (frame_count − 1))`.

    0 is the first frame, 1 the last, and **a single-frame video degenerates to 0 rather than
    erroring** — which is also the whole of the image path's poster behaviour, reached without a
    branch worth naming.
    """
    return int(round(at_fraction * max(0, frame_count - 1)))


def build(spec, master_path, source_path, workdir, frame_count, scale, request_id,
          warn=None, identity=None):
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
    # **The identity every role stamps into what it delivers**, `api.md` §6 item 4. Assembled
    # once here rather than per role: the three writers disagree about the container and about
    # nothing else, and a per-role copy is how two of them would end up saying different things
    # about the same job.
    identity = dict(identity or {})
    identity.setdefault("cf_request_id", request_id)

    entries = []
    for item in spec:
        role = item["role"]
        try:
            if role == "poster":
                entries.append(_poster(item, master_path, workdir, frame_count, request_id,
                                       identity=identity, warn=warn))
            elif role == "proxy":
                entries.append(_proxy(item, master_path, workdir, request_id,
                                      identity=identity))
            elif role == "crop":
                entries.extend(_crops(item, master_path, source_path, workdir, frame_count, scale,
                                      request_id,
                                      warn=warn, identity=identity))
        except Exception as exc:  # noqa: BLE001 — the master outranks every derive, see above
            if warn is not None:
                warn("derive '{}' failed and was omitted: {}: {}".format(
                    role, type(exc).__name__, str(exc)[:200]))
    return entries


def _poster(item, master_path, workdir, frame_count, request_id, identity=None, warn=None):
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
    name = keys.for_role(request_id, "poster")
    path = os.path.join(workdir, name)
    index = poster_frame_index(item.get("at_fraction", 0.25), frame_count)
    encoder.extract_poster(master_path, path, index, fps=None)
    # **After the encode, and without touching it.** ffmpeg wrote these bytes and dropped every
    # `-metadata` it was given; the container is rewritten around the picture rather than the
    # picture re-encoded. `bytes` is read afterwards so the recorded size is the delivered size.
    stamped = _stamp(path, dict(identity or {}, cf_role="poster", cf_frame_index=index),
                     warn=warn)
    return {
        "role": "poster", "name": name, "path": path,
        "content_type": keys.content_type(name),
        "frame_index": index,
        "identity_stamped": stamped,
        "bytes": os.path.getsize(path),
    }


def _proxy(item, master_path, workdir, request_id, identity=None):
    name = keys.for_role(request_id, "proxy")
    path = os.path.join(workdir, name)
    # **The proxy is an MP4, so it is stamped in the mux that is already happening** — the same
    # mechanism the master uses, `+use_metadata_tags` included, because without that flag the MP4
    # muxer drops every key it does not recognise and `cf_request_id` is one of them.
    encoder.encode_proxy(master_path, path, max_duration_s=item.get("max_duration_s"),
                         identity=dict(identity or {}, cf_role="proxy"))
    return {
        "role": "proxy", "name": name, "path": path,
        "content_type": keys.content_type(name),
        # **Read back, not asserted.** The muxer drops what it does not recognise and says
        # nothing; a literal True here would report a tag that is not in the file.
        "identity_stamped": verify(path),
        "bytes": os.path.getsize(path),
    }


def _crops(item, master_path, source_path, workdir, frame_count, scale, request_id,
           warn=None, identity=None):
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
        name = keys.crop_name(request_id, ordinal)
        path = os.path.join(workdir, name)
        # **PIL writes this one, and PIL carries XMP itself** — no container surgery is needed
        # where the encoder is ours to ask. Still lossless: the identity rides in a metadata
        # chunk and the evidence pixels are untouched.
        crops.write_lossless_webp(
            pair, path,
            xmp=_xmp_packet(dict(identity or {}, cf_role="crop", cf_crop_ordinal=ordinal,
                                 cf_frame_index=index)))
        entries.append({
            "role": "crop", "name": name, "path": path,
            # PIL ignores a save option it does not implement, so this is read back too.
            "identity_stamped": verify(path),
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
