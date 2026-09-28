"""Fetching the source and writing artefacts under the prefix.

The credentials in `output` are temporary, write-only and scoped to the prefix, and they expire.
Everything here is built so an expired credential surfaces as a clean error rather than a hang:
bounded timeouts, few retries, no unbounded waits. Measured against real R2 on 2026-08-12 —
write-only, prefix-scoped and multipart-capable (`docs/decisions.md` 3.2).

**Keys are the worker's; the prefix and the authorization are CF's.** CF records the prefix and
not the keys, so recovery after the job record expires is a `ListObjectsV2` against names CF
never chose. That makes the naming part of the contract rather than an implementation detail:
deterministic, derivable from the request, and identical on a re-run. See `keys.py`.
"""

import os
import queue
import re
import threading
import time

import requests

from diagnostics import redact
from errors import OUTPUT_WRITE_FAILED, SOURCE_FETCH_FAILED, Remedy, WorkerError

CONNECT_TIMEOUT_S = 10
READ_TIMEOUT_S = 60
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# R2 ignores the region but boto3 insists on one.
R2_REGION = "auto"

#: MiB = 2^20, throughout this module (T1: the RIFE worker's "100 MB" was 100 MiB in code).
MIB = 1024 * 1024

#: **The ranged fetch** (T1d, adopted from the RIFE worker's §25a): a source of at least
#: `FETCH_RANGED_MIN_BYTES` whose server answers a one-byte probe with a 206, a total and a strong
#: ETag is fetched in `FETCH_PART_BYTES` slices by up to `FETCH_STREAMS` streams, each held to
#: that ETag. Anything else is the single stream it replaces, and the record says why.
FETCH_RANGED_MIN_BYTES = 64 * MIB
FETCH_STREAMS = 8
FETCH_PART_BYTES = 32 * MIB
#: A slice is tried this many times before the fetch is `source_fetch_failed`. No backoff: the
#: single stream it replaced did not retry at all.
FETCH_ATTEMPTS = 3
_CONTENT_RANGE = re.compile(r"^\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$", re.IGNORECASE)

# Above this, boto3 splits the write into a multipart upload. A single PUT tops out around 5 GiB
# on R2, and this worker's master will cross that on ordinary content — the media worker measured
# a 985 MB remux from a two-minute 4K source, and an upscale of the same source is larger again.
#
# Multipart needs four actions beyond PutObject: CreateMultipartUpload, UploadPart,
# CompleteMultipartUpload and AbortMultipartUpload. A credential scoped to PutObject alone fails
# at the exact moment a write crosses this threshold — **so the threshold and the credential's
# actions are one decision, not two.** All five are proved working against real R2
# (`docs/decisions.md` 3.2), so the lower threshold only uses them more often.
#
# **16 MiB, and no longer an environment knob** (T1c, adopted from the RIFE worker's §26d): below
# it, one PUT; at or above it, parts. The caller has no say and neither does the endpoint.
MULTIPART_THRESHOLD_BYTES = 16 * MIB

#: **Parts in flight** (T1c). One at a time was this worker's rule — "parallel parts hold more
#: buffers resident" — and it held a 920 MiB master to ~13 MiB/s. Sixteen parts of at most 64 MiB
#: bound the buffers near 1 GiB against hosts measured at 46 GiB and up; the reading that
#: judges it safe is `phasewatch.TransferPeak`'s, inside the upload.
UPLOAD_CONCURRENCY = 16

#: The part size is taken from the file (`upload_part_bytes`): ~32 parts, within these bounds.
UPLOAD_PARTS_TARGET = 32
UPLOAD_PART_MIN_BYTES = 8 * MIB
UPLOAD_PART_MAX_BYTES = 64 * MIB


def upload_part_bytes(nbytes):
    """T1c: `ceil(nbytes / 32)`, rounded UP to a whole MiB, kept within 8-64 MiB.

    A 480 MiB 4K master goes up in 32 parts of 15 MiB rather than 8 of 64 — on the RIFE worker
    those 64 MiB parts read 19-46 MB/s at 4K against 181-242 at 8K, because eight parts cannot
    keep sixteen streams busy. 8K keeps 64 MiB."""
    per_part = -(-max(0, int(nbytes)) // UPLOAD_PARTS_TARGET)
    whole = -(-per_part // MIB) * MIB
    return max(UPLOAD_PART_MIN_BYTES, min(UPLOAD_PART_MAX_BYTES, whole))

# Errors R2 returns for a credential that has expired or was never valid for this prefix.
CREDENTIAL_ERROR_CODES = {
    "ExpiredToken",
    "ExpiredTokenException",
    "InvalidAccessKeyId",
    "SignatureDoesNotMatch",
    "AccessDenied",
    "InvalidToken",
}


def _transfer(transfers, direction, name, started, nbytes, ok, **how):
    """One object's clock and byte count onto `transfers` (J12). A None list records nothing.

    `how` is how it moved (T1c/T1d): an upload's `role`, `parts` and `part_bytes`; the fetch's
    `mode`, `mode_reason`, `streams` and `part_bytes`."""
    if transfers is None:
        return
    entry = {"direction": direction, "name": name,
             "seconds": round(time.time() - started, 3), "bytes": nbytes, "ok": ok}
    entry.update(how)
    transfers.append(entry)


def _size_or_none(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def fetch_source(source_url, destination, transfers=None, on_bytes=None, stats=None):
    """Fetch the presigned GET to disk — in parallel ranges where the server allows it, as one
    stream where it does not (T1d). No media ever arrives in the payload.

    **Timed and counted onto `transfers`, on failure too** (J12): the fetch that broke a job is
    the one worth having, and without it the residual `wall_s` minus the phases mixed the fetch,
    the uploads and everything else into one number nobody could read. **With how it was
    attempted** — `mode`, `mode_reason`, `streams`, `part_bytes` — decided by the probe BEFORE a
    byte of the body moves, so a fetch that fails still says which way it tried. `stats`, a dict
    or None, receives the same four.

    **`on_bytes(done, expected)` is monotonic in both modes** (`_Relay`).
    """
    started = time.time()
    stats = {} if stats is None else stats
    counted = {}
    try:
        received = _fetch(source_url, destination, on_bytes, stats, counted)
    except BaseException:
        # **A ranged fetch's file is pre-sized to the total**, so on failure its size would read
        # as a whole source; what the slices counted is what was received.
        _transfer(transfers, "fetch", "source", started,
                  counted["bytes"] if "bytes" in counted else _size_or_none(destination), False,
                  **stats)
        raise
    # **What was counted, not the file's size**: a ranged fetch pre-sizes the file to the total,
    # so its size would read whole with a slice never written.
    _transfer(transfers, "fetch", "source", started, received, True, **stats)
    return destination


def _fetch(source_url, destination, on_bytes, stats, counted):
    """The probe decides; returns the bytes received. A ranged fetch keeps its running count in
    `counted["bytes"]`, so a failure can say what arrived."""
    total, etag, reason = _probe_ranged(source_url)
    if reason is None and total < FETCH_RANGED_MIN_BYTES:
        reason = "small: {} bytes, under {} MiB".format(total, FETCH_RANGED_MIN_BYTES // MIB)
    if reason is not None:
        stats.update(mode="single", mode_reason=reason, streams=1, part_bytes=None)
        print("[fetch] one stream: {}".format(reason))
        return _fetch_single(source_url, destination, on_bytes)
    streams = max(1, min(FETCH_STREAMS, -(-total // FETCH_PART_BYTES)))
    stats.update(mode="ranged", mode_reason=None, streams=streams, part_bytes=FETCH_PART_BYTES)
    print("[fetch] {} streams of {} MiB slices over {} bytes".format(
        streams, FETCH_PART_BYTES // MIB, total))
    received = _fetch_ranged(source_url, destination, total, etag, streams, on_bytes, counted)
    # **The size check** (T1d step 5): counted per slice, against the probe's total.
    if received != total:
        raise WorkerError(SOURCE_FETCH_FAILED,
                          "the ranged fetch wrote {} bytes of the {} the source declared"
                          .format(received, total))
    return received


def _probe_ranged(url):
    """`(total, etag, None)` when ranged mode is possible, else `(None, None, reason)`.

    A `Range: bytes=0-0` GET — a GET, not a HEAD, because a presigned URL is signed for one
    method. Ranged mode needs a 206, a total in `Content-Range`, and a STRONG ETag: a weak one
    cannot pass `If-Match`, which is a strong comparison. **Never raises**: a probe that fails
    outright leads to the single stream, which then fails — or succeeds — exactly as before."""
    try:
        with requests.get(url, headers={"Range": "bytes=0-0"}, stream=True,
                          timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)) as response:
            status = response.status_code
            content_range = response.headers.get("Content-Range") or ""
            etag = response.headers.get("ETag")
    except requests.exceptions.RequestException as exc:
        return None, None, "probe failed: {}".format(type(exc).__name__)
    if status != 206:
        return None, None, "probe answered {}, not 206".format(status)
    match = _CONTENT_RANGE.match(content_range)
    if match is None:
        return None, None, "probe gave no total size (Content-Range {!r})".format(
            redact(content_range))
    if not etag:
        return None, None, "probe gave no ETag"
    if etag.startswith("W/"):
        return None, None, "probe gave a weak ETag"
    return int(match.group(3)), etag, None


class _SliceError(Exception):
    """One attempt at one slice failed in a way another attempt may not."""


def _fetch_ranged(url, destination, total, etag, streams, on_bytes, counted):
    """The file pre-sized, each slice written IN PLACE at its offset by one of `streams` workers,
    every request held to `etag`; nothing held beyond read buffers. Returns the bytes WRITTEN,
    counted per slice.

    **A final failure of any slice stops the others and raises** — `source_fetch_failed` for a
    transfer that may succeed next time, anything else (a full disk) as itself, which is what the
    single stream does with it too. **The descriptor is closed only after every started worker
    has been joined**, whatever raised.
    """
    pending = queue.Queue()
    for first in range(0, total, FETCH_PART_BYTES):
        pending.put((first, min(first + FETCH_PART_BYTES, total) - 1))
    relay = _Relay(on_bytes, total)
    stop = threading.Event()
    failures = []
    counted["bytes"] = 0
    tally = threading.Lock()

    def work():
        session = requests.Session()
        try:
            while not stop.is_set():
                try:
                    first, last = pending.get_nowait()
                except queue.Empty:
                    return
                landed = _fetch_slice(session, url, fd, first, last, total, etag, relay, stop)
                with tally:
                    counted["bytes"] += landed
        except BaseException as exc:  # noqa: BLE001 — carried to the caller's thread, raised there
            failures.append(exc)
            stop.set()
        finally:
            session.close()

    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    started = []
    try:
        os.ftruncate(fd, total)
        for n in range(streams):
            worker = threading.Thread(target=work, name="fetch-{}".format(n), daemon=True)
            worker.start()
            started.append(worker)
        for worker in started:
            worker.join()
    except BaseException:
        stop.set()
        raise
    finally:
        for worker in started:
            worker.join()
        os.close(fd)
    if failures:
        raise failures[0]
    relay.flush()
    return counted["bytes"]


def _fetch_slice(session, url, fd, first, last, total, etag, relay, stop):
    """Bytes `first`-`last`, written at `first`; returns the bytes written — the slice's length,
    or 0 when the fetch was stopped. `FETCH_ATTEMPTS` tries; a retried slice takes back what its
    failed attempt reported. **A 412 is not retried**: the object changed under the fetch, and
    every further range would be refused the same way."""
    want = last - first + 1
    why = None
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        if stop.is_set():
            return 0
        got = 0
        try:
            with session.get(url, headers={"Range": "bytes={}-{}".format(first, last),
                                           "If-Match": etag},
                             stream=True, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)) as response:
                if response.status_code == 412:
                    raise WorkerError(
                        SOURCE_FETCH_FAILED,
                        "the source changed while it was being fetched: bytes {}-{} no longer "
                        "match the ETag {} the fetch began with (412). Retry once the object "
                        "at source_url is stable.".format(first, last, etag))
                if response.status_code != 206:
                    raise _SliceError("answered {}, not 206".format(response.status_code))
                match = _CONTENT_RANGE.match(response.headers.get("Content-Range") or "")
                if match is None or tuple(int(g) for g in match.groups()) != (first, last, total):
                    raise _SliceError("Content-Range {!r} is not bytes {}-{}/{}".format(
                        response.headers.get("Content-Range"), first, last, total))
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                    if stop.is_set():
                        return 0
                    if not chunk:
                        continue
                    if got + len(chunk) > want:
                        raise _SliceError("more than the {} bytes asked for".format(want))
                    _write_at(fd, chunk, first + got)
                    got += len(chunk)
                    relay(len(chunk))
            if got == want:
                return want
            why = "{} of {} bytes".format(got, want)
        except (requests.exceptions.RequestException, _SliceError) as exc:
            # The text can carry the presigned query, in either form `redact` knows.
            why = redact(str(exc))
        relay(-got)
        print("[fetch] bytes {}-{}, attempt {} of {}: {}".format(
            first, last, attempt, FETCH_ATTEMPTS, why))
    raise WorkerError(SOURCE_FETCH_FAILED,
                      "could not fetch bytes {}-{} of source_url in {} attempts: {}".format(
                          first, last, FETCH_ATTEMPTS, why))


def _write_at(fd, chunk, offset):
    """All of `chunk` at `offset`, or `OSError`. **`os.pwrite` may write less than it is handed
    without raising** — a disk filling under a sparse, pre-sized file is the realistic case — and
    a count that assumed the whole chunk would read whole over a zero-filled gap (review). A write
    that makes no progress raises, as the buffered single stream's would."""
    view = memoryview(chunk)
    while view:
        wrote = os.pwrite(fd, view, offset)
        if wrote <= 0:
            raise OSError("wrote 0 of {} bytes at offset {}".format(len(view), offset))
        view, offset = view[wrote:], offset + wrote


def _fetch_single(source_url, destination, on_bytes=None):
    """Stream the presigned GET to disk — the fetch this worker always made, kept as the single
    stream the probe falls back to. Returns the bytes received."""
    relay = _Relay(on_bytes, None)
    try:
        response = requests.get(
            source_url, stream=True, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)
        )
        response.raise_for_status()
        declared = response.headers.get("Content-Length")
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if chunk:
                    handle.write(chunk)
                    relay(len(chunk))
    except requests.exceptions.RequestException as exc:
        # Scrubbed here as well as at every surface (T1a): the surfaces are what hold when the
        # next error path is added, and this one is known.
        raise WorkerError(SOURCE_FETCH_FAILED,
                          "could not fetch source_url: {}".format(redact(str(exc))))

    received = os.path.getsize(destination)
    if received == 0:
        raise WorkerError(SOURCE_FETCH_FAILED, "source_url returned an empty body")

    # A truncated transfer may succeed next time, so it belongs in the retryable table — unlike
    # bytes that arrive whole and will not decode, which never will.
    if declared is not None and received < int(declared):
        raise WorkerError(
            SOURCE_FETCH_FAILED,
            "source_url returned {} bytes of a declared {}".format(received, declared),
        )
    return received


def client_for(output):
    """boto3 is imported here, not at module scope, and that is deliberate.

    Importing it costs about 100 ms, and no refusal ever reaches this function. So the cost is
    paid only by jobs that actually write something.
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=output["endpoint"],
        aws_access_key_id=output["access_key_id"],
        aws_secret_access_key=output["secret_access_key"],
        aws_session_token=output.get("session_token"),
        config=Config(
            region_name=R2_REGION,
            signature_version="s3v4",
            connect_timeout=CONNECT_TIMEOUT_S,
            read_timeout=READ_TIMEOUT_S,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def upload(client, output, name, path, content_type, transfers=None, role=None, on_bytes=None,
           stats=None):
    """Write one file under the prefix. The key is deterministic, so a re-run overwrites.

    **Timed and counted per object onto `transfers`, failures included** (J12), with how it went
    up (T1c): `role` as the caller names it (`master`, `derive`, `manifest`), `parts` as the
    store's calls said, and `part_bytes`. **`stats`, a dict or None, receives the same two.**

    **`on_bytes(done, expected)` reports an absolute, monotonic total** (`_Relay`), though boto3
    calls back from its own threads with deltas that go negative when a part is retried.
    """
    started = time.time()
    stats = {} if stats is None else stats
    try:
        key = _upload(client, output, name, path, content_type, on_bytes, stats)
    except BaseException:
        # **None, not the file's size**: how much of a failed upload moved is unknown, and the
        # size would total exactly as a delivery does. A failed fetch knows what it received.
        _transfer(transfers, "upload", name, started, None, False, role=role,
                  parts=stats.get("parts"), part_bytes=stats.get("part_bytes"))
        raise
    _transfer(transfers, "upload", name, started, _size_or_none(path), True, role=role,
              parts=stats.get("parts"), part_bytes=stats.get("part_bytes"))
    return key


class _Relay(object):
    """Byte deltas from many threads to an absolute, MONOTONIC `on_bytes(done, expected)`.

    **boto3 calls the callback from its worker threads, and a retried part reports a NEGATIVE
    delta** (s3transfer rewinds its progress); a retried fetch slice takes back what its failed
    attempt reported the same way. The running total is kept under a lock and can fall; a total
    is published only when it is higher than the last one published, so the reported number
    never goes backwards.

    **`on_bytes` runs outside the counting lock and by one thread at a time**: a thread that finds
    another publishing skips, and the publisher sends the latest total, so the published sequence
    is strictly increasing and no part waits on a publish. `flush()` on the caller's thread sends
    what a skipped report may have left. A raising `on_bytes` is dropped, never re-raised into the
    transfer. Adopted from the RIFE worker, where it was found in review.
    """

    def __init__(self, on_bytes, expected):
        self._on_bytes = on_bytes
        self._expected = expected
        self._lock = threading.Lock()
        self._emitting = threading.Lock()
        self.total = 0
        self.emitted = 0

    def __call__(self, delta):
        with self._lock:
            self.total += int(delta)
        if self._on_bytes is None or not self._emitting.acquire(False):
            return
        try:
            with self._lock:
                latest = self.total
            if latest > self.emitted:
                self.emitted = latest
                try:
                    self._on_bytes(latest, self._expected)
                except Exception:  # noqa: BLE001 — a report never costs a transfer
                    self._on_bytes = None
        finally:
            self._emitting.release()

    def flush(self):
        self(0)


class _CallWatch(object):
    """What one upload's S3 calls said, heard on the client's own events. **Never raises.**

    - **A failed abort.** s3transfer registers `AbortMultipartUpload` as the failure cleanup the
      moment the upload is created, and swallows the abort's own failure (it logs at DEBUG). This
      hears the abort's answer, so a failed abort reaches the error instead of leaving parts under
      the caller's prefix with no trace. `NoSuchUpload` is an abort that already succeeded.
    - **The parts the object is made of**: the list `CompleteMultipartUpload` sends, kept once
      that call succeeds, or 1 for a single `PutObject` — **counted, not computed**.

    A client with no event system (a test double) leaves both unheard: `parts` stays None.
    """

    def __init__(self, client):
        self.failures = []
        self.parts = None
        self._listed = None
        self._events = getattr(getattr(client, "meta", None), "events", None)
        self._id = "cf-upscale-call-watch-{}".format(id(self))
        self._hooks = (("after-call.s3.AbortMultipartUpload", self._heard),
                       ("after-call-error.s3.AbortMultipartUpload", self._heard),
                       ("provide-client-params.s3.CompleteMultipartUpload", self._listing),
                       ("after-call.s3.CompleteMultipartUpload", self._completed),
                       ("after-call.s3.PutObject", self._put))
        if self._events is not None:
            for event, hook in self._hooks:
                self._events.register(event, hook, unique_id=self._id + event)

    def _listing(self, params=None, **_kwargs):
        try:
            self._listed = len(((params or {}).get("MultipartUpload") or {}).get("Parts") or ())
        except Exception:  # noqa: BLE001 — see the class docstring
            pass

    def _completed(self, http_response=None, **_kwargs):
        try:
            if http_response is not None and http_response.status_code < 300:
                self.parts = self._listed
        except Exception:  # noqa: BLE001
            pass

    def _put(self, http_response=None, **_kwargs):
        try:
            if http_response is not None and http_response.status_code < 300:
                self.parts = 1
        except Exception:  # noqa: BLE001
            pass

    def _heard(self, http_response=None, parsed=None, exception=None, **_kwargs):
        try:
            if exception is not None:
                self.failures.append("{}: {}".format(type(exception).__name__, exception))
            elif http_response is not None and http_response.status_code >= 300:
                error = (parsed or {}).get("Error") or {}
                if error.get("Code") == "NoSuchUpload":
                    return
                self.failures.append("{} {}".format(error.get("Code") or http_response.status_code,
                                                    error.get("Message") or "").strip())
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        if self._events is not None:
            for event, _hook in self._hooks:
                try:
                    self._events.unregister(event, unique_id=self._id + event)
                except Exception:  # noqa: BLE001
                    pass

    def said(self):
        """The clause a failed upload's error ends with, or empty."""
        if not self.failures:
            return ""
        return (" — and aborting the multipart upload failed too ({}), so its parts may remain "
                "under the prefix".format("; ".join(self.failures)))


def _upload(client, output, name, path, content_type, on_bytes, stats):
    import botocore.exceptions
    from boto3.s3.transfer import TransferConfig
    from s3transfer.utils import ChunksizeAdjuster

    prefix = output["prefix"]
    key = "{}{}".format(prefix if prefix.endswith("/") else prefix + "/", name)
    nbytes = _size_or_none(path) or 0
    part = upload_part_bytes(nbytes)
    # Set only by a success below; a caller's dict must not carry an earlier upload's count.
    stats["parts"] = None
    config = TransferConfig(
        multipart_threshold=MULTIPART_THRESHOLD_BYTES,
        multipart_chunksize=part,
        # **Sixteen parts in flight, on boto3's threads** (T1c). "One part at a time" stood here,
        # for a headroom the hosts this endpoint serves do not need — see `UPLOAD_CONCURRENCY`.
        max_concurrency=UPLOAD_CONCURRENCY,
        use_threads=True,
    )
    # **A file handle's parts are read into memory under a SEPARATE bound**, 10 by default:
    # without this, sixteen threads ran ten parts.
    config.max_in_memory_upload_chunks = UPLOAD_CONCURRENCY
    # **The part size the parts actually took**, through the same adjuster s3transfer applies —
    # it moves a size only past S3's own limits, which 8-64 MiB never reaches below 625 GiB.
    # None for a single PUT, which has no parts to size.
    stats["part_bytes"] = (min(nbytes, ChunksizeAdjuster().adjust_chunksize(part, nbytes))
                           if nbytes >= MULTIPART_THRESHOLD_BYTES else None)
    relay = _Relay(on_bytes, nbytes)
    watch = _CallWatch(client)
    try:
        # upload_fileobj switches to multipart at the threshold and stays a single PUT below
        # it, so a poster keeps exactly the behaviour a single PUT would have given it.
        with open(path, "rb") as handle:
            client.upload_fileobj(
                handle,
                output["bucket"],
                key,
                ExtraArgs={"ContentType": content_type},
                Callback=relay if on_bytes is not None else None,
                Config=config,
            )
        relay.flush()
        stats["parts"] = watch.parts
    except botocore.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in CREDENTIAL_ERROR_CODES:
            # **The remedy this refusal spent 8 255 s of H200 time not having**
            # (F-2026-08-20-39). The GPU work all succeeded; only the door was locked. Naming
            # what to do about it is worth more here than on almost any other code, because the
            # answer is cheap and the alternative is a caller concluding the job is impossible.
            raise WorkerError(
                OUTPUT_WRITE_FAILED,
                "output credentials rejected writing {} ({}); they are temporary and may have "
                "expired. The work itself succeeded — resubmit the same request with a freshly "
                "minted credential whose lifetime covers this endpoint's execution timeout. "
                "Retrying the identical request will fail identically: the credential is the "
                "part that has to change{}.".format(key, code, watch.said()),
                remedy=Remedy.RETRY_SAME,
            )
        raise WorkerError(OUTPUT_WRITE_FAILED,
                          "could not write {}: {}{}".format(key, exc, watch.said()),
                          remedy=Remedy.RETRY_SAME)
    except (botocore.exceptions.BotoCoreError, OSError) as exc:
        # A transport failure against the caller's own bucket. The same card would serve the same
        # job again — this is the textbook `retry_same`, and it was returning null.
        raise WorkerError(OUTPUT_WRITE_FAILED,
                          "could not write {}: {}{}".format(key, exc, watch.said()),
                          remedy=Remedy.RETRY_SAME)
    finally:
        watch.close()
    return key


#: **The small writes' own timeouts** (`api.md` §4d clause 3, J7): the bundle and the run-record,
#: after the stop. `CONNECT_TIMEOUT_S`/`READ_TIMEOUT_S` above exist for the master; a 71 KB object
#: does not need 60 s, and it keeps these whether or not a deadline stopped the job — one write
#: with two timeout behaviours is what the next reader gets wrong.
SMALL_CONNECT_TIMEOUT_S = 5
SMALL_READ_TIMEOUT_S = 20
#: What one further attempt can cost. **A READ TIMEOUT IS NOT A WALL-CLOCK BOUND** (§4d clause 7):
#: it limits each socket read, not the PUT, so a peer that dribbles bytes can outlive it. The gate
#: below bounds the DECISION to spend again, not the attempt already running.
SMALL_ATTEMPT_S = SMALL_CONNECT_TIMEOUT_S + SMALL_READ_TIMEOUT_S


def put_small(url, body, content_type, deadline_at=None, clock=time.time, owed_after=0):
    """PUT one small object to a presigned URL: once, and a second time only where there is room.

    **The one question both post-stop writes ask** (§4d clause 3(b)). The retry is gated on the
    real clock, read at the moment it would be spent: a second attempt only where the time still
    left before `deadline_at` — handler entry plus `execution_timeout_ms` — covers this retry
    AND every first attempt still owed after it, `(1 + owed_after) x SMALL_ATTEMPT_S`. No
    deadline means none was given, and the retry stands.

    **A RESERVE SHARED BY SEVERAL WRITES CANNOT BE SPENT BY A PER-WRITE DECISION.** The bundle
    and the record share `WRITE_RESERVE_S`; gating each retry on its own attempt alone let the
    bundle's retry spend the record's first attempt, 75 s against 60. **`owed_after` is passed
    by the caller that knows the sequence — the handler — and never inferred here.** A third
    small write added later must join this count, or it reintroduces that defect exactly while
    looking correct at its own call site.

    **The first attempt is unconditional**, so a stop that fires late still eats the reserve:
    this bounds the decision to spend again, not the lag before the first PUT (clause 7).

    Returns `(ok, error, attempts)`. **Raises nothing**; the callers' posture is that a record or
    a bundle must never cost a job.
    """
    # Resolved per call rather than at module scope, so a caller that swaps the module in
    # `sys.modules` — the run-record's own witnesses do — reaches this path too.
    import requests as http  # noqa: PLC0415

    attempts, error = 0, None
    while True:
        attempts += 1
        try:
            response = http.put(url, data=body, headers={"Content-Type": content_type},
                                timeout=(SMALL_CONNECT_TIMEOUT_S, SMALL_READ_TIMEOUT_S))
            response.raise_for_status()
            return True, None, attempts
        except Exception as exc:  # noqa: BLE001 — see the docstring
            error = exc
        if attempts >= 2:
            return False, error, attempts
        if deadline_at is not None and \
                deadline_at - clock() < (1 + owed_after) * SMALL_ATTEMPT_S:
            return False, error, attempts


def put_diagnostics(diagnostics_url, body, content_type="application/json", deadline_at=None,
                    owed_after=0):
    """PUT the diagnostics bundle to CF's presigned URL. **Never raises.**

    A single presigned PUT rather than a second scoped credential, deliberately: it is one
    object against a different bucket, and a second credential would be another thing to scope,
    mint, expire and get wrong for an object written once or never.

    **Returns True/False rather than raising, and that is the whole point.** The one outcome
    worse than losing the diagnostics is losing the result because the diagnostics could not be
    stored. This is called on a path where the job has already failed, and a bare `except` here
    is correct rather than lazy.
    """
    if not diagnostics_url:
        return False
    try:
        return put_small(diagnostics_url, body, content_type, deadline_at=deadline_at,
                         owed_after=owed_after)[0]
    except Exception:  # noqa: BLE001 — see the docstring; this must never fail the job
        return False
