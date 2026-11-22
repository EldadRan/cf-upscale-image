"""Fetch the weights at start, when the image did not bake them (X4, CF 2026-11-22).

**Why an image would not bake them.** The X1 Vast test pulled the fp8 image onto a 4.7 Gbps host
and spent 20+ minutes on it: the two largest layers (3.98 GB and the 6.66 GB weights layer) both
failed part-way and **a Docker layer retry restarts the blob from zero**. A fetch here resumes a
partial file instead, and the only recorded reason for baking — RunPod network volumes in few
datacentres — does not apply to a Vast instance. `cf-upscale-7b-fp16-runpod` still bakes.

**The pins stay in one place**: `bake_weights.REVISIONS` and `bake_weights.EXPECTED`, read here
exactly as the build reads them. Each file comes from the repository the vendored registry
places it in, at that repository's pinned revision, and is checked for size and sha256 before
anything loads it. A file that fails its hash is deleted and fetched again, never loaded.

**How it fetches.** Parallel ranged GETs against HuggingFace's `resolve` URL — the same transfer
shape `hf_transfer` uses — each range resumable from the bytes already on disk, each retried
with backoff. Hand-written rather than `hf_hub_download` so the resume, the retry and the cut
are this module's own behaviour, which a stand-in server can witness; the library's resume
semantics vary by version and by storage backend. `HF_ENDPOINT` moves the host (the witness
uses it). **`HF_TOKEN` is sent if set, never required and never printed**; `requests` drops it on
the redirect to the CDN, which is a different host.

**Hashed once per host.** A verified file gets a marker beside it naming the size, sha256, repo
and revision it was verified against; a later call that finds a marker matching the pins and a
file of the right size loads without reading 8 GB again.

**A fetch that cannot complete is a refusal, not a hang.** Every read has a timeout, every range
a bounded number of attempts, the whole fetch a ceiling, and the failure is a `WorkerError` with
code `internal` and remedy `retry_same` — an existing code (CF may give it its own later,
doc-first). What was attempted is in the run record either way.
"""

import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import socket
import threading
import time

import errors

#: The flag the Dockerfile sets when it bakes. Read by the vendored validation skip too
#: (`vendor_patch.py`), which is why this module sets it IN-PROCESS once every file it is about
#: to load has been verified here: the skip then applies to exactly the files this path checked.
#: **Cleared again at the start of every fetching call**, so a failure never inherits an earlier
#: success's claim.
BAKED_FLAG = "SEEDVR2_WEIGHTS_BAKED"

#: Whether THIS module set the flag. A second job in the same process must still be recorded as
#: a fetch (its markers are re-checked, which is cheap), not mistaken for a baked image.
_FLAG_SET_HERE = [False]

DEFAULT_ENDPOINT = "https://huggingface.co"
#: Ranges per file. Eight streams is where `hf_transfer`'s own default sits; a range is never
#: smaller than `MIN_RANGE_BYTES`, so the 501 MB VAE gets two.
STREAMS = 8
MIN_RANGE_BYTES = 256 * 1024 * 1024
#: The most one read asks for. Reads return whatever has arrived (`read1`), so a cut loses
#: nothing already received and the deadline is checked after every read, however slow the
#: stream — a trickling range cannot outlive the fetch's ceiling (X4 review).
CHUNK = 1024 * 1024
#: Attempts per range, with backoff 1, 2, 4 … capped at `BACKOFF_CAP_S`, none after the last.
RANGE_ATTEMPTS = 8
BACKOFF_CAP_S = 30.0
#: Whole-file attempts when the bytes arrive but the hash is wrong.
HASH_ATTEMPTS = 2
#: Connect and per-read timeouts: a silent stream errors and is retried from where it stopped.
CONNECT_TIMEOUT_S = 15.0
READ_TIMEOUT_S = 60.0
#: The whole fetch's ceiling, including waiting for another worker's fetch on this host. 8.9 GB
#: at a pessimistic 20 MB/s is ~450 s; this is three times that.
MAX_FETCH_S = 1800.0
#: How often the fetch is checked against its ceiling and for a failed sibling range.
WATCH_S = 0.5
#: Headroom kept free beyond what the download and the join need.
DISK_MARGIN_BYTES = 256 * 1024 * 1024


def fetching():
    """Whether `ensure` will do more than record `baked` — for the caller's progress line."""
    return os.environ.get(BAKED_FLAG) != "1" or _FLAG_SET_HERE[0]


def _marker_path(path):
    return path + ".verified.json"


def _sha256_file(path):
    """Streamed; the DiT is 8 GB. Module-level so the witness can count calls to it."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_marker(path):
    try:
        with open(_marker_path(path), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _marker_matches(path, size, sha, repo, revision):
    marker = _read_marker(path)
    return (marker is not None and os.path.isfile(path) and os.path.getsize(path) == size
            and marker.get("sha256") == sha and marker.get("size") == size
            and marker.get("repo") == repo and marker.get("revision") == revision)


def _write_marker(path, size, sha, repo, revision):
    tmp = _marker_path(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"sha256": sha, "size": size, "repo": repo, "revision": revision,
                   "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, handle)
    os.replace(tmp, _marker_path(path))


def _discard(path):
    for doomed in (path, _marker_path(path)):
        try:
            os.remove(doomed)
        except FileNotFoundError:
            pass


def _ranges(size, streams=None, min_range=None):
    streams = streams or STREAMS
    min_range = min_range or MIN_RANGE_BYTES
    count = max(1, min(streams, -(-size // min_range)))
    step = -(-size // count)
    return [(start, min(size, start + step) - 1) for start in range(0, size, step)]


def _backoff(attempt):
    return min(BACKOFF_CAP_S, 2.0 ** attempt)


class _Cancelled(Exception):
    pass


def _abort(response):
    """Wake a thread blocked reading this response, from another thread.

    **`close()` alone does not**: a `recv` already blocked on the socket keeps waiting for its read
    timeout after another thread closes the file object (the X4 stall witness measured exactly
    that). `shutdown` does wake it, so the socket is shut down first, then the response closed.
    """
    raw = getattr(response, "raw", None)
    connection = getattr(raw, "connection", None) or getattr(raw, "_connection", None)
    sock = getattr(connection, "sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    try:
        response.close()
    except Exception:  # noqa: BLE001 — best effort; the thread exits either way
        pass


class _Fetch:
    """What the ranges of one file share: the byte count, the cancel signal, the deadline, and
    the open responses — so a cancel can close a socket a thread is blocked reading."""

    def __init__(self, deadline, max_fetch_s):
        self.deadline, self.max_fetch_s = deadline, max_fetch_s
        self.bytes = 0
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.open = set()

    def add(self, n):
        with self.lock:
            self.bytes += n

    def track(self, response, on):
        with self.lock:
            (self.open.add if on else self.open.discard)(response)

    def stop(self):
        """Cancel every range and close what they are reading, so none waits out a timeout."""
        self.cancel.set()
        with self.lock:
            responses = list(self.open)
        for response in responses:
            _abort(response)

    def check(self):
        if self.cancel.is_set():
            raise _Cancelled()
        if time.time() > self.deadline:
            raise TimeoutError("the weights fetch passed its {:.0f} s ceiling".format(
                self.max_fetch_s))


def _fetch_range(session, url, headers, part, start, end, fetch, attempts, backoff, chunk,
                 read_timeout):
    """One byte range into its own part file, resuming from whatever that file already holds."""
    want = end - start + 1
    last_error = None
    for attempt in range(attempts):
        fetch.check()
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > want:
            os.remove(part)
            have = 0
        if have == want:
            return
        response = None
        try:
            ranged = dict(headers, Range="bytes={}-{}".format(start + have, end))
            response = session.get(url, headers=ranged, stream=True, allow_redirects=True,
                                   timeout=(CONNECT_TIMEOUT_S, read_timeout))
            fetch.track(response, True)
            if response.status_code != 206:
                raise IOError("HTTP {} for a ranged GET (wanted 206)".format(response.status_code))
            served = response.headers.get("Content-Range", "")
            if not served.startswith("bytes {}-".format(start + have)):
                raise IOError("served {!r} for bytes {}-{}".format(served, start + have, end))
            raw = response.raw
            read = getattr(raw, "read1", None) or raw.read
            with open(part, "ab") as handle:
                while True:
                    fetch.check()
                    block = read(chunk)
                    if not block:
                        break
                    handle.write(block)
                    fetch.add(len(block))
            if os.path.getsize(part) == want:
                return
            last_error = IOError("range {}-{} ended short".format(start, end))
        except (_Cancelled, TimeoutError):
            raise
        except Exception as exc:  # noqa: BLE001 — every transport failure is retried, then reported
            if fetch.cancel.is_set():
                raise _Cancelled()
            last_error = exc
        finally:
            if response is not None:
                fetch.track(response, False)
                response.close()
        if attempt + 1 < attempts and fetch.cancel.wait(backoff(attempt)):
            raise _Cancelled()
    raise IOError("range {}-{} failed after {} attempts: {}".format(
        start, end, attempts, last_error))


def _download(url, path, size, token, deadline, max_fetch_s, on_bytes=None, streams=None,
              min_range=None, attempts=None, backoff=None, chunk=None, read_timeout=None,
              session=None):
    """Fetch `url` into `path` as parallel resumable ranges; return (bytes_downloaded, seconds,
    sha256, verify_seconds). The hash is computed while the ranges are joined, so the file is
    read once."""
    import requests  # noqa: PLC0415 — only a fetching worker needs it

    session = session or requests.Session()
    headers = {"Authorization": "Bearer " + token} if token else {}
    spans = _ranges(size, streams, min_range)
    parts = ["{}.part{}".format(path, index) for index in range(len(spans))]
    tmp = path + ".joining"
    if os.path.exists(tmp):
        os.remove(tmp)

    # **Room for the download AND the join, checked before a byte moves** (X4 review). Peak use
    # is the whole file in parts plus the largest part being copied into the joined file.
    held = sum(os.path.getsize(p) for p in parts if os.path.exists(p))
    largest = max(end - start + 1 for start, end in spans)
    need = (size - held) + largest + DISK_MARGIN_BYTES
    free = shutil.disk_usage(os.path.dirname(path) or ".").free
    if free < need:
        raise errors.WorkerError(
            errors.INTERNAL, "weights: {} needs {:.2f} GB free to fetch and join, the disk has "
            "{:.2f} GB".format(os.path.basename(path), need / 1e9, free / 1e9),
            remedy=errors.Remedy.RETRY_SAME)

    fetch = _Fetch(deadline, max_fetch_s)
    started = time.time()
    reported = -1
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(spans)) as pool:
        futures = [pool.submit(_fetch_range, session, url, headers, part, start, end, fetch,
                               attempts or RANGE_ATTEMPTS, backoff or _backoff, chunk or CHUNK,
                               read_timeout or READ_TIMEOUT_S)
                   for part, (start, end) in zip(parts, spans)]
        failure = None
        while True:
            done, pending = concurrent.futures.wait(futures, timeout=WATCH_S)
            # Reported only when it moved: a callback on a timer would make a stalled fetch look
            # alive to anything reading the progress timestamp.
            if on_bytes is not None and fetch.bytes != reported:
                reported = fetch.bytes
                try:
                    on_bytes(reported)
                except Exception:  # noqa: BLE001 — progress must never fail a fetch
                    pass
            for future in done:
                exc = future.exception()
                if exc is not None and not isinstance(exc, _Cancelled) and failure is None:
                    failure = exc
            if failure is None and time.time() > deadline:
                failure = TimeoutError("the weights fetch passed its {:.0f} s ceiling".format(
                    max_fetch_s))
            if failure is not None:
                # One range failed or the ceiling passed: stop the siblings now and close their
                # sockets, so the refusal does not wait on the slowest of them.
                fetch.stop()
                concurrent.futures.wait(futures)
                raise failure
            if not pending:
                break
    download_s = time.time() - started

    verify_started = time.time()
    digest = hashlib.sha256()
    try:
        with open(tmp, "wb") as out:
            for part in parts:
                with open(part, "rb") as handle:
                    for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                        digest.update(block)
                        out.write(block)
                os.remove(part)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return fetch.bytes, download_s, digest.hexdigest(), time.time() - verify_started


def _default_repo_of(filename):
    from src.utils.model_registry import get_model_repo  # noqa: PLC0415 — vendored, after load_cli

    return get_model_repo(filename)


def _lock(lock_handle, deadline, max_fetch_s):
    """One fetcher per host, waited for no longer than the fetch's own ceiling."""
    while True:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.time() > deadline:
                raise TimeoutError("another fetch held this host's weights lock past the {:.0f} s "
                                   "ceiling".format(max_fetch_s))
            time.sleep(WATCH_S)


def ensure(model_dir, dit_file, into=None, on_bytes=None, repo_of=None, expected=None,
           revisions=None, endpoint=None, streams=None, min_range=None, attempts=None,
           backoff=None, max_fetch_s=None, chunk=None, read_timeout=None):
    """Make `dit_file` and the VAE present and verified in `model_dir`, or refuse.

    Returns, and writes into `into` as it goes, the record of where the weights came from:
    `{"source": "baked"}`, or `{"source": "fetched", "files": {...}, "bytes", "download_s",
    "verify_s", "error"}` where a file found already verified on this host contributes no bytes
    and no hash time. `into` is filled before a refusal is raised, so a failed fetch is recorded.
    """
    record = into if into is not None else {}
    if os.environ.get(BAKED_FLAG) == "1" and not _FLAG_SET_HERE[0]:
        record["source"] = "baked"
        return record

    # An earlier success in this process vouched for ITS files; this call vouches afresh.
    os.environ.pop(BAKED_FLAG, None)
    _FLAG_SET_HERE[0] = False

    import bake_weights  # noqa: PLC0415 — the one home of the pins

    expected = expected if expected is not None else bake_weights.EXPECTED
    revisions = revisions if revisions is not None else bake_weights.REVISIONS
    repo_of = repo_of or _default_repo_of
    endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
    token = os.environ.get("HF_TOKEN") or None
    max_fetch_s = max_fetch_s or MAX_FETCH_S
    deadline = time.time() + max_fetch_s

    record.update({"source": "fetched", "files": {}, "bytes": 0, "download_s": 0.0,
                   "verify_s": 0.0, "error": None})
    lock_handle = None
    try:
        os.makedirs(model_dir, exist_ok=True)
        lock_handle = open(os.path.join(model_dir, ".fetch.lock"), "w")
        # One fetcher per host: a second worker waits here and then finds the markers.
        _lock(lock_handle, deadline, max_fetch_s)
        for filename in (dit_file, bake_weights.VAE_FILE):
            entry = record["files"].setdefault(filename, {})
            if filename not in expected:
                raise errors.WorkerError(
                    errors.INTERNAL, "weights: {} has no recorded size and sha256 in "
                    "bake_weights.EXPECTED, so it cannot be fetched and verified".format(filename),
                    remedy=errors.Remedy.NONE)
            size, sha = expected[filename]
            repo = repo_of(filename)
            revision = revisions.get(repo)
            entry.update({"repo": repo, "revision": revision, "size": size})
            if revision is None:
                raise errors.WorkerError(
                    errors.INTERNAL, "weights: {} comes from {}, which has no pinned revision"
                    .format(filename, repo), remedy=errors.Remedy.NONE)
            path = os.path.join(model_dir, filename)

            if _marker_matches(path, size, sha, repo, revision):
                entry["how"] = "verified_on_host"
                continue
            # A whole file without a marker — say a fetch that finished but died before marking
            # it — is hashed rather than downloaded again.
            if os.path.isfile(path) and os.path.getsize(path) == size:
                started = time.time()
                got = _sha256_file(path)
                record["verify_s"] += time.time() - started
                if got == sha:
                    _write_marker(path, size, sha, repo, revision)
                    entry["how"] = "hashed_existing"
                    continue
                _discard(path)

            url = "{}/{}/resolve/{}/{}".format(endpoint, repo, revision, filename)
            for attempt in range(1, HASH_ATTEMPTS + 1):
                done_before = record["bytes"]
                got_bytes, download_s, got_sha, verify_s = _download(
                    url, path, size, token, deadline, max_fetch_s,
                    on_bytes=(lambda n, base=done_before: on_bytes(base + n))
                    if on_bytes else None,
                    streams=streams, min_range=min_range, attempts=attempts, backoff=backoff,
                    chunk=chunk, read_timeout=read_timeout)
                record["bytes"] += got_bytes
                record["download_s"] += download_s
                record["verify_s"] += verify_s
                entry["attempts"] = attempt
                if os.path.getsize(path) == size and got_sha == sha:
                    _write_marker(path, size, sha, repo, revision)
                    entry["how"] = "downloaded"
                    break
                # **Never loaded**: the bytes are deleted before anything can open them.
                _discard(path)
                entry["hash_mismatch"] = entry.get("hash_mismatch", 0) + 1
            else:
                raise errors.WorkerError(
                    errors.INTERNAL, "weights: {} failed its sha256 {} time(s) at {}@{}".format(
                        filename, HASH_ATTEMPTS, repo, revision[:12]),
                    remedy=errors.Remedy.RETRY_SAME)
    except errors.WorkerError as exc:
        record["error"] = exc.message
        raise
    except Exception as exc:  # noqa: BLE001 — every other failure becomes one clean refusal
        record["error"] = "{}: {}".format(type(exc).__name__, exc)
        raise errors.WorkerError(
            errors.INTERNAL, "weights fetch failed: {}".format(record["error"]),
            remedy=errors.Remedy.RETRY_SAME) from exc
    finally:
        record["download_s"] = round(record["download_s"], 1)
        record["verify_s"] = round(record["verify_s"], 1)
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
            finally:
                lock_handle.close()

    # Every file this process is about to load was verified by this call, just now or on an
    # earlier run whose marker matches the pins — so the vendored re-hash may be skipped.
    os.environ[BAKED_FLAG] = "1"
    _FLAG_SET_HERE[0] = True
    return record
