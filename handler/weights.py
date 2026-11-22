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
import threading
import time

import errors

#: The flag the Dockerfile sets when it bakes. Read by the vendored validation skip too
#: (`vendor_patch.py`), which is why this module sets it IN-PROCESS once every file it is about
#: to load has been verified here: the skip then applies to exactly the files this path checked.
BAKED_FLAG = "SEEDVR2_WEIGHTS_BAKED"

#: Whether THIS module set the flag. A second job in the same process must still be recorded as
#: a fetch (its markers are re-checked, which is cheap), not mistaken for a baked image.
_FLAG_SET_HERE = [False]

DEFAULT_ENDPOINT = "https://huggingface.co"
#: Ranges per file. Eight streams is where `hf_transfer`'s own default sits; a range is never
#: smaller than `MIN_RANGE_BYTES`, so the 501 MB VAE gets two.
STREAMS = 8
MIN_RANGE_BYTES = 256 * 1024 * 1024
#: Bytes per read. **Also the resume granularity**: `iter_content` yields a block only once it is
#: whole, so a cut loses at most the block in flight — 1 MiB, not the 8 MiB this first was (the
#: stand-in witness caught a cut losing everything before its first block).
CHUNK = 1024 * 1024
#: Attempts per range, with backoff 1, 2, 4 … capped at `BACKOFF_CAP_S`.
RANGE_ATTEMPTS = 8
BACKOFF_CAP_S = 30.0
#: Whole-file attempts when the bytes arrive but the hash is wrong.
HASH_ATTEMPTS = 2
#: Connect and per-read timeouts: a stalled stream errors and is retried from where it stopped.
CONNECT_TIMEOUT_S = 15.0
READ_TIMEOUT_S = 60.0
#: The whole fetch's ceiling. 8.9 GB at a pessimistic 20 MB/s is ~450 s; this is three times that.
MAX_FETCH_S = 1800.0


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


class _Cancelled(Exception):
    pass


def _fetch_range(session, url, headers, part, start, end, deadline, counter, lock, cancel,
                 attempts, sleep, chunk=None):
    """One byte range into its own part file, resuming from whatever that file already holds."""
    want = end - start + 1
    last_error = None
    for attempt in range(attempts):
        if cancel.is_set():
            raise _Cancelled()
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > want:
            os.remove(part)
            have = 0
        if have == want:
            return
        if time.time() > deadline:
            raise TimeoutError("the weights fetch passed its {:.0f} s ceiling".format(MAX_FETCH_S))
        try:
            ranged = dict(headers, Range="bytes={}-{}".format(start + have, end))
            with session.get(url, headers=ranged, stream=True, allow_redirects=True,
                             timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)) as response:
                if response.status_code != 206:
                    raise IOError("HTTP {} for a ranged GET (wanted 206)".format(
                        response.status_code))
                served = response.headers.get("Content-Range", "")
                if not served.startswith("bytes {}-".format(start + have)):
                    raise IOError("served {!r} for bytes {}-{}".format(served, start + have, end))
                with open(part, "ab") as handle:
                    for block in response.iter_content(chunk or CHUNK):
                        if cancel.is_set():
                            raise _Cancelled()
                        handle.write(block)
                        with lock:
                            counter[0] += len(block)
                        if time.time() > deadline:
                            raise TimeoutError("the weights fetch passed its {:.0f} s ceiling"
                                               .format(MAX_FETCH_S))
            if os.path.getsize(part) == want:
                return
            last_error = IOError("range {}-{} ended short".format(start, end))
        except (_Cancelled, TimeoutError):
            raise
        except Exception as exc:  # noqa: BLE001 — every transport failure is retried, then reported
            last_error = exc
        sleep(min(BACKOFF_CAP_S, 2.0 ** attempt))
    raise IOError("range {}-{} failed after {} attempts: {}".format(
        start, end, attempts, last_error))


def _download(url, path, size, token, deadline, on_bytes=None, streams=None, min_range=None,
              attempts=None, sleep=time.sleep, session=None, chunk=None):
    """Fetch `url` into `path` as parallel resumable ranges; return (bytes_downloaded, seconds,
    sha256, verify_seconds). The hash is computed while the ranges are joined, so the file is
    read once."""
    import requests  # noqa: PLC0415 — only a fetching worker needs it

    session = session or requests.Session()
    headers = {"Authorization": "Bearer " + token} if token else {}
    spans = _ranges(size, streams, min_range)
    parts = ["{}.part{}".format(path, index) for index in range(len(spans))]
    counter, lock, cancel = [0], threading.Lock(), threading.Event()
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(spans)) as pool:
        futures = [pool.submit(_fetch_range, session, url, headers, part, start, end, deadline,
                               counter, lock, cancel, attempts or RANGE_ATTEMPTS, sleep, chunk)
                   for part, (start, end) in zip(parts, spans)]
        try:
            while True:
                done, pending = concurrent.futures.wait(futures, timeout=5.0)
                if on_bytes is not None:
                    on_bytes(counter[0])
                for future in done:
                    future.result()
                if not pending:
                    break
        except BaseException:
            cancel.set()
            raise
    download_s = time.time() - started

    verify_started = time.time()
    digest = hashlib.sha256()
    tmp = path + ".joining"
    with open(tmp, "wb") as out:
        for part in parts:
            with open(part, "rb") as handle:
                for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                    digest.update(block)
                    out.write(block)
            os.remove(part)
    os.replace(tmp, path)
    return counter[0], download_s, digest.hexdigest(), time.time() - verify_started


def _default_repo_of(filename):
    from src.utils.model_registry import get_model_repo  # noqa: PLC0415 — vendored, after load_cli

    return get_model_repo(filename)


def ensure(model_dir, dit_file, into=None, on_bytes=None, repo_of=None, expected=None,
           revisions=None, endpoint=None, streams=None, min_range=None, attempts=None,
           sleep=time.sleep, max_fetch_s=None, chunk=None):
    """Make `dit_file` and the VAE present and verified in `model_dir`, or refuse.

    Returns, and writes into `into` as it goes, the record of where the weights came from:
    `{"source": "baked"}`, or `{"source": "fetched", "files": {...}, "bytes", "download_s",
    "verify_s"}` where a file found already verified on this host contributes no bytes and no
    hash time. `into` is filled before a refusal is raised, so a failed fetch is recorded.
    """
    record = into if into is not None else {}
    if os.environ.get(BAKED_FLAG) == "1" and not _FLAG_SET_HERE[0]:
        record["source"] = "baked"
        return record

    import bake_weights  # noqa: PLC0415 — the one home of the pins

    expected = expected if expected is not None else bake_weights.EXPECTED
    revisions = revisions if revisions is not None else bake_weights.REVISIONS
    repo_of = repo_of or _default_repo_of
    endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
    token = os.environ.get("HF_TOKEN") or None
    deadline = time.time() + (max_fetch_s or MAX_FETCH_S)

    record.update({"source": "fetched", "files": {}, "bytes": 0, "download_s": 0.0,
                   "verify_s": 0.0, "error": None})
    os.makedirs(model_dir, exist_ok=True)
    lock_handle = open(os.path.join(model_dir, ".fetch.lock"), "w")
    try:
        # One fetcher per host: a second worker waits here and then finds the markers.
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
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
                    url, path, size, token, deadline,
                    on_bytes=(lambda n, base=done_before: on_bytes(base + n))
                    if on_bytes else None,
                    streams=streams, min_range=min_range, attempts=attempts, sleep=sleep,
                    chunk=chunk)
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
    except Exception as exc:  # noqa: BLE001 — transport failures become one clean refusal
        record["error"] = "{}: {}".format(type(exc).__name__, exc)
        raise errors.WorkerError(
            errors.INTERNAL, "weights fetch failed: {}".format(record["error"]),
            remedy=errors.Remedy.RETRY_SAME) from exc
    finally:
        record["download_s"] = round(record["download_s"], 1)
        record["verify_s"] = round(record["verify_s"], 1)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()

    # Every file this process is about to load was verified by this function, just now or on
    # an earlier run whose marker matches the pins — so the vendored re-hash may be skipped.
    os.environ[BAKED_FLAG] = "1"
    _FLAG_SET_HERE[0] = True
    return record
