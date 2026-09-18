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
import time

import requests

from errors import OUTPUT_WRITE_FAILED, SOURCE_FETCH_FAILED, Remedy, WorkerError

CONNECT_TIMEOUT_S = 10
READ_TIMEOUT_S = 60
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# R2 ignores the region but boto3 insists on one.
R2_REGION = "auto"

# Above this, boto3 splits the write into a multipart upload. A single PUT tops out around 5 GiB
# on R2, and this worker's master will cross that on ordinary content — the media worker measured
# a 985 MB remux from a two-minute 4K source, and an upscale of the same source is larger again.
#
# Multipart needs four actions beyond PutObject: CreateMultipartUpload, UploadPart,
# CompleteMultipartUpload and AbortMultipartUpload. A credential scoped to PutObject alone fails
# at the exact moment a write crosses this threshold — **so the threshold and the credential's
# actions are one decision, not two.** All five are proved working against real R2
# (`docs/decisions.md` 3.2); what is still owed CF is the master's real size, which is what
# should set this number rather than the inherited default.
MULTIPART_THRESHOLD_BYTES = int(os.environ.get("MULTIPART_THRESHOLD_BYTES", 100 * 1024 * 1024))
MULTIPART_CHUNK_BYTES = 32 * 1024 * 1024

# Errors R2 returns for a credential that has expired or was never valid for this prefix.
CREDENTIAL_ERROR_CODES = {
    "ExpiredToken",
    "ExpiredTokenException",
    "InvalidAccessKeyId",
    "SignatureDoesNotMatch",
    "AccessDenied",
    "InvalidToken",
}


def _transfer(transfers, direction, name, started, nbytes, ok):
    """One object's clock and byte count onto `transfers` (J12). A None list records nothing."""
    if transfers is None:
        return
    transfers.append({"direction": direction, "name": name,
                      "seconds": round(time.time() - started, 3), "bytes": nbytes, "ok": ok})


def _size_or_none(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def fetch_source(source_url, destination, transfers=None):
    """Stream the presigned GET to disk. No media ever arrives in the payload.

    **Timed and counted onto `transfers`, on failure too** (J12): the fetch that broke a job is
    the one worth having, and without it the residual `wall_s` minus the phases mixed the fetch,
    the uploads and everything else into one number nobody could read.
    """
    started = time.time()
    try:
        destination = _fetch(source_url, destination)
    except BaseException:
        _transfer(transfers, "fetch", "source", started, _size_or_none(destination), False)
        raise
    _transfer(transfers, "fetch", "source", started, _size_or_none(destination), True)
    return destination


def _fetch(source_url, destination):
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
    except requests.exceptions.RequestException as exc:
        raise WorkerError(SOURCE_FETCH_FAILED, "could not fetch source_url: {}".format(exc))

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
    return destination


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


def upload(client, output, name, path, content_type, transfers=None):
    """Write one file under the prefix. The key is deterministic, so a re-run overwrites.

    **Timed and counted per object onto `transfers`, failures included** (J12).
    """
    started = time.time()
    try:
        key = _upload(client, output, name, path, content_type)
    except BaseException:
        # **None, not the file's size**: how much of a failed upload moved is unknown, and the
        # size would total exactly as a delivery does. A failed fetch knows what it received.
        _transfer(transfers, "upload", name, started, None, False)
        raise
    _transfer(transfers, "upload", name, started, _size_or_none(path), True)
    return key


def _upload(client, output, name, path, content_type):
    import botocore.exceptions
    from boto3.s3.transfer import TransferConfig

    prefix = output["prefix"]
    key = "{}{}".format(prefix if prefix.endswith("/") else prefix + "/", name)
    try:
        # upload_fileobj switches to multipart above the threshold and stays a single PUT below
        # it, so a poster keeps exactly the behaviour a single PUT would have given it.
        with open(path, "rb") as handle:
            client.upload_fileobj(
                handle,
                output["bucket"],
                key,
                ExtraArgs={"ContentType": content_type},
                Config=TransferConfig(
                    multipart_threshold=MULTIPART_THRESHOLD_BYTES,
                    multipart_chunksize=MULTIPART_CHUNK_BYTES,
                    # One part at a time. On the media worker this was because it was CPU-bound
                    # elsewhere; here the reason is stronger and is the standing rule — parallel
                    # parts hold more buffers resident, and this worker trades throughput for
                    # headroom every time, without asking.
                    max_concurrency=1,
                    use_threads=False,
                ),
            )
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
                "part that has to change.".format(key, code),
                remedy=Remedy.RETRY_SAME,
            )
        raise WorkerError(OUTPUT_WRITE_FAILED,
                          "could not write {}: {}".format(key, exc),
                          remedy=Remedy.RETRY_SAME)
    except (botocore.exceptions.BotoCoreError, OSError) as exc:
        # A transport failure against the caller's own bucket. The same card would serve the same
        # job again — this is the textbook `retry_same`, and it was returning null.
        raise WorkerError(OUTPUT_WRITE_FAILED,
                          "could not write {}: {}".format(key, exc),
                          remedy=Remedy.RETRY_SAME)
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
