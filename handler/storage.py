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


def fetch_source(source_url, destination):
    """Stream the presigned GET to disk. No media ever arrives in the payload."""
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


def upload(client, output, name, path, content_type):
    """Write one file under the prefix. The key is deterministic, so a re-run overwrites."""
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


def put_diagnostics(diagnostics_url, body, content_type="application/json"):
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
        response = requests.put(
            diagnostics_url,
            data=body,
            headers={"Content-Type": content_type},
            timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
        )
        response.raise_for_status()
        return True
    except Exception:  # noqa: BLE001 — see the docstring; this must never fail the job
        return False
