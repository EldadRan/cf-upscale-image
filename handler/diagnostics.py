"""What a failed or recovered run leaves behind.

A failure you cannot read afterwards teaches nothing, and this worker will fail in ways worth
learning from — **an estimator that was wrong is a data point, not just an incident**.

**The worker pushes; nothing scrapes.** Pulling RunPod's worker logs was considered and rejected,
and the reasons compound: the stream is per *worker* rather than per job, so a busy worker
interleaves and attribution has to be reconstructed from timestamps that are themselves
unreliable — two adjacent statements in one function have been observed stamped 2 h 17 m apart.
Retention is undocumented. And SeedVR2 is verbose, so a puller would ship everything and filter
later, when **the one process that knows which lines matter is the one that caught the
exception.** Filtering in-process costs nothing and loses nothing.

**Written on failure, and on any job that retried — including one that recovered.** A recovered
job is the most useful record there is: it holds both the estimate that was wrong and the
configuration that worked.

**Text, configuration and measurements. Never pixels.** No frames, no source bytes, and no
presigned URLs — those are a customer's asset and a live credential respectively, and neither
belongs in a corpus somebody reads six weeks later. `_redact` enforces the last one rather than
trusting every caller to remember it.
"""

import datetime
import json
import logging
import os
import re
import sys
import time
import traceback

#: A bounded tail also rides in the failure envelope, so CF can classify without fetching
#: anything. **Bounded because the output payload limit is 10 MB and an inline log is exactly
#: the sort of thing that grows until a job silently returns nothing** (playbook §1).
INLINE_TAIL_CHARS = 4000

#: The bundle itself is not payload-limited, but a corpus of unbounded logs is a corpus nobody
#: reads. SeedVR2 is verbose.
BUNDLE_TAIL_LINES = 2000

#: Lines kept from the **start** of a run, whatever happens later.
#:
#: **The explanation lives at the head and the tail is where it was being kept.** A job that OOMs
#: and recovers writes its failure in the first twenty lines and then thousands more succeeding at
#: the fallback configuration — a 929-frame clip would produce roughly 15,000 — so a tail-only
#: buffer returns a bundle full of the model working, over and over, with no trace of what went
#: wrong. Small, because the head is fixed cost on every job and the tail is where a *hang* shows.
BUNDLE_HEAD_LINES = 400

# Anything that looks like a signed URL or a credential. Presigned URLs are the realistic case:
# they arrive in the request, and a traceback that includes the request would carry one.
_REDACTIONS = (
    (re.compile(r"https?://[^\s\"']*[?&]X-Amz-[^\s\"']*", re.I), "<presigned-url-redacted>"),
    (re.compile(r"https?://[^\s\"']*[?&](?:Signature|token|sig)=[^\s\"']*", re.I),
     "<signed-url-redacted>"),
    (re.compile(r"(?i)(secret_access_key|session_token|access_key_id)[\"']?\s*[:=]\s*[\"']?"
                r"[A-Za-z0-9/+=_.-]{8,}"), r"\1=<redacted>"),
)


#: Where the reserve PUT is kept between jobs. **A file, not just a module global**, because the
#: failures it exists for include the ones that take the process with them: a global dies with the
#: interpreter, and the container usually outlives it.
#:
#: `/tmp` inside the container, and that bound is the honest one — a *new* container starts with no
#: reserve, so a worker that has never run a job still cannot report. CF accepts that gap: an
#: endpoint with zero successful jobs is not a subtle signal.
RESERVE_PATH = os.environ.get("CF_DIAGNOSTICS_RESERVE_PATH", "/tmp/cf-diagnostics-reserve")


def remember_reserve(url, path=None):
    """Keep the newest reserve PUT, discarding the previous one. **Never raises.**

    Replaced on every job, so it refreshes on ordinary traffic and needs no call to CF at boot —
    which is the point of the design rather than a convenience: a boot-time fetch would need CF
    reachable exactly when it is least able to tell you it is not.

    Its TTL is hours rather than minutes, because it is for a failure that has not happened yet,
    and that continual renewal is what lets it stay short-lived at all — CF never has to hand out
    something long-lived enough to be a credential. **An idle worker's reserve lapses**, which is
    accepted: a worker not running jobs is not running anything to fail at.
    """
    if not url:
        return False
    try:
        target = path or RESERVE_PATH
        # Written whole and moved into place: a bundle is written on the path where something has
        # already gone wrong, and a half-written URL read there would fail in a way that looks
        # like having no reserve at all.
        temporary = target + ".new"
        with open(temporary, "w") as handle:
            handle.write(url)
        os.replace(temporary, target)
        return True
    except Exception:  # noqa: BLE001 — bookkeeping must never fail a job
        return False


def reserve(path=None):
    """The kept reserve PUT, or None. **Never raises**, for the same reason."""
    try:
        with open(path or RESERVE_PATH) as handle:
            return handle.read().strip() or None
    except Exception:  # noqa: BLE001
        return None


def redact(text):
    """Strip anything credential-shaped. Applied to everything written, without exception.

    A live presigned URL written into a diagnostic outlives the incident it documents, which is
    the reason `CF_storage` states the no-credentials rule for the diagnostics bucket as a rule
    rather than a convention.
    """
    if not text:
        return text
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class _Tee:
    """Writes through to the real stream and keeps the tail. Never raises."""

    def __init__(self, stream, sink):
        self._stream = stream
        self._sink = sink

    def write(self, text):
        try:
            self._sink(text)
        except Exception:  # noqa: BLE001 — capturing must never break the thing being captured
            pass
        return self._stream.write(text)

    def flush(self):
        return self._stream.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


class LogCapture(logging.Handler):
    """SeedVR2's own output, intercepted in-process — **stdout and stderr, not just `logging`**.

    Playbook §9: *"If your progress source is a library's log output, intercept it in-process"* —
    reading it back through RunPod's log API costs the collection lag for nothing. The same
    argument applies to diagnostics, and more strongly: the lines that matter are the ones around
    the exception, and only this process knows when that was.

    **It captured nothing at all for the first two months.** This was a `logging.Handler` on the
    root logger, in a process where the vendored model `print`s every line it emits and this
    worker mostly does too — so `log_tail` went out empty on every error and the diagnostics
    bundle carried no `log` key at all. Nobody noticed, because an empty log looks like a quiet
    run rather than like a capture pointed at the wrong stream.

    The lines that matter are exactly the ones it was missing: on an OOM, the model's own phase
    output — which batch, which resolution, how far in — is the difference between a diagnosable
    failure and a stack trace.
    """

    def __init__(self, limit=BUNDLE_TAIL_LINES, head=BUNDLE_HEAD_LINES):
        super().__init__()
        self.limit = limit
        self.head_limit = head
        self.head = []
        self.lines = []
        self.dropped = 0
        self._partial = ""
        self._streams = None

    def emit(self, record):
        try:
            self._append(self.format(record))
        except Exception:  # noqa: BLE001 — a broken log record must not break the run
            return

    def _append(self, line):
        if len(self.head) < self.head_limit:
            self.head.append(line)
            return
        self.lines.append(line)
        if len(self.lines) > self.limit:
            self.dropped += len(self.lines) - self.limit
            del self.lines[: len(self.lines) - self.limit]

    def _absorb(self, text):
        """Whole lines only, so a progress bar's carriage returns do not each become a line."""
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            line = line.rsplit("\r", 1)[-1].rstrip()
            if line:
                self._append(line)

    def text(self):
        """Head, then a stated gap, then tail. **The gap is stated rather than implied**: a log
        that silently skips is a log that reads as complete, and this project has paid for
        absence looking like success more than for any other single thing."""
        parts = list(self.head)
        if self.dropped:
            parts.append("... {:,} lines dropped between the start of the run and the tail "
                         "below ...".format(self.dropped))
        parts.extend(self.lines)
        return redact("\n".join(parts))

    def tail(self, chars=INLINE_TAIL_CHARS):
        return self.text()[-chars:]

    def __enter__(self):
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(self)
        # Both streams: the vendored model prints its phases to stdout and its warnings to stderr,
        # and an OOM traceback arrives on stderr while the context that explains it is on stdout.
        self._streams = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(sys.stdout, self._absorb)
        sys.stderr = _Tee(sys.stderr, self._absorb)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            logging.getLogger().removeHandler(self)
        finally:
            # **Restored in a `finally`, unconditionally.** A worker that leaves a wrapper on
            # `sys.stdout` after an exception has broken every subsequent job on that worker, and
            # a serverless worker outlives the job that damaged it.
            if self._streams:
                sys.stdout, sys.stderr = self._streams
                self._streams = None
            if self._partial.strip():
                self._append(self._partial.rsplit("\r", 1)[-1].strip())
                self._partial = ""
        return False


def bundle(request_id, hardware, attempts, exception=None, log_text=None, extra=None,
           request=None, rationale=None, warnings=None, job=None, build=None, started=None):
    """The JSON written to CF's diagnostics destination.

    `attempts` is the list this worker builds as it runs — what each one was configured with,
    what it cost and how it ended.

    **The first bundle anyone read could not answer three questions**: what was asked for, what
    the worker expected before it started, and which build produced it. The docstring claimed the
    attempts list "contains both the estimate that was wrong and the configuration that worked";
    it contained the configurations. The estimate lives in `rationale` and was not being written
    at all — on the run that prompted this, the worker predicted 31.37 GB and 12 minutes against a
    reality of 43.01 GB and 33, and that gap is the single most diagnostic number in the job.

    `job` carries RunPod's own identifiers. Without them a bundle cannot be tied back to the
    platform's record of the same job, which is the first thing anyone reaches for.
    """
    body = {
        "request_id": request_id,
        # **When, and by what.** A bundle with no timestamp and no build is a description of
        # something that happened to a machine at an unknown time under unknown code, and this
        # image changed eight times in the two days before the first bundle was read.
        "utc": _now(),
        "build": build,
        "runpod": _runpod_identity(job),
        "hardware": hardware,
        # What was asked for, which nothing recorded. Reconstructing it from the log is possible
        # and is not the same as having it.
        "request": _request_summary(request),
        # What the worker expected before it started — the half of "the estimate that was wrong"
        # that was missing.
        "estimate": rationale,
        "warnings": list(warnings or []),
        "attempts": attempts,
    }
    if started is not None:
        body["elapsed_s"] = round(time.time() - started, 1)
    if exception is not None:
        body["exception"] = {
            "type": type(exception).__name__,
            "message": redact(str(exception))[:4000],
            "traceback": redact("".join(traceback.format_exception(
                type(exception), exception, exception.__traceback__)))[-8000:],
        }
    if log_text:
        body["log"] = log_text
    if extra:
        body.update(extra)
    # **Redacted after serialising, not field by field.** Redacting the exception message and the
    # log covered the two channels anyone thought about, and missed the one that mattered: an
    # `attempts` entry is embedded verbatim, so a presigned URL reaching a structured field went
    # straight through. Found by the check that was written to prove the claim rather than to
    # confirm it.
    #
    # Sweeping the finished JSON is the only form of this that cannot be outflanked by a field
    # somebody adds later. The replacements contain no quotes or backslashes, so the document
    # stays parseable — asserted by the same check, which reads it back.
    return redact(json.dumps(body, indent=2, default=str))


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _runpod_identity(job):
    """RunPod's own handles for this job, so a bundle can be found in the platform's record.

    `RUNPOD_POD_ID` is the worker; the job id comes from the dict the SDK hands the handler. Both
    absent off-platform, which is why they are read defensively rather than indexed.

    **The data centre rides beside the worker** (F-2026-08-19-37's wave). It is the axis the
    `[load]` strip's figures have to be sorted by — one DC streams image layers lazily and pays
    minutes faulting the checkpoint in, another does not — and a run-record carrying a strip
    figure without it records a number that cannot be compared to any other. `datacenter_source`
    is kept for the same reason the banner prints it: an absent value that cannot say what was
    tried is indistinguishable from a value we looked for under the wrong name.
    """
    import hardware  # noqa: PLC0415 — stdlib-only; keeps this module's import list unchanged

    dc, source = hardware.datacenter()
    return {
        "job_id": (job or {}).get("id"),
        "worker_id": os.environ.get("RUNPOD_POD_ID"),
        "endpoint_id": os.environ.get("RUNPOD_ENDPOINT_ID"),
        "datacenter": dc,
        "datacenter_source": source,
    }


#: Fields of the request worth keeping. **Named rather than copied wholesale**, because the
#: request carries `output`, and `output` carries a credential. A bundle that took the request
#: entire would write a temporary key into a different bucket that outlives the incident — which
#: is the one thing `CF_storage` states as a rule for this destination.
#:
#: **The codec surface, `tile_quality` and `schedule` are here because the corpus could not key on
#: them** (`api.md` §6, ruled item 3). Zero of the 82 run records banked before this line carried a
#: codec field of any kind, so "how many jobs shipped h265" was not a question `records/` could
#: answer — every certification of the codec work rested on dated ledger verdicts and nothing
#: queryable. These are the variables we ship; a pile of runs that cannot be keyed on them prices
#: nothing.
#:
#: **The names are the FLATTENED ones, which is what a record reader will not expect.** On the wire
#: the codec block is nested (`params.output.codec`) and `validation.py` flattens it exactly once;
#: `request` here is always the flattened form, so `codec` is a top-level key by the time it
#: reaches this tuple.
#:
#: **`codec` is what was ASKED FOR, not what shipped.** `"source"` is resolved after the probe
#: (`envelope.resolve_codec`), and that resolution is not in this summary — it cannot be, because
#: this is the request. A record whose `request.codec` reads `source` says which codec was
#: *requested*; nothing in the record says which one came out. Filed to the gate rather than fixed
#: here: the record's `output` block is scoped to "size and frame counts only" by its own comment
#: in `handler.py`, and widening it is a spec decision.
_REQUEST_FIELDS = (
    "target_short_edge_px", "output_size", "color_correction",
    "keep_audio", "allow_oom_retry", "execution_timeout_ms", "force_rung", "force_batch_size",
    "force_chunk_size", "force_temporal_overlap", "keep_alpha_in_model", "debug",
    # The codec surface — `api.md` §2c, `contract.md` §1 is the law.
    "codec", "crf", "preset", "head_keyframes", "keyframes", "keyframe_frames",
    "keyframe_seconds",
    # Named in the same ruling, and levers that move the plan rather than the picture.
    "tile_quality", "schedule",
)


def _request_summary(request):
    if not request:
        return None
    summary = {field: request.get(field) for field in _REQUEST_FIELDS
               if request.get(field) is not None}
    if request.get("derive"):
        summary["derive"] = [entry.get("role") for entry in request["derive"]]
    return summary


def should_write(failed, retried):
    """On failure, and on any job that retried — including one that recovered."""
    return bool(failed or retried)
