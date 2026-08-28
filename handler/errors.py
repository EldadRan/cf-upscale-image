"""Refusal codes, and the error type the handler renders into the contract's error envelope.

Two tables, because CF's retry classification reads them differently: the first will fail
identically forever, the second may not. A worker reporting a bad field as `internal` costs CF
three retries and a wrong diagnosis.

**This worker needs a third answer that the media worker's two tables cannot express**, and it
is the one CF actually asked for. When work does not fit, "retry" and "never retry" are both
wrong: the useful answer is *a larger card would work, and here is by how much*, whose remedy is
not a retry at all — it is routing the job to a different endpoint. That is carried as a
`remedy` field beside the code rather than as a third table, so a consumer that only knows the
two tables still classifies correctly and one that knows better can act. See `Remedy` below.
"""


class Remedy:
    """What would make this job succeed. The actionable half of a refusal.

    Present on any refusal where the worker knows something about what would help, absent
    otherwise. CF's question on a job that did not fit is *would sending this anywhere again
    help?*, and only the worker can answer it — it knows what it tried, on what card, and how
    far short it fell.
    """

    #: Nothing will help. The request is beyond this model at any size. Refuse it and tell the
    #: caller; a retry anywhere is spend with a known outcome.
    NONE = "none"
    #: The same hardware would probably work on a second attempt — a transient. A fetch failed,
    #: a neighbour took the memory.
    RETRY_SAME = "retry_same"
    #: A larger card would work. **This is the valuable one**, and CF's answer to it is not a
    #: retry: it is routing to an endpoint with stronger hardware, which becomes possible the
    #: moment one exists. Always accompanied by `shortfall`, because `needed ~62 GB at the most
    #: conservative configuration, had 48` is actionable and `out of memory` is not.
    LARGER_GPU = "larger_gpu"
    #: The same request, on the same card, with a bigger `execution_timeout_ms`. **Unlike
    #: `larger_gpu`, CF can grant this one by resending** — which is why `deadline_exceeded`
    #: carries `predicted_seconds` and the limit it was measured against rather than leaving the
    #: caller to double blindly.
    #:
    #: **It was emitted for a release before it was declared here** (CF, 2026-08-28).
    #: `deadline_exceeded`'s own docstring below specifies it, `estimator.py` emits it, and rung 1
    #: asserts it — while `ALL` named three values. Nothing enforces `ALL`, so nothing broke at
    #: runtime; what it would have broken is anything built ON `ALL`, which is the obvious thing
    #: to build: a published vocabulary missing a value it emits, and a validator that would
    #: reject a legitimate refusal.
    LONGER_DEADLINE = "longer_deadline"

    #: **The published vocabulary, and the rule about it is directly below.** Adding a value here
    #: is a contract change: doc-first, the gate's to rule, and this one was ruled by CF on
    #: 2026-08-28 as part of the wave that found it.
    #:
    #: **`planner.py`'s `larger_host` is NOT in this set and is not an omission.** It is a second
    #: vocabulary under the same field name, on the planner's terminal rather than on a
    #: `cf_error`, and collapsing the two would be asserting an equivalence nobody has ruled.
    ALL = (NONE, RETRY_SAME, LARGER_GPU, LONGER_DEADLINE)


class WorkerError(Exception):
    """A refusal with a contract code. Anything else becomes `internal`."""

    def __init__(self, code, message, remedy=None, shortfall=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.remedy = remedy
        #: dict — what the work needed against what the machine had, and at what configuration.
        self.shortfall = shortfall

    def to_dict(self):
        # `cf_error`, not `error`. RunPod's SDK pops `error` out of a handler's return value and
        # promotes it to the job envelope, so a refusal returned under that name reaches CF as
        # COMPLETED with no output at all — no code, nothing for the retry classification to
        # read. `refresh_worker` is popped too; assume any short generic key may be reserved.
        #
        # The prefix earns its keep beyond avoiding the collision: `error` at the RunPod envelope
        # means the platform could not run the job, while `cf_error` inside the output means the
        # job ran and the answer was no. A caller's retry logic needs to tell those apart.
        body = {"code": self.code, "message": self.message}
        if self.remedy is not None:
            body["remedy"] = self.remedy
        if self.shortfall is not None:
            body["shortfall"] = self.shortfall
        return {"cf_error": body}


# ── The caller got it wrong. Will fail identically forever, so never retryable ──────────────

FIELD_NOT_SUPPORTED = "field_not_supported"
MISSING_REQUIRED_FIELD = "missing_required_field"
INVALID_FIELD_VALUE = "invalid_field_value"

#: The source fetched but is not usable — no video stream, undecodable, corrupt, or a container
#: the decoder will not open. The same bytes will fail the same way forever.
INVALID_SOURCE = "invalid_source"

#: The work does not fit and no configuration this worker can reach would make it fit. Carries
#: `remedy` — `none` where nothing helps, `larger_gpu` with a `shortfall` where a bigger card
#: would. It sits in *this* table because the request, as sent, cannot be served by this
#: endpoint however many times it is tried. The `larger_gpu` case is not a contradiction: what
#: CF does with it is change where the job goes, not repeat it.
CAPACITY_EXCEEDED = "capacity_exceeded"

#: **The job cannot finish inside the time it was given**, refused before the GPU is spent rather
#: than discovered when RunPod kills the container. `execution_timeout_ms` is a hard kill: no
#: master is written, no error is returned, every second is billed and the caller gets `TIMED_OUT`
#: with nothing attached.
#:
#: Named for `capacity_exceeded`'s shape rather than for what happened, because neither has
#: happened: both mean *this request, as sent, exceeds a stated limit*, and both are refused in
#: seconds. `remedy` is `longer_deadline` — and unlike `larger_gpu`, **CF can grant this one by
#: resending**, which is why the refusal carries `predicted_seconds` and the limit it was measured
#: against. A caller that has to double blindly is being told less than the worker knows.
DEADLINE_EXCEEDED = "deadline_exceeded"

#: **The host slice will not hold this job, discovered while the job is still alive**
#: (F-2026-08-20-42, ruled with the host wave). Distinct from `capacity_exceeded`, which is about
#: the card, and refused before the GPU is spent: this one can only be known part-way, because
#: it is the *host* model being wrong about a run already in flight.
#:
#: It exists because the alternative is nothing at all. A host breach is a cgroup SIGKILL — no
#: exception, no bundle, no walk, no envelope — and RunPod then restarts the container with no
#: resume, so a deterministic host overrun is a money-burning retry loop that writes no evidence.
#: F-41 died that way twice on the same plan. Refusing while alive converts that into a refusal
#: with a bundle, a run-record and a measured drift: **a calibration datum instead of an exit
#: code 137.**
#:
#: `remedy` is `larger_gpu` — the host slice scales with the card tier on this platform, so the
#: answer is the same one: route it somewhere bigger. `shortfall` carries what the model
#: predicted against what the container is actually using, which is the number that reprices the
#: constants rather than merely explaining the refusal.
HOST_CAPACITY_EXCEEDED = "host_capacity_exceeded"


# ── Something failed. May not fail again; CF's ordinary retry classification applies ─────────
#
# The line between the two tables is where the bytes stopped being in doubt: a GET that errors
# or comes up short of its content length may succeed next time, but bytes that arrived whole
# and will not decode are the same bytes forever.

SOURCE_FETCH_FAILED = "source_fetch_failed"

#: **Carries `retry_same`, and it took losing a job to notice it carried nothing**
#: (F-2026-08-20-39). An 8 255 s run came back with `remedy: null` on the one refusal where the
#: caller's next move was obvious and cheap — resubmit; the work is reproducible and the
#: credential will be fresh. A null there does not read as "no remedy exists", it reads as "this
#: worker has nothing to say", and the harness printed the two identically until the same finding
#: made it stop.
#:
#: `retry_same` rather than a new value: the vocabulary is `Remedy.ALL` and adding to it is a
#: contract change, which is doc-first and the gate's to rule. And it is not a compromise — an
#: expired write credential is precisely a case where **the same hardware works on a second
#: attempt**, because the credential is minted per request and a resubmit carries a new one. What
#: the caller must not do is retry the *same* request unchanged, and the message says so.
OUTPUT_WRITE_FAILED = "output_write_failed"

#: An OOM the worker caught, on a job whose estimate said it should have fit. Distinct from
#: `capacity_exceeded`, and the distinction is the whole point: this one means the *estimator*
#: was wrong, which is a data point rather than only an incident. Carries `remedy` and, where
#: the exception gave enough to compute one, `shortfall`.
OUT_OF_MEMORY = "out_of_memory"

INTERNAL = "internal"


#: Every code this worker can return, so a test can assert the set rather than a sample and a
#: harness can reject an expectation naming a code that does not exist. A suite whose expected
#: values are not validated against the source of truth goes stale in the rung that does not
#: own them.
NEVER_RETRYABLE = (
    DEADLINE_EXCEEDED,
    FIELD_NOT_SUPPORTED,
    MISSING_REQUIRED_FIELD,
    INVALID_FIELD_VALUE,
    INVALID_SOURCE,
    CAPACITY_EXCEEDED,
    # Never retryable **on the same endpoint**: the host slice is a property of the machine, so
    # an identical resend meets an identical slice. Routing is what changes the answer.
    HOST_CAPACITY_EXCEEDED,
)
RETRYABLE = (
    SOURCE_FETCH_FAILED,
    OUTPUT_WRITE_FAILED,
    OUT_OF_MEMORY,
    INTERNAL,
)
ALL_CODES = NEVER_RETRYABLE + RETRYABLE

#: **Codes that mean the worker DECIDED, rather than the worker broke.** `handler.py` classifies an
#: attempt's outcome with this: a deliberate refusal is recorded as `refused` and everything else
#: as `error`, because the corpus has to be able to tell a judgement from a crash — the whole value
#: of refusing while alive is that the record explains itself (F-2026-08-20-46).
#:
#: **`INVALID_FIELD_VALUE` is in here because of WHERE it can be raised, not what it means.** Every
#: other one is raised at the door, before an attempt exists, and never reaches the classifier. The
#: keyframe check is the exception: it can only run at the END of a successful encode, because the
#: true frame count exists nowhere earlier — so it is a refusal on a run that did everything right,
#: and recording that as a crash would put a caller's typo in the same bucket as a traceback.
DELIBERATE_REFUSALS = (HOST_CAPACITY_EXCEEDED, INVALID_FIELD_VALUE)
