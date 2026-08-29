"""A record of every run, written by the worker, so the corpus stops depending on who watched.

**The failure bundle answers "what went wrong"; this answers "what happened".** They are different
kinds and they stay different kinds (F-2026-08-19-36, and the contract amendment of the same day).
A bundle's *presence* is certified shorthand for "a run struggled" — the R-series verdicts read it
that way, in evidence — so widening bundles to cover successful runs would retroactively change
what those verdicts were reading. The run-record therefore lands under its own `runs/` prefix, and
its presence means only that a run happened.

**Why the worker writes it and not the harness.** Every calibration row this project owns was
banked client-side by `run_one.py`, which works exactly as long as a person is watching through
our own tool. Three failures in one day showed what that costs: the first 8K customer delivery —
the most expensive single measurement in the ledger — banked `build: None` and a wall clock of
357 s for a job that ran 4147 s, because it was recovered through `--attach` rather than watched
(F-2026-08-19-35). And the callers this worker exists for will not run our harness at all. A
record the worker pushes is complete by construction: it does not care who was watching, whether
the client survived, or whether anyone was there.

What it feeds, concretely: §8b's per-card time rows (the 491-versus-69-minute prediction gap is a
missing-row problem, not a formula problem), the host tail-term anchors, and per-frame pricing per
card — each of which currently improves only when a human remembers to bank a row.

**Metadata only, never content.** No frames, no source bytes, no presigned URLs, no customer text.
The body is swept through `diagnostics.redact` before it leaves, which is defence in depth rather
than the primary control: nothing here is supposed to be able to carry a credential in the first
place, and the sweep is what makes that true of fields somebody adds later.

**The write can never fail a job.** Same posture as progress emission, and for a stronger reason:
this runs on the success path too, so an exception here would turn a delivered master into a
failed job over a bookkeeping object. Every entry point returns a value and raises nothing.

**The address arrives with the request, exactly as the failure bundle's always has** (CF, ruled
2026-08-19, superseding this module's first design). A presigned PUT URL in a `run_record` field
beside `diagnostics`, its key under `runs/`, minted by whoever builds the request — our harness
today, CF's front-end tomorrow.

An endpoint-provisioned standing credential (`RUNS_S3_*`) was built first and rejected, and the
reasons are worth keeping: the caller owns its telemetry destination and should not have to
discover ours, and a worker holding a long-lived write credential to somebody else's bucket is a
durable liability in exchange for one object per job. A presigned URL is scoped to one key, one
verb and one window, and it arrives and expires with the work.

**The window has to outlast the job.** Whoever mints the URL matches its expiry to the endpoint's
execution timeout: a URL that dies before a long job finishes manufactures record loss on exactly
the runs most worth recording — the 8K ones, which is where this finding came from in the first
place.

**Absent field is a supported state, not an error.** No URL means the record is skipped with one
line in the log. A worker that refused to run without somewhere to file its paperwork would be a
worse worker.
"""

import json

import diagnostics

#: The request field carrying this record's presigned PUT. Named beside `diagnostics` on the wire
#: because it is the same mechanism answering a different question, and validated as a string like
#: every other URL the request carries.
REQUEST_FIELD = "run_record"

#: The prefix the minted key sits under. A constant here because the worker does not build the
#: key — the minter does — but the worker's own tests and the harness both need one name for it,
#: and the whole point of a separate kind is that it can be enumerated without meeting bundles.
PREFIX = "runs/"

#: Long enough that a slow object store does not lose the record, short enough that it cannot
#: meaningfully extend a job that has already finished its real work.
CONNECT_TIMEOUT_S = 5
READ_TIMEOUT_S = 20


#: The configuration fields an attempt carries. Named explicitly rather than taken as "everything
#: that is not a measurement": an attempt also holds outcomes and peaks, and a `plan` block that
#: quietly grew a measurement into it would be read as a decision the planner made.
CONFIGURATION_KEYS = (
    "batch_size", "chunk_size", "blocks_to_swap", "temporal_overlap",
    "output_short_edge_px", "vae_encode_tiled", "vae_decode_tiled",
    "dit_offload_device", "vae_offload_device", "tensor_offload_device",
    "swap_io_components", "crf", "tile_quality", "schedule", "name",
)


def _configuration_of(attempts):
    """The configuration the last attempt actually ran, or None when none ever started."""
    for attempt in reversed(attempts or []):
        if not isinstance(attempt, dict):
            continue
        # A nested `plan` if one is ever added; otherwise the flattened fields as they stand today.
        nested = attempt.get("plan")
        if isinstance(nested, dict) and nested:
            return nested
        chosen = {k: attempt[k] for k in CONFIGURATION_KEYS if k in attempt}
        if chosen:
            return chosen
    return None


#: The two states a record is written in (F-2026-08-20-43). `stub` is filed once the plan exists
#: and before any GPU phase; `final` overwrites it at exit, same key, same URL.
#:
#: **An unclosed stub is the finding.** A cgroup SIGKILL writes no bundle, raises no exception
#: and returns no envelope — F-41 died twice that way — so in the corpus today a run that was
#: killed and a run that never happened are the same absence. A stub turns the first into a
#: record that says what was planned, on what hardware, at what host readings, and then stops.
#: That is the only artefact that class of death can leave.
PHASE_STUB = "stub"
PHASE_FINAL = "final"


def build_stub(build_identity, machine, request=None, rationale=None, source=None,
               host_banners=None, job=None, started_utc=None):
    """The record filed on the way in: what this run is about to attempt.

    Deliberately thin. It carries the plan, the machine and the request echo — everything already
    known before the expensive part — and nothing that only exists afterwards. A stub that tried
    to guess at outcomes would be a record that disagrees with its own overwrite.
    """
    return build(
        "started", build_identity, machine, request=request, rationale=rationale, source=source,
        host_banners=host_banners, job=job,
        timings=None if not started_utc else {"started_utc": started_utc},
        phase=PHASE_STUB)


def _shadow_estimate(machine, output, rationale, source):
    """The shadow model's answer for this run, or its stated absence. **Never raises.**

    Reads the DELIVERED frame count and the DELIVERED plane where they exist, because the record
    is written after the fact and an estimate keyed on what was predicted would be comparing the
    model against a different job from the one that ran. Falls back to the rationale's geometry
    on a run with no output — a refusal, a crash — so a record still carries the number the model
    would have given, which is what makes a refused job comparable with a delivered one.
    """
    try:
        import timemodel  # noqa: PLC0415 — leaf module; keeps the import graph honest

        out = output or {}
        rat = rationale or {}
        # **BOTH FROM THE SAME RUN, or neither.** The first version preferred DELIVERED frames
        # and PLANNED pixels, with opposite precedence on the two inputs — so on any job where
        # the two disagree it priced delivered frames at a plane the model did not produce.
        # They do disagree, and the worker says so out loud: handler.py warns "the model
        # produced WxH where WxH was computed from target_short_edge_px=N; the model rounds to
        # its own grid and the output written is the model's size". A delivered plane LARGER
        # than the planned one yields an estimate BELOW the true product, which breaks the one
        # property this model promises — true >= estimate — through an input mismatch rather
        # than through arithmetic.
        #
        # So: delivered frames with the delivered plane, else planned frames with the planned
        # plane, and never one of each.
        delivered_frames = out.get("frames_written")
        delivered_pixels = ((out.get("width") or 0) * (out.get("height") or 0)) or None
        # **`is not None`, not truthiness.** `frames_written: 0` is a run that opened an output
        # and wrote nothing — a fact — and `or` read it as absent and silently substituted the
        # PLANNED count, pricing an estimate off a number nothing delivered.
        if delivered_frames is not None and delivered_pixels:
            frames, pixels = delivered_frames, delivered_pixels
        else:
            frames = (source or {}).get("estimated_frames")
            pixels = rat.get("output_pixels")
        return timemodel.predict(
            (machine or {}).get("gpu_name"), frames, pixels,
            still=bool((source or {}).get("still")))
    except Exception as exc:  # noqa: BLE001 — a shadow must never cost a record
        return {"model": "v0", "estimate_seconds": None,
                "absent_because": "{}: {}".format(type(exc).__name__, exc)}


def build(status, build_identity, machine, request=None, rationale=None, source=None,
          attempts=None, output=None, load_strip=None, host_banners=None, timings=None,
          progress=None, job=None, error=None, warnings=None, phase=PHASE_FINAL):
    """The record body. Metadata only — every argument here is a number, a name or a shape."""
    body = {
        "kind": "run-record",
        # **Which of the two writes this is.** A reader finding `stub` in the corpus is holding a
        # run that began and never reported: the SIGKILL class, visible at last. A reader who
        # cannot tell a stub from a truncated final would draw the opposite conclusion from the
        # same bytes, which is why this is a field and not an inference from what is missing.
        "record_phase": phase,
        # **Version the shape, because this one is meant to be read years from now** by something
        # that was not written yet. A corpus whose entries cannot say which shape they are is a
        # corpus that can only be parsed by the code that wrote it.
        "record_version": 1,
        # **The id, in the BODY and not only in the key** (`api.md` §6, ruled item 2). CF mints
        # the key and the id lives in the name it chose; a record harvested into a fitting
        # directory, or renamed, or read out of a bucket listing, joined to nothing at all. This
        # document's own docstring says it is meant to be read years from now by something not
        # written yet, and until this line the only thing that could identify it was its filename.
        #
        # **Top level rather than inside `request`**, which is the summary of what was ASKED for.
        # The id is what this record IS, not a parameter of the job — and a reader looking for it
        # should not have to know that the summary exists, or that the summary drops `None`.
        #
        # **`None` is written rather than omitted.** A run refused before validation has no id to
        # carry, and a key present-but-null says that; an absent key says the record predates this
        # line, which is a different fact.
        "request_id": (request or {}).get("request_id"),
        "utc": diagnostics._now(),
        "status": status,
        "build": build_identity,
        "runpod": diagnostics._runpod_identity(job),
        # The card is the axis every constant in the registry is keyed on, so it is named at the
        # top rather than buried in the hardware block a reader has to know to open.
        "gpu": (machine or {}).get("gpu_name"),
        "hardware": machine,
        "request": diagnostics._request_summary(request),
        # What the planner decided and why — the half that makes a measurement re-derivable
        # instead of merely recorded. Lifted off the *winning* attempt rather than the first,
        # because a run that stepped down was measured at the configuration that finished, and
        # pairing a rung's name with another rung's peak is a corruption this ledger has already
        # met once.
        "plan": _configuration_of(attempts),
        "rationale": rationale,
        # **THE SHADOW TIME MODEL, computed here and consumed nowhere** (CF, 2026-08-28;
        # `docs/gate/time-model.md` §9). Both predictions are computed on every job, only the old
        # one is consumed, and both are recorded — `rationale.predicted_seconds` above is the
        # live lookup and still the only number any decision reads.
        #
        # **Assembled at the record rather than in the planner, deliberately.** The planner is
        # what the deadline gate consumes; a shadow number computed there would be one refactor
        # away from being read, and `timemodel` is a leaf whose only importer is this line. The
        # import graph is what keeps "shadow" true, rather than a comment asking people to be
        # careful.
        #
        # **It is computed even when it cannot answer**, so an unmeasured card produces a ROW
        # saying absent rather than no row at all. That row is the one that tells CF its coverage
        # is missing, and it is the whole defect the new model replaces — the old one borrowed a
        # rate from whatever was furthest away and reported it as measured.
        "time_model_shadow": _shadow_estimate(machine, output, rationale, source),
        "source": source,
        "output": output,
        # **The strip that used to be silent**, measured in halves because its two costs have
        # different causes: a CPU-and-page-cache-bound import, and a checkpoint read whose price
        # is the host's storage (F-2026-08-19-31).
        "load_strip": load_strip,
        "host": host_banners,
        "timings": timings,
        "attempts": attempts or [],
        "warnings": list(warnings or []),
    }
    if progress is not None:
        body["seconds_per_frame"] = progress.seconds_per_frame()
    if error:
        body["error"] = error
    return diagnostics.redact(json.dumps(body, indent=2, default=str, sort_keys=True))


def write(document, url, log=print, label="run-record"):
    """PUT the record to the caller's presigned URL. **Never raises, never fails a job.**

    Returns True if it landed, False otherwise. Three outcomes, all reported and none fatal:
    written; skipped because the request carried no `run_record` field; or failed because the
    object store said no. The middle one is a supported state and says so, because a line that
    reads like an error every time a correctly-configured job runs is a line people learn to
    ignore — and this one will appear on every job until CF's front-end starts minting the field.

    A single PUT rather than a client, matching `storage.put_diagnostics` exactly: one object
    against a URL that already names the bucket, the key and the window it is good for.

    **The same URL is written twice on a healthy job** (F-2026-08-20-43): a stub on the way in
    and the full record at exit, the second overwriting the first. That is deliberate and it is
    why the key must be stable — a second object would make an unclosed stub indistinguishable
    from a run that filed twice, which is the one distinction this whole design exists to draw.
    `label` only names which write is speaking in the log; both go to the same place.
    """
    try:
        if not url:
            log("[{}] skipped: the request carried no {} URL. This is not an error — "
                "the record is optional and the job is unaffected.".format(label, REQUEST_FIELD))
            return False

        import requests  # noqa: PLC0415 — already a dependency; imported here to match storage

        response = requests.put(
            url,
            data=document.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
        )
        response.raise_for_status()
        log("[{}] wrote {:,} bytes".format(label, len(document)))
        return True
    except Exception as exc:  # noqa: BLE001 — see the docstring; this must never fail the job
        # Named, not swallowed. A record that silently never appears is the same class of defect
        # as the empty log this project carried for two months. **An expired URL lands here**,
        # which is why the minter matches the expiry to the endpoint's execution timeout: the
        # longest jobs are both the most valuable to record and the most likely to outlive a
        # short window.
        try:
            log("[{}] NOT written ({}: {}). The job is unaffected; if this is a signature "
                "or expiry error the URL died before the job did.".format(
                    label, type(exc).__name__, str(exc)[:200]))
        except Exception:  # noqa: BLE001 — a last resort that raises is not one
            pass
        return False
