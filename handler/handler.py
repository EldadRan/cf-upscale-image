"""RunPod entrypoint for CF's SeedVR2 upscale worker.

The order below is the contract rather than a preference (§4):

    fetch → decode → estimate → upscale → **write the master** → derives → manifest

**Nothing incomplete is promoted, but an incomplete job must never cost the inference.** A failed
derive is recoverable from the master; a master that was never written costs hours of GPU to
reproduce. So a crash between any two of those steps leaves CF something to re-run the cheap half
against, and `artefacts_written` says what landed so a recovery knows what it is completing
rather than redoing.

**One image, two endpoints**, and the *processing* is genuinely medium-blind: one frame in, one
frame out, the same estimator, the same model call. **Delivery is not.** Three things branch, and
each is a property of the container rather than a preference:

  - the master is a lossless image for a still, because `yuv420p` cannot carry alpha at all and an
    H.264 still would be lossy again under every image derive taken from it;
  - alpha is retained for a still and warned about for a video, for the same reason;
  - `proxy` is refused on a still, a duration-bounded video being meaningless over one frame.

What decides "still" is that the **source is an image**, not that it has one frame. A one-frame
MP4 stays a video: it has a rate, it can carry audio, and a frame count is the number
`docs/decisions.md` 0.2 says not to act on.
"""

import datetime
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

# **`--dry-run-walk`, gated above the imports it must not trigger.** The walk is pure arithmetic
# over `solver`, and the acceptance kit has to be able to ask it on a laptop; `derives` pulls
# numpy and `encoder` the media stack, neither of which a walk needs. Hence here rather than beside
# `runpod.serverless.start` — the alternative is an entry point that cannot run where the spec is
# being argued about.
if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--dry-run-walk":
    import dryrun

    sys.exit(dryrun.main(sys.argv[2]))

import derives
import diagnostics
import encoder
import envelope
import errors
import estimator
import hardware
import hostguard
import keys
import manifest as manifest_module
import phasewatch
import pipeline
import planner
import probe
import solver
import progress as progress_module
import runrecord
import storage
import validation
from errors import FIELD_NOT_SUPPORTED, WorkerError

WORKER_VERSION = os.environ.get("WORKER_VERSION", "0.1.0-dev")

#: The one checkpoint this image ships. A build parameter, not a runtime choice: the weights are
#: too large to bake two, so a different checkpoint is a different build of this repo. CF records
#: it as `routing.provider_model`, per endpoint.
MODEL_BUILD = os.environ.get("SEEDVR2_MODEL", "seedvr2_ema_7b_fp16.safetensors")


def build_identity():
    """What code produced this result, in a form that survives the run.

    **Every number measured before this existed rests on an assertion.** The ten-run calibration
    campaign of 2026-08-17 reports `"image": null` in every bundle, because `IMAGE_TAG` and
    `WORKER_IMAGE` were read here and set nowhere — so the campaign's claim that all ten runs
    share one image comes from "the endpoint was not re-created", not from anything the worker
    saw. A registry keyed on that is keyed on a memory.

    The image cannot know its own digest: the digest is computed when the image is pushed, which
    is after the last layer is sealed. What it can know is the immutable reference it was pushed
    under — `ghcr.io/owner/repo:sha-<commit>`, a tag CI never reuses — and the commit that reference
    names. A digest is one registry lookup away from either, and neither can drift.

    **Every key is always present, null when unknown.** A local build has no honest answer for
    most of these, and a placeholder string that looks like one is worse than a null; but an
    absent key is worse still, because "this build could not identify itself" and "nobody thought
    to ask" then read identically to whoever finds the bundle later.
    """
    return {
        # `IMAGE_TAG` first, then `WORKER_IMAGE`: the second is what the diagnostics path has
        # always looked for, and is kept so an endpoint that sets it by hand still works.
        "image": (os.environ.get("IMAGE_TAG") or os.environ.get("WORKER_IMAGE")),
        "commit": os.environ.get("BUILD_COMMIT"),
        "built_utc": os.environ.get("BUILD_UTC"),
        # The pinned vendored source, which is half of what any VRAM figure is a measurement of.
        "seedvr2_commit": os.environ.get("SEEDVR2_COMMIT"),
        "worker_version": WORKER_VERSION,
        "model": MODEL_BUILD,
        # **Which constants planned the run, beside which build carried them.** CF compares this
        # against the version its own embedded predicate reports and alerts when the two drift
        # apart, which is the ratified use rather than a nicety.
        "registry_version": planner.REGISTRY_VERSION,
    }


#: Printed once per container, on the first job it handles. **Once, not per job**: it describes
#: the machine rather than the work, and a line repeated on every request is a line people learn
#: to skip.
_SAID_BOOT = []

#: Every `[host]` reading this job took, in order — **`phasewatch.BANNERS`, not a second list**.
#: It moved there when the per-phase readings joined the four this module banked by hand: two
#: lists would have meant two orders, and the corpus is read as a series. Reset per job by
#: `handle`, because a worker serves many jobs and a record carrying the previous one's banners
#: would be worse than one carrying none.
_HOST_BANNERS = phasewatch.BANNERS


def handle(job_input, job=None):
    started = time.time()
    machine = hardware.read()

    # **How many cores this container may actually use, said out loud before anything else.**
    # The phase-4 tail runs at one core's worth while thirty sit idle, and the first question
    # that investigation has to answer is how many cores there were to be idle — a number that
    # was, until now, only obtainable by someone thinking to run `nproc` on a live worker during
    # a tail. Now every log carries it.
    if not _SAID_BOOT:
        _SAID_BOOT.append(True)
        print(phasewatch.boot_banner())

    # **Per job, not per worker.** A worker serves many jobs and this list is module-level, so a
    # record carrying the previous job's host readings would be a measurement attributed to the
    # wrong run — the exact defect class the build identity exists to prevent.
    del _HOST_BANNERS[:]

    # **Kept before the request is validated**, and read straight off the raw input rather than
    # off `request`. A request that fails validation never produces a `request`, and a job that
    # arrives while the reserve is stale is exactly the traffic that refreshes it — so taking it
    # only from validated jobs would drop the refresh on the requests most likely to precede
    # trouble.
    diagnostics.remember_reserve(
        job_input.get("diagnostics_reserve") if isinstance(job_input, dict) else None)

    # **Validation is inside the envelope, not in front of it.** It sat outside the try below
    # until rung 1 caught it: a `WorkerError` raised here escaped `handle`, and the outer
    # `handler()` turned it into `internal` — so every bad field was reported as a worker fault.
    # That is precisely the failure the two error tables exist to prevent, and it costs CF three
    # retries and a wrong diagnosis on a request that will fail identically forever.
    # **Constructed BEFORE validation, because the validation refusal is a response shape and
    # `_decorate` needs it.** It depends only on `job`, so there is nothing to wait for; what
    # made it sit lower was that nothing above it emitted. It still emits nothing above -- the
    # decorator reads `progress.emitted`, which is empty on this path and correctly reports no
    # `progress_emitted` key at all.
    progress = progress_module.Progress(job=job)

    try:
        request = validation.validate(job_input)
    except WorkerError as exc:
        # **Through `_decorate`, not around it.** This built its two fields by hand, so the one
        # response shape the debug gate itself produces -- a refusal for sending a lever without
        # `debug` -- was the one shape carrying no `debug` key. `_decorate`'s own docstring
        # claims every shape passes through it; this is what makes that true.
        #
        # **`job_input` rather than `request`, because `request` does not exist here**: this is
        # the branch where validation raised, so the only reading of `debug` available is the
        # raw one. A caller who sent `debug: true` and then a malformed lever is recorded as the
        # debug run they were.
        return _decorate(exc.to_dict(), machine, [], [], progress, started,
                         debug=(job_input or {}).get("debug"))

    warnings = []
    attempts = []
    # **What the diagnostics bundle needs, filled in as the run learns it.** The estimator's
    # rationale is built inside `_run` and the bundle is written out here, so without somewhere
    # shared to put it the most diagnostic number in a failed job — what the worker expected
    # before it started — could not be written at all.
    trace = {}
    workdir = tempfile.mkdtemp(prefix="cf-upscale-")

    # **The outcome, recorded where all three exits can reach it.** The run-record is written in
    # the `finally` below because that is the only point every run passes through — delivered,
    # refused and crashed alike — and "every run" is the whole content of F-2026-08-19-36. A
    # record written on the success path only would rebuild, in a new place, exactly the sampling
    # bias the finding exists to remove.
    outcome = {"status": "internal", "error": None}
    with diagnostics.LogCapture() as captured:
        try:
            response = _run(request, job, machine, warnings, attempts, workdir, progress,
                            captured, started, trace)
            outcome["status"] = "refused" if response.get("cf_error") else "ok"
            outcome["error"] = response.get("cf_error")
            return response
        except WorkerError as exc:
            _write_diagnostics(request, machine, attempts, exc, captured, failed=True,
                               trace=trace, warnings=warnings, job=job, started=started)
            payload = exc.to_dict()
            payload["cf_error"]["log_tail"] = captured.tail()
            outcome["status"] = "refused"
            # The code and the reason, never the log tail: this is a record, not a bundle, and a
            # tail is the bundle's job.
            outcome["error"] = {k: v for k, v in payload["cf_error"].items() if k != "log_tail"}
            return _decorate(payload, machine, attempts, warnings, progress, started,
                             debug=request.get("debug"))
        except Exception as exc:  # noqa: BLE001 — a job must return an envelope, never raise
            traceback.print_exc()
            _write_diagnostics(request, machine, attempts, exc, captured, failed=True,
                               trace=trace, warnings=warnings, job=job, started=started)
            payload = {"cf_error": {
                "code": errors.INTERNAL,
                "message": "{}: {}".format(type(exc).__name__, exc),
                "log_tail": captured.tail(),
            }}
            outcome["status"] = "internal"
            outcome["error"] = {"code": errors.INTERNAL,
                                "message": "{}: {}".format(type(exc).__name__, exc)}
            return _decorate(payload, machine, attempts, warnings, progress, started,
                             debug=request.get("debug"))
        finally:
            _write_run_record(outcome, request, machine, attempts, warnings, progress,
                              trace, job, started)
            shutil.rmtree(workdir, ignore_errors=True)


def _write_run_record_stub(request, machine, rationale, source, job, started):
    """File the **stub** record: what this run is about to attempt (F-2026-08-20-43).

    **Written where the plan first exists, not at literal job entry.** A stub carrying no plan
    would say only that a job arrived, which the platform already knows; the value is naming the
    configuration that was about to run when the run stopped reporting. Everything after this
    line — the load strip, every model phase, the drain — is where a cgroup SIGKILL lives, and
    the stub is upstream of all of it.

    Costs one small PUT on the way in. Never raises, for the same reason the final one does not:
    this must not be able to turn a delivered master into a failed job.
    """
    try:
        url = (request or {}).get(runrecord.REQUEST_FIELD)
        if not url:
            return
        document = runrecord.build_stub(
            build_identity(), machine, request=request, rationale=rationale, source=source,
            host_banners=list(_HOST_BANNERS), job=job,
            started_utc=datetime.datetime.fromtimestamp(
                started, datetime.timezone.utc).isoformat())
        runrecord.write(document, url, label="run-record/stub")
    except Exception as exc:  # noqa: BLE001 — a record must never cost a delivered master
        print("[run-record] stub NOT assembled ({}: {}). The job is unaffected.".format(
            type(exc).__name__, str(exc)[:200]))


def _write_run_record(outcome, request, machine, attempts, warnings, progress, trace, job,
                      started):
    """Assemble and file the run-record. **Never raises** — see `runrecord`'s own posture.

    Wrapped even though `runrecord.write` cannot raise, because *assembling* the body reads a
    dozen fields off structures that a crashed run may have left half-built, and the whole point
    of writing this in a `finally` is that it runs on exactly those jobs.
    """
    try:
        source = (trace or {}).get("source") or {}
        document = runrecord.build(
            outcome.get("status"),
            build_identity(),
            machine,
            request=request,
            rationale=(trace or {}).get("rationale"),
            source=source,
            attempts=attempts,
            output=(trace or {}).get("output"),
            load_strip=(trace or {}).get("load_strip") or None,
            host_banners=list(_HOST_BANNERS),
            timings={
                "wall_s": round(time.time() - started, 1),
                # The worker's own clock, which is the only one that is not somebody else's view
                # of this job — and the figure F-2026-08-19-35 showed a client cannot be trusted
                # to reconstruct after the fact.
                "started_utc": datetime.datetime.fromtimestamp(
                    started, datetime.timezone.utc).isoformat(),
            },
            progress=progress,
            job=job,
            error=outcome.get("error"),
            warnings=warnings,
        )
        # **The address came with the job**, like the bundle's always has. `request` may be None
        # if validation itself failed — in which case there is no URL to have been given, and the
        # skip path says so.
        runrecord.write(document, (request or {}).get(runrecord.REQUEST_FIELD))
    except Exception as exc:  # noqa: BLE001 — a record must never cost a delivered master
        print("[run-record] NOT assembled ({}: {}). The job is unaffected.".format(
            type(exc).__name__, str(exc)[:200]))


def _run(request, job, machine, warnings, attempts, workdir, progress, captured, started,
         trace=None):
    progress.phase("fetch", pct=0, force=True)

    # ── fetch ────────────────────────────────────────────────────────────────────────────────
    download = os.path.join(workdir, "source")
    storage.fetch_source(request["source_url"], download)
    # The vendored CLI dispatches on file extension and treats an unknown one as 'skip, return
    # zero frames' rather than as an error, so the extension comes from the bytes.
    extension = probe.detect_extension(download)
    source_path = probe.named_with_extension(download, extension)
    source = probe.probe_source(source_path)
    # **What makes this a still is the source being an image**, not a frame count — a count is the
    # number decisions.md 0.2 says not to act on. A one-frame MP4 stays a video: it has a rate,
    # it can carry audio, and CF asked for one.
    still = probe.is_still(extension)
    # Alpha is carried only where it can be: a still, whose master is a lossless image. A video
    # with alpha is warned about below rather than silently flattened.
    keep_alpha = bool(source["has_alpha"]) and still

    # **The codec is resolved HERE, not at the writer, and the two defects that forces are worth
    # naming.** `envelope.derive` validates the request at the door, but `"source"` names a codec
    # only the probed file can supply — so resolution has to wait for the probe and must not wait
    # any longer than that.
    #
    # Resolved at the writer instead, a source this worker cannot encode was refused only after
    # the cold start, the download, `build_args` and `open_source` had all been paid for, and it
    # arrived at the attempt loop as a bare `WorkerError` that classified as `outcome: "error"` —
    # the crash bucket the comment there says a deliberate refusal must never land in. Worse,
    # `--plan-only` returned a cheerful plan for a request that could not run.
    #
    # **And a still cannot honour either knob.** Its master is a lossless image written by
    # `StillWriter`, which takes no codec and no preset; accepting them and encoding a PNG anyway
    # is the silent-reinterpretation class `envelope.py` refuses `"Medium"` for, one whole request
    # shape up. A request that says h265 and receives a PNG has been answered, not served.
    # **Only `codec` and `preset` — NOT `crf`.** `crf` has been accepted and ignored on a still
    # since long before this wave, and refusing it now would be a contract change for callers who
    # send one today, made as a side effect of a codec wave. The two fields this wave introduced
    # are the two it may refuse; the third keeps the behaviour it has always had, and that
    # asymmetry is deliberate rather than an oversight.
    # **Every knob that only a video encode can honour.** `keyframes` and `head_keyframes` join
    # `codec` and `preset` here: `StillWriter` takes none of the four, so a still that named one
    # would be answered with a PNG that honours nothing — and the end-of-encode keyframe refusal
    # cannot catch it either, because a `StillWriter` has no `unplaced_keyframes` to ask.
    asked_for_encode = (request.get("codec", envelope.DEFAULT_CODEC) != envelope.DEFAULT_CODEC
                        or request.get("preset", encoder.DEFAULT_PRESET)
                        != encoder.DEFAULT_PRESET
                        or request.get("keyframes", envelope.DEFAULT_KEYFRAMES)
                        != envelope.DEFAULT_KEYFRAMES
                        or bool(request.get("head_keyframes")))
    if still and asked_for_encode:
        raise WorkerError(
            errors.INVALID_FIELD_VALUE,
            "'codec', 'preset', 'keyframes' and 'head_keyframes' are the master ENCODE's "
            "settings and this source is a still, "
            "whose master is a lossless image written with no encoder at all. A request naming "
            "them for an image has been answered rather than served. Omit them, or send a video.",
        )
    # **The OUTCOME cap, here for the same reason `resolve_codec` is: it needs the probe and it
    # must refuse before anything expensive is paid for.** `derive` can count a list; only the
    # clip's duration says how many keyframes an INTERVAL produces.
    envelope.check_keyframe_cap(request,
                                source.get("duration_s") or source.get("video_duration_s"),
                                source.get("fps"))
    resolved_codec = envelope.resolve_codec(
        request.get("codec", envelope.DEFAULT_CODEC), source.get("codec")) if not still else None

    # **`proxy` is refused on a still rather than produced badly.** A proxy is a duration-bounded,
    # long-edge-capped *video* for scrubbing; over one frame it is a worse copy of the master with
    # a name that promises something else. Returning one would be the failure this worker exists
    # to avoid in miniature — an artefact that exists, passes every shape check, and is not what
    # the role means.
    #
    # Here rather than in `validation`, because whether the source is a still is not known until
    # it has been fetched and sniffed. Same code and same never-retryable class as the static
    # refusals: `proxy` on a still is `proxy` on a still on every attempt.
    if still and any(entry["role"] == "proxy" for entry in request["derive"]):
        raise WorkerError(
            FIELD_NOT_SUPPORTED,
            "derive role 'proxy' is not supported for a still source: a proxy is a "
            "duration-bounded video, and this source has no time axis. Use 'poster' or 'crop'.",
        )

    # ── plan ─────────────────────────────────────────────────────────────────────────────────
    # An estimate of the frame count, for planning and the ETA **only**. Every frame count this
    # worker reports comes from the decode.
    estimated_frames = None
    if source["duration_s"] and source["fps"]:
        estimated_frames = int(round(source["duration_s"] * source["fps"]))
    progress = progress_module.Progress(job=job, estimated_frames=estimated_frames,
                                        debug=request.get("debug"))

    # **An exact canvas changes what the model is asked for, not just what is delivered.** The
    # short edge has to cover the request in both axes or the final fit would enlarge one of them
    # — inventing pixels after the model finished, which is what the caller paid to avoid.
    exact_size = request.get("output_size")
    model_short_edge = request["target_short_edge_px"]
    if exact_size:
        # **The backstop CF asked to keep** (2026-08-15). CF rejects a caller who sends both forms
        # before dispatch, so this should never fire; if it does, CF has a bug and the choice is
        # between completing with a warning and dying on a technicality with the fetch already
        # paid for. `output_size` wins because it is the more specific statement — a caller who
        # named an exact canvas has a composite to land on, and a short edge cannot express it.
        if model_short_edge:
            warnings.append(
                "both sizing forms arrived: output_size {}x{} and target_short_edge_px {}. CF "
                "guarantees one per request, so this is a CF-side bug worth reporting. Honouring "
                "output_size, which is the more specific of the two."
                .format(exact_size[0], exact_size[1], model_short_edge))
        model_short_edge = estimator.short_edge_covering(
            source["width"], source["height"], *exact_size)
        # Everything downstream — the plan, the manifest, the pipeline's `--resolution` — reads
        # the request's short edge. With `output_size` alone there was none to read, and a None
        # here reached the plan as a missing target rather than as an error anyone could act on.
        request["target_short_edge_px"] = model_short_edge
        # **A still can be any size; a video master cannot.** `yuv420p` subsamples chroma by two
        # in both axes, so an odd dimension is not encodable — the encoder fails after the GPU is
        # spent, with "Could not open encoder" and nothing written. Refused up front, naming the
        # nearest size that works, because the caller can act on that and cannot act on an ffmpeg
        # error. Stills go through untouched: their master is a lossless image with no such rule.
        if not still and (exact_size[0] % 2 or exact_size[1] % 2):
            raise WorkerError(
                errors.INVALID_FIELD_VALUE,
                "output_size of {}x{} cannot be encoded as video: yuv420p "
                "subsamples chroma by two, so both dimensions must be even. {}x{} is the nearest "
                "size that works. A still source has no such limit.".format(
                    exact_size[0], exact_size[1],
                    exact_size[0] - exact_size[0] % 2, exact_size[1] - exact_size[1] % 2),
            )
        requested_aspect = exact_size[0] / float(exact_size[1])
        source_aspect = source["width"] / float(source["height"])
        warnings.append(
            "output_size pins the output to {}x{}; the model was run at a short edge "
            "of {} so the fit only ever shrinks.".format(exact_size[0], exact_size[1],
                                                         model_short_edge))
        # A caller asking for an aspect the source does not have is asking for a stretch. That is
        # sometimes exactly right — it is how CF's layer pipeline undoes a separator that rounded
        # its width to a 16 grid — and it is never something to do silently.
        if abs(requested_aspect - source_aspect) / max(source_aspect, 1e-9) > 0.001:
            warnings.append(
                "the requested {}x{} is aspect {:.4f} where the source is {:.4f} — the output is "
                "stretched by {:+.2f}% horizontally against a proportional resize. Intended when "
                "correcting a producer that changed the aspect; otherwise omit output_size and "
                "give target_short_edge_px instead.".format(
                    exact_size[0], exact_size[1], requested_aspect, source_aspect,
                    100 * (requested_aspect / source_aspect - 1)))

    job_shape = {
        "target_short_edge_px": model_short_edge,
        "source_width": source["width"],
        "source_height": source["height"],
        "estimated_frames": estimated_frames,
        "allow_oom_retry": request["allow_oom_retry"],
        # **The two facts the planner cannot read off the geometry.** `still` is what lifts the
        # temporal floor — a single image has no time axis for a 21-frame quality floor to apply
        # to — and it is known only after the source has been fetched and sniffed. `tile_quality`
        # is the caller's decode-seam lever.
        "still": still,
        "tile_quality": request.get("tile_quality", "default"),
        "schedule": request.get("schedule", "max_window"),
    }
    # **Before the plan, because planning an impossible frame count is the cost being avoided**
    # (F-2026-08-18-15). `estimated_frames` is duration x fps off container metadata, which an
    # untrusted source controls; the schedule simulation and the chunk arithmetic are both
    # functions of it. Refused by arithmetic against RunPod's own execution ceiling — never
    # against a frame count this worker invented, and never duplicating the caller's deadline,
    # which `refuse_if_the_deadline_cannot_be_met` owns further down with its own margins.
    _planned_w, _planned_h = estimator.output_dimensions(
        source["width"], source["height"], model_short_edge or 1)
    estimator.refuse_frames_no_deadline_admits(
        job_shape, estimator.load_calibration(), _planned_w * _planned_h)

    plan, rationale = estimator.plan(job_shape, machine)

    # **Calibration override.** The estimator picks the fastest rung whose *measured* peak fits,
    # and it only ever measures the rung it ran — so an empty table means the floor, for ever,
    # with no path to learning that anything faster would have worked. Pinning the rung is how
    # the first rows get written. Recorded in the rationale so a manifest never claims a
    # configuration was chosen when it was dictated.
    if request["force_rung"]:
        index = next(i for i, r in enumerate(estimator.RUNGS)
                     if r["name"] == request["force_rung"])
        # **Re-planned, not patched.** This block used to build the plan by hand and copy three
        # keys onto the old rationale, which left every *number* in it describing the rung that
        # was not used. Measured: a job pinned to `swapped` was refused on a deadline predicting
        # 1903.6 s, which is 30 frames at the **floor** rung's 63.5 s/frame; `swapped` predicts
        # 12.8. The report named the right rung beside the wrong rung's arithmetic, and the
        # refusal it drove was wrong by a factor of five.
        #
        # That is the same failure as the `why`/`reason` spelling below it — a rationale half
        # updated — so the fix is to stop updating it by hand. `plan()` builds a coherent one for
        # the forced rung and only the sentence explaining the pin is layered on top.
        estimators_own = rationale
        plan, rationale = estimator.plan(job_shape, machine, force_rung=index)
        plan["batch_size"] = pipeline.snap_batch_size(plan.get("batch_size", 1))
        # **`reason`, not `why`.** `estimator.plan` calls this field `reason`; writing `why` here
        # added a key nobody reads and left `reason` holding the estimator's *unforced* rationale.
        # A pinned run then reported "no calibration data, so the floor is the only honest choice"
        # next to a rung of `resident` — the exact thing the comment above says must not happen,
        # defeated by a spelling. Observed on a 300-frame job whose printed reason contradicted
        # its own configuration.
        rationale = dict(rationale, forced=True,
                         reason="pinned by force_rung; the estimator's own choice was '{}' "
                                "because: {}".format(estimators_own["rung"],
                                                     estimators_own["reason"]))
        warnings.append(
            "force_rung={} — the configuration was pinned for calibration, not chosen from "
            "measurement. Do not read this run's rung as the estimator's judgement.".format(
                plan["name"]))

    # **One knob at a time, applied after the rung so it survives whatever the rung chose.**
    # `batch_size` is frames per model batch — the model's temporal window, and the lever CF
    # identifies as dominant for video quality. It has only ever moved bundled inside a rung
    # (21 at `fast`, 1 at the floor) alongside five other changes, so nothing here has ever
    # measured what it is worth on its own. Same for `temporal_overlap`.
    # Every lever the configuration carries, so "one knob at a time" is true rather than
    # aspirational. Tiling and block-swapping used to be reachable only by changing rung, which
    # also changes the window -- the confound that made the tiling question unanswerable.
    for field, key, snap in (("force_batch_size", "batch_size", pipeline.snap_batch_size),
                             ("force_temporal_overlap", "temporal_overlap", None),
                             ("force_chunk_size", "chunk_size", None),
                             ("force_vae_encode_tiled", "vae_encode_tiled", None),
                             ("force_vae_encode_tile_size", "vae_encode_tile_size", None),
                             ("force_vae_encode_tile_overlap", "vae_encode_tile_overlap", None),
                             ("force_vae_decode_tiled", "vae_decode_tiled", None),
                             ("force_vae_decode_tile_size", "vae_decode_tile_size", None),
                             ("force_vae_decode_tile_overlap", "vae_decode_tile_overlap", None),
                             ("force_blocks_to_swap", "blocks_to_swap", None),
                             ("force_swap_io_components", "swap_io_components", None)):
        requested = request.get(field)
        if requested is None:
            continue
        applied = snap(requested) if snap else requested
        plan[key] = applied
        warnings.append(
            "{}={} — {} was pinned for calibration, not chosen from measurement{}".format(
                field, requested, key,
                "; snapped to {} on the 4n+1 lattice".format(applied) if applied != requested
                else ""))
        rationale = dict(rationale, **{key: applied})

    # **Checked after the forced levers, because that is the path that can build an incoherent
    # plan.** Everything else assembles a configuration whole; forcing sets one field at a time,
    # which is the point of it and also the way to end up with a flag whose companion is missing.
    try:
        plan, preflight_warnings = solver.preflight(plan, frames=estimated_frames)
    except ValueError as exc:
        raise WorkerError(FIELD_NOT_SUPPORTED, str(exc))
    for line in preflight_warnings:
        warnings.append(line)
        rationale = dict(rationale, preflight=preflight_warnings)

    # **A pinned configuration fails rather than ratchets.** The ratchet exists to rescue a job
    # the estimator misjudged; on a calibration run it does the opposite -- it quietly substitutes
    # a different configuration, the job succeeds, and the row banked describes something nobody
    # asked for. That is how 68 of 70 peak measurements became unattributable. `pin` is consulted
    # by `_Ratchet.handles`, so the OOM propagates with its shortfall intact.
    if request.get("pin"):
        warnings.append(
            "pin=true — this configuration will not be ratcheted. An OOM here fails the job and "
            "reports the shortfall measured at exactly what was asked for, which is the point.")
        rationale = dict(rationale, pinned=True)

    # **The window is the smaller of the two, and saying so is the whole point of this block.**
    # `chunk_size` is how many frames reach the model at a time and `batch_size` is how many it
    # attends to at once, so a batch above the chunk is silently the chunk. Nothing said this, and
    # four runs at batch 21, 33, 49 and 65 on a rung that chunks at 9 produced byte-identical
    # masters at an identical 23.15 GB peak — three of them measuring an experiment that had
    # already concluded. A knob that quietly does nothing is worse than one that refuses.
    # **Silent on a still, because a still has no temporal window to shrink.** One frame is one
    # frame whatever the batch says, so this fired on every image job carrying nothing a caller
    # could act on — and a warning that is always present and never useful is how the warnings
    # that matter stop being read. That failure mode is one message old: 4.19 is about diagnostics
    # nobody could see, and burying them in noise is the same outcome by a different route.
    # **And silent when the chunk IS the clip.** "Raise chunk_size to raise the window" is
    # advice nobody can take on a job whose chunk already holds every frame there is — the
    # window is the clip's own length, which is decision 6 working, not a configuration anyone
    # chose badly. The batch exceeding it is the caller asking for more context than the clip
    # contains. Same failure mode as the still case one line up: a warning that is always
    # present and never actionable is how the warnings that matter stop being read.
    chunk_is_the_clip = (estimated_frames is not None
                         and plan["chunk_size"] >= estimated_frames)
    if not still and not chunk_is_the_clip and plan["batch_size"] > plan["chunk_size"]:
        warnings.append(
            "batch_size={} exceeds chunk_size={}, so the model's temporal window is {} frames, "
            "not {}. Only {} frames ever reach it at once. Raise chunk_size to raise the window."
            .format(plan["batch_size"], plan["chunk_size"], plan["chunk_size"],
                    plan["batch_size"], plan["chunk_size"]))
        rationale = dict(rationale, effective_temporal_window=plan["chunk_size"])
    else:
        # **The fact is recorded even where the advice is suppressed.** Silencing a warning must
        # not silence the measurement behind it: the window really is min(batch, chunk), and the
        # rationale is what a later fit reads.
        rationale = dict(rationale, effective_temporal_window=min(plan["batch_size"],
                                                                 plan["chunk_size"]))

    # **The deadline, checked before the GPU is spent.** `execution_timeout_ms` is a hard kill —
    # the container is ended, no master is written, nothing is returned, and every second is
    # billed. Measured against elapsed-since-entry rather than a clock: CF sends a *cap* because
    # `executionTimeout` bounds processing time while queue time lives under `ttl`, so an absolute
    # deadline would be measuring the wrong clock and wrong by the queue wait.
    estimator.refuse_if_the_deadline_cannot_be_met(
        rationale, request.get("execution_timeout_ms"), time.time() - started, estimated_frames)

    # **A deadline that could not be checked has to say so.** Silence here reads exactly like a
    # deadline that was checked and passed — which is how a 60s cap let a 591s job run and bill.
    if request.get("execution_timeout_ms") and not rationale.get("predicted_seconds"):
        warnings.append(
            "execution_timeout_ms was given but this job could not be predicted at rung '{}' — "
            "nothing comparable is calibrated, so the deadline was NOT checked and a job that "
            "cannot finish inside it will be killed with the cost billed.".format(
                rationale["rung"]))

    # Comfortable but close is worth saying before it bites, since the caller chooses the cap and
    # a resubmit costs them the whole job over again. **Read off the same arithmetic the refusal
    # used** rather than a second threshold of its own: a warning that disagreed with the check it
    # warns about would be worse than no warning, and the 0.75 that used to live here was one more
    # number nobody could tie to the refusal that actually fires.
    checked = rationale.get("deadline")
    if checked and checked.get("headroom") and checked["headroom"] < 1.3:
        warnings.append(
            "this job is predicted to take {:.0f}s ({:.0f}s with the {:.1f}x safety factor) of a "
            "remaining {:.0f}s deadline — inside it, but with {:.0f}% margin, and the estimate "
            "covers inference only. A larger execution_timeout_ms would cost nothing unhit."
            .format(checked["predicted_seconds"], checked["required_seconds"],
                    checked["safety_factor"], checked["remaining_seconds"],
                    100 * (checked["headroom"] - 1)))

    # The bundle is written from `handle`, which never sees these. Recorded as they are learned,
    # and re-recorded after the force overrides so a pinned run's bundle describes what ran.
    if trace is not None:
        trace["rationale"] = rationale
        trace["source"] = {
            "width": source.get("width"), "height": source.get("height"),
            "fps": source.get("fps"), "duration_s": source.get("duration_s"),
            "estimated_frames": estimated_frames, "has_audio": source.get("has_audio"),
            "has_alpha": source.get("has_alpha"), "pix_fmt": source.get("pix_fmt"),
            "still": still, "bytes": os.path.getsize(source_path),
        }
        # **The stub, here: the last moment before anything can die without saying so.** Below
        # this line lie the load strip and every model phase, which is where a cgroup SIGKILL
        # happens — no exception, no bundle, no envelope. Filed now, the record survives that
        # death carrying the plan that caused it.
        _write_run_record_stub(request, machine, rationale, trace["source"], job, started)

    out_w_planned, out_h_planned = exact_size or estimator.output_dimensions(
        source["width"], source["height"], model_short_edge)

    # **The planner's own rate, handed to the reporter before any work starts**
    # (F-2026-08-19-29). Without it `eta_s` cannot answer until a frame is written, and on a
    # whole-clip chunk that is the final drain — so the ETA, and the poll cadence that keys off
    # it, were absent for the entire model stretch of a 1,117 s job.
    progress.expect(rationale.get("seconds_per_frame"),
                    basis=rationale.get("prediction_basis"))

    progress.phase("estimate", pct=1, rung=rationale["rung"], force=True)

    # **A 10-bit source is truncated to 8 bits before the model sees it, and nothing said so.**
    # Frames are read through `cv2.VideoCapture`, which always decodes to 8 bits per component
    # (`pipeline.py` says as much). ffprobe knows the source is 10-bit, this worker records the
    # `pix_fmt`, and until now the two facts never met — so graded, log or HDR footage lost four
    # bits of precision with every check passing. The master is 8-bit too, so the loss is real
    # rather than merely internal.
    if not still and probe.bits_per_component(source.get("pix_fmt")) > 8:
        warnings.append(
            "source is {} — more than 8 bits per component. Frames are decoded through OpenCV, "
            "which always yields 8-bit, so the extra precision is discarded before the model "
            "sees it, and the H.264 master is 8-bit as well. Structurally correct, measurably "
            "less than the source carried.".format(source["pix_fmt"]))

    if source["has_alpha"] and not still:
        warnings.append(
            "source carries an alpha channel ({}) but is a video; the master is H.264 and cannot "
            "hold alpha, so the output is opaque. Alpha is retained for still sources only."
            .format(source["pix_fmt"]))

    if request["target_short_edge_px"] < min(source["width"], source["height"]):
        warnings.append(
            "target_short_edge_px {} is below the source's short edge {} — this is a downscale"
            .format(request["target_short_edge_px"], min(source["width"], source["height"])))

    # ── plan-only: the priced plan, and not one GPU-second ───────────────────────────────────
    #
    # **Placed here on purpose — after every override, the preflight and the deadline check.**
    # The contract says plan-only runs "through the same code path production takes", and a
    # short-circuit any earlier would report a plan that the forced levers, the coherence check
    # or the deadline could still have changed. Everything above this line is the decision;
    # everything below it is the doing.
    #
    # The source *is* fetched and probed first, because the plan is a function of the real
    # geometry: a plan computed from a caller's guess at the frame count answers a different
    # question from the one the run would ask.
    if request.get("plan_only"):
        return _decorate({
            "plan_only": True,
            "registry_version": planner.REGISTRY_VERSION,
            "rationale": rationale,
            "configuration": {k: v for k, v in plan.items() if k != "name"},
            "source": {
                "width": source["width"], "height": source["height"],
                "fps": source["fps"], "duration_s": source["duration_s"],
                "estimated_frames": estimated_frames, "still": still,
            },
            # **`planned_output`, not `output`.** `output` is the envelope's own key for the
            # artefacts a run delivered, and a plan-only response carrying one would look, to
            # anything reading the envelope, exactly like a run that had produced something.
            "planned_output": {"width": out_w_planned, "height": out_h_planned},
            "requested": {
                "target_short_edge_px": request["target_short_edge_px"],
                "crf": request.get("crf"), "tile_quality": request.get("tile_quality"),
                "color_correction": request["color_correction"],
            },
        }, machine, attempts, warnings, progress, started, debug=request.get("debug"))

    # ── the load strip, which used to be silent ──────────────────────────────────────────────
    #
    # **Between `estimate` and the first model phase lies minutes of work that reported nothing**
    # (F-2026-08-19-31 collateral). The vendored import pulls torch, diffusers and the attention
    # backends; the model materialization that follows reads the 16.4 GiB checkpoint, which on a
    # host that streams image layers lazily is where the layer is actually faulted in — 6m06s and
    # 12m29s, both measured. Nothing was emitted across any of it, so `/status` held the
    # `estimate` payload the whole way and a watcher could not tell a loading worker from a hung
    # one. **Three paid jobs were cancelled blind into this strip in a single day**, and every one
    # of those cancels made things worse while every wait was vindicated.
    #
    # A `pct` is offered but is not the point — *existence* is. The payload's `at` advances, which
    # is what `run_one`'s stall detector reads liveness from, and the phase name says which of the
    # two long silences the worker is inside.
    # **Said once, before anything else happens.** Every banner after this one belongs to a
    # phase and knows only its own span.
    print(plan_summary(rationale, machine, estimated_frames,
                       (machine or {}).get("host_ram_gb")))
    progress.phase("load", pct=2, force=True)
    _say_host("load")
    load_strip = trace.setdefault("load_strip", {}) if trace is not None else {}
    load_started = time.time()
    print("[load] importing the vendored CLI (torch, diffusers, attention backends)")
    cli = pipeline.load_cli()
    import_s = time.time() - load_started
    load_strip["import_s"] = round(import_s, 1)
    print("[load] vendored import complete in {:.1f}s".format(import_s))

    # **The weights are measured on every run, whether or not anyone is watching.** The strip's
    # two halves have very different causes — an import that is CPU and page-cache bound, and a
    # checkpoint read whose cost is the host's storage — and until they were timed separately the
    # whole strip was one unattributed silence. The second half is measured from here to the
    # first model phase; `phasewatch` stamps that boundary, so the figure lands in the same run
    # whose banners the tail-term fit reads.
    progress.phase("load", pct=3, force=True, note="weights", import_s=round(import_s, 1))
    print("[load] model preparation begins (checkpoint read + runner build); "
          "first model phase ends this strip")

    # ── upscale, with one conservative retry ─────────────────────────────────────────────────
    out_w, out_h = out_w_planned, out_h_planned
    # **The caller may name its own master** (F-2026-08-19-38): CF's front-end holds a request
    # id per job and wants the file called by it. The stem is theirs, the extension is ours, and
    # an absent field lands on `master.mp4` / `master.png` exactly as before — so this line is a
    # no-op for every request written before the field existed.
    # **`output.name` is retired** (storage.md §4, CF 2026-08-28). It let a caller choose the
    # master's stem, for a need `request_id` now serves better and serves for every artefact
    # rather than for one.
    master = keys.master_name(request["request_id"], still, out_w, out_h)
    master_path = os.path.join(workdir, master)
    result = _upscale_with_retry(cli, request, source, source_path, master_path, plan,
                                 rationale, machine, attempts, progress, estimated_frames,
                                 still=still, keep_alpha=keep_alpha, exact_size=exact_size,
                                 warnings=warnings, load_strip=load_strip,
                                 # Resolved once in this function, right after the probe, and
                                 # passed down rather than recomputed — `"source"` needs the
                                 # probed codec and the writer is two frames further in.
                                 codec=resolved_codec,
                                 # The stopwatch the retry loop's stop is measured against — the
                                 # same one the deadline refusal uses, started at handler entry.
                                 started=started)

    # **What was delivered, on the trace, for the run-record** (F-2026-08-19-36). Size and frame
    # counts only — the key and the bucket are the customer's, and a corpus does not need them to
    # price a card.
    if trace is not None:
        trace["output"] = {
            "width": (result.get("actual_size") or (None, None))[0],
            "height": (result.get("actual_size") or (None, None))[1],
            "frames_written": result.get("written_out"),
            "frames_decoded": result.get("decoded_in"),
            "seconds": round(result["seconds"], 1) if result.get("seconds") else None,
        }

    predicted, actual = result.get("predicted_size"), result.get("actual_size")
    if predicted and actual and tuple(predicted) != tuple(actual):
        warnings.append(
            "the model produced {}x{} where {}x{} was computed from target_short_edge_px={}; "
            "the model rounds to its own grid and the output written is the model's size"
            .format(actual[0], actual[1], predicted[0], predicted[1],
                    request["target_short_edge_px"]))

    # ── the master goes to storage before any derive is attempted ────────────────────────────
    progress.phase("upload", pct=90, force=True)
    client = storage.client_for(request["output"])
    artefacts = []
    master_key = storage.upload(client, request["output"], master, master_path,
                                keys.content_type(master))
    artefacts.append(master)

    measured = probe.probe_output(master_path)
    output_entry = dict(measured)
    output_entry.update({
        "key": master_key,
        "bytes": os.path.getsize(master_path),
        "content_type": keys.content_type(master),
        # Only meaningful for a container that has a moov atom to move.
        "faststart": probe.is_faststart(master_path) if not still else None,
        "channels": 4 if keep_alpha else 3,
    })

    # ── derives ──────────────────────────────────────────────────────────────────────────────
    derived = []
    if request["derive"]:
        progress.phase("derive", pct=94, force=True)
        scale = measured["width"] / float(source["width"])
        try:
            entries = derives.build(request["derive"], master_path, source_path, workdir,
                                    frame_count=result["written_out"], scale=scale,
                                    request_id=request["request_id"],
                                    warn=warnings.append)
        except Exception as exc:  # noqa: BLE001 — a derive must never lose a written master
            # **A failed derive leaves the master intact and recoverable**, so it is reported
            # rather than raised: raising here would return a refusal for a job whose master is
            # already in the caller's bucket, and throw away hours of GPU over a poster.
            #
            # `WorkerError` alone was not enough. The roles shell out to ffmpeg and read images
            # through PIL, so the realistic failures are `CalledProcessError` and `OSError` —
            # neither of which this caught, and both of which would have escaped as `internal`.
            warnings.append("derives failed: {}: {}".format(
                type(exc).__name__, getattr(exc, "message", None) or str(exc)[:200]))
            entries = []
        for entry in entries:
            entry["key"] = storage.upload(client, request["output"], entry["name"],
                                          entry["path"], entry["content_type"])
            artefacts.append(entry["name"])
            derived.append({k: v for k, v in entry.items() if k not in ("path", "name")})

    # ── the manifest, last ───────────────────────────────────────────────────────────────────
    body = {
        "output": output_entry,
        "derived": derived,
        "frames": {"decoded_in": result["decoded_in"], "written_out": result["written_out"]},
        # Mean and spread of the written pixels. A standard deviation near zero means a flat
        # image, which is either a broken run or a very plain photograph — worth seeing either way.
        "pixels": result.get("pixels"),
        "estimate": {
            "predicted": rationale,
            "actual": {
                # **No `rung`.** It was an internal bundle name -- six hard-coded configurations
                # this worker happened to choose between -- and it was never part of the contract:
                # CF sends a source and a target short edge. What CF *did* ask for is whether a
                # run tiled and whether it chunked temporally, which is the three fields below.
                # Three lines further down this file already says a field CF can see is a field
                # CF may come to depend on; publishing `rung` ignored our own warning.
                "seconds": round(result["seconds"], 1),
                "seconds_per_frame": progress.seconds_per_frame(),
                "tiled_encode": result["plan"]["vae_encode_tiled"],
                "tiled_decode": result["plan"]["vae_decode_tiled"],
                "chunk_size": result["plan"]["chunk_size"],
                # **No `batch_size` or `window` here, deliberately.** Both are carried on each
                # attempt, which is where `chunk_size` and `blocks_to_swap` already live and what
                # the diagnostics bundle and the harness read. This block is CF's top-level
                # summary, and a field CF can see is a field CF may come to depend on -- at which
                # point it is contract, whether or not that was intended. The window is an
                # internal quality lever and nothing outside this worker chooses it.
                "blocks_swapped": result["plan"]["blocks_to_swap"],
            },
        },
        "timings": {"worker_ms": int((time.time() - started) * 1000)},
        # **What was checked, not only what the checks found.** Every guard in this worker
        # reports a *finding* and reports nothing when it does not run — so a check that silently
        # skipped and a check that passed produced identical output. That shape has cost this
        # project a missing calibration table, a deadline that could not predict, a `warnings`
        # array nothing read, an empty log for two months, and a redaction that covered three
        # channels of four. Fixing each instance leaves the class intact.
        #
        # A named entry that reads `false` is visible. A line that was never printed is not.
        "checks": {
            "deadline": bool(request.get("execution_timeout_ms")
                             and rationale.get("predicted_seconds")),
            "capacity": bool(rationale.get("usable_vram_gb") is not None
                             and rationale.get("calibrated")),
            "source_exhausted": True,          # asserted in `_upscale_once` or the job raised
            "frames_match": result["decoded_in"] == result["written_out"],
            "alpha_carried": bool(keep_alpha),
            "bit_depth_known": probe.bits_per_component(source.get("pix_fmt")) is not None,
            "derives_requested": len(request["derive"]),
            "derives_produced": len(derived),
        },
        "warnings": warnings,
    }
    # **`WORKER_VERSION` and `MODEL_BUILD` are no longer passed** (CF, 2026-08-28): they came out
    # of the manifest's top level, and `build_identity()` already carries both. Passing them here
    # would be handing the function a fact it has a better source for.
    manifest_body = manifest_module.build(request, body, machine, attempts, job=job,
                                          build_identity=build_identity())
    manifest_name = keys.manifest_name(request["request_id"])
    manifest_path = os.path.join(workdir, manifest_name)
    with open(manifest_path, "w") as handle:
        handle.write(manifest_module.serialise(manifest_body))
    manifest_key = storage.upload(client, request["output"], manifest_name, manifest_path,
                                  keys.content_type(manifest_name))
    artefacts.append(manifest_name)

    # A job that retried is worth a diagnostics bundle even though it succeeded: it holds both
    # the estimate that was wrong and the configuration that worked.
    if diagnostics.should_write(False, len(attempts) > 1):
        _write_diagnostics(request, machine, attempts, None, captured, failed=False,
                           trace=trace, warnings=warnings, job=job, started=started)

    response = {
        "status": "ok",
        "output": output_entry,
        "derived": derived,
        "manifest_key": manifest_key,
        "artefacts_written": artefacts,
        "frames": body["frames"],
        "estimate": body["estimate"],
        # Carried into the response as well as the manifest. A caller reading the reply should not
        # have to fetch an object to find out which guards ran — and the manifest is written after
        # this, so a reader who never gets one has the same question and no answer.
        "checks": body["checks"],
        "attempts": attempts,
    }
    if warnings:
        response["warnings"] = warnings
    return _decorate(response, machine, attempts, warnings, progress, started,
                     debug=request.get("debug"))


def _job_shape_for(source, plan, estimated_frames, exact_size=None):
    """What the planner needs to re-plan: real dimensions, not pixel counts.

    **Dimensions rather than areas.** A decode grid is laid out on the actual 16:9 frame, and a
    square of the same pixel count gives a different grid -- 2x1 becomes 2x2 at 4K. Every caller
    of the walk goes through here so that cannot drift between them.
    """
    width = source.get("width") or 0
    height = source.get("height") or 0
    out_w, out_h = exact_size or estimator.output_dimensions(
        width, height, plan["target_short_edge_px"])
    return {
        "target_short_edge_px": plan["target_short_edge_px"],
        "source_width": width, "source_height": height,
        "output_width": out_w, "output_height": out_h,
        "estimated_frames": estimated_frames,
    }


def _upscale_with_retry(cli, request, source, source_path, master_path, plan, rationale,
                        machine, attempts, progress, estimated_frames, still=False,
                        keep_alpha=False, exact_size=None, warnings=None, started=None,
                        load_strip=None, codec=None):
    """Run the model, stepping down the ladder until a rung fits or the floor is reached.

    A job that OOMs has already spent everything: the source is fetched and on local disk, the
    model is loaded, and RunPod has billed every second. Failing there throws away a paid
    position.

    **This walks the ladder rather than taking one step, and the docstring used to claim
    otherwise.** It read "one retry — not a ladder, not a loop" while the code below was `while
    True:` with no attempt counter, so the stated policy was never the implemented one. The code
    is what is correct here, and the measurement that settles it is 3.19: a 12288² still OOMed at
    `resident`, at `balanced`, and at `tiled`, then succeeded at `swapped`. **One retry would have
    refused a job the hardware could do**, which puts cost above success and inverts this
    project's order — quality, then success, then cost.

    The cost of walking is bounded in practice by where OOMs happen: every one observed so far
    fired within seconds, during VAE encode, before a frame was produced. The walk above cost
    284.8 s against 221.1 s for the attempt that worked — 29%, to turn a failure into a delivery.

    **What is genuinely owed here is a stop for the case nobody has seen**: an OOM arriving late
    in a long clip, where each further attempt costs minutes rather than seconds and
    `executionTimeout` is 900 s. Bound that by time already spent, never by attempt count — a
    count refuses cheap recoveries and permits expensive ones, which is the wrong distinction.
    """
    while True:
        attempt_started = time.time()
        # **`batch_size` and the window it implies, because neither has ever left this process.**
        # The attempt carried `chunk_size` alone, so the temporal window -- `min(batch, chunk)`,
        # the dominant quality lever on video -- was not derivable from any response. A reader
        # seeing `chunk_size=49` on the `balanced` rung would take the window for 49 where it is
        # 9, and a reader seeing `chunk_size=10` on `resident` would take it for 10 where the
        # batch of 21 is clamped to exactly that. Both readings are wrong in opposite directions
        # from the same field. `_effective_window` is the only honest answer and it is one line.
        # **No `rung_index`.** It was the index into `RUNGS`, and there is no list to index --
        # the next attempt is derived from the failure, so it cannot be enumerated in advance and
        # a number claiming otherwise is a fingerprint of the table this replaced. `rung` stays as
        # a label: `solved` when the frontier chose it, `replanned` after a walk, or a bundle name
        # when a caller forced one.
        record = {"rung": plan["name"],
                  "batch_size": plan["batch_size"], "chunk_size": plan["chunk_size"],
                  "window": _effective_window(plan),
                  "blocks_to_swap": plan["blocks_to_swap"],
                  "vae_encode_tiled": plan["vae_encode_tiled"],
                  "vae_decode_tiled": plan["vae_decode_tiled"],
                  # **The decode grid's lever, on the attempt that ran it** (F-2026-08-20-40).
                  # `runrecord.CONFIGURATION_KEYS` has listed this since F-36 and has been reading
                  # an absent field ever since, and `run_one` banks the calibration row from the
                  # winning attempt — so the time table could never be keyed on the one lever that
                  # doubles a long job's wall clock. Taken from the plan, which is what ran, not
                  # from the request, which is what was asked for.
                  "tile_quality": plan.get("tile_quality")}
        # Per attempt, not per job: a retry must not inherit the peak of the attempt that OOMed.
        estimator.reset_peak_vram()
        try:
            # **Per attempt, not per job.** A fresh holder each time round the ladder, so a
            # failed rung's encoder peak can never be read onto a later rung's record.
            encoder_out = {}
            outcome = _upscale_once(cli, request, source, source_path, master_path, plan,
                                    progress, estimated_frames, still=still,
                                    keep_alpha=keep_alpha, exact_size=exact_size,
                                    warnings=warnings if warnings is not None else [],
                                    load_strip=load_strip, codec=codec,
                                    encoder_out=encoder_out)
            # **The only place a successful run's cost is recorded.** `diagnose_oom` reads the
            # peak on failure, which tells you a rung does not fit; nothing recorded what a rung
            # actually costs, so the calibration table could never fill and every job ran the
            # floor rung for ever. It rides in `attempts`, so it reaches manifest.json on every
            # success and the diagnostics bundle on every retry.
            record.update({"outcome": "ok", "seconds": round(time.time() - attempt_started, 1),
                           "peak_vram_gb": _attempt_peak_gb(),
                           # **Beside `seconds`, so a rate can be computed without it.** The
                           # attempt's total is what the calibration table has always banked;
                           # subtracting the tail is what makes two runs of the same job
                           # comparable (F-2026-08-20-44).
                           "tail_seconds": outcome.get("tail_seconds"),
                           # Banked per attempt so a codec measurement can be read off the ledger
                           # rather than off a log, and so an OOM ladder's rows each carry the
                           # encoder cost of the rung that produced them.
                           "encoder_peak_rss_gb": outcome.get("encoder_peak_rss_gb"),
                           "output_short_edge_px": request["target_short_edge_px"]})
            # **Four numbers where there was one.** The peak counter is read and reset at every
            # phase boundary, so a single run now says what encode, the sampler, decode and
            # post-processing each cost -- and which of them set the ceiling. A memory sweep that
            # needed one run per point needs one run per four.
            _record_phases(record)
            # **A run that ratcheted is not a clean measurement of the rung it started on**, and
            # the calibration table must not learn it as one. Recorded on the attempt so the
            # manifest, the ledger and the diagnostics bundle all carry it, and so a per-frame
            # figure computed from this run can be recognised as covering two configurations.
            ratcheted = list(getattr(pipeline.run, "last_ratchet", []) or [])
            if ratcheted:
                record["ratchet"] = ratcheted
                record["window_changed_mid_clip"] = True
                # **`window` must be the one that ran, not the one that was planned.** A forced
                # window of 49 that OOMed twice and finished at 9 reported `window=49` beside the
                # warnings describing the two steps down -- the number contradicting the prose in
                # the same block, which is worse than not reporting it at all and is the exact
                # failure this field was added to end. Measured on 2026-08-15 on a 94.97 GB card.
                #
                # The planned figure is kept rather than overwritten: what the estimator intended
                # is evidence about the estimator, and a run that started at 49 and ended at 9 is
                # not the same as one that was planned at 9.
                stepped = [entry for entry in ratcheted if entry.get("window") is not None]
                if stepped:
                    record["planned_window"] = record.get("window")
                    record["window"] = stepped[-1]["window"]
                    record["planned_batch_size"] = record.get("batch_size")
                    record["planned_chunk_size"] = record.get("chunk_size")
                    for field in ("batch_size", "chunk_size"):
                        if stepped[-1].get(field) is not None:
                            record[field] = stepped[-1][field]
            attempts.append(record)
            outcome["rung"] = plan["name"]
            outcome["plan"] = plan
            outcome["seconds"] = time.time() - attempt_started
            return outcome
        except Exception as exc:  # noqa: BLE001 — classified immediately below
            record["seconds"] = round(time.time() - attempt_started, 1)
            # **Banked here, above the OOM test, so EVERY failure carries it.** `__exit__` drains
            # and finishes sampling before the exception leaves the writer, so the number is final
            # by this line whatever killed the attempt. These are the runs the instrument was
            # restored for — its own docstring cites an 8K run that "died at ~46 GiB inside x264's
            # working set" with the ceiling "inferred from a kernel kill". That run is a FAILURE,
            # and banking the peak only where the attempt succeeded would have left it exactly as
            # unreadable as it was before.
            record["encoder_peak_rss_gb"] = getattr(
                (encoder_out or {}).get("writer"), "encoder_peak_rss_gb", None)
            if not estimator.is_oom(exc):
                # **A deliberate refusal is not an error, and the corpus has to be able to tell.**
                # The host guard stops a doomed run on purpose; recording that as `error` would
                # put it in the same bucket as a crash, and the whole value of refusing while
                # alive is that the record explains itself (F-2026-08-20-46).
                record["outcome"] = (
                    "refused" if getattr(exc, "code", None) in errors.DELIBERATE_REFUSALS
                    else "error")
                # **Phases are recorded on *any* failure, not only on an OOM.** These two lines
                # were on the ok path and the OOM path and not here, so a run that died for a
                # non-memory reason threw away every per-phase peak it had already taken -- the
                # tap held them, the log printed them, and the JSON got `seconds` and `outcome`.
                # The class of the exception has nothing to do with whether the measurements
                # before it are worth keeping; a crash in post-processing still measured encode,
                # sampling and decode, and those are the numbers a campaign is made of.
                record["peak_vram_gb"] = _attempt_peak_gb()
                _record_phases(record)
                attempts.append(record)
                raise
            record["outcome"] = "oom"
            # The peak of the attempt that failed. Not a calibration figure — a rung that OOMed
            # has no honest cost — but it is what says how far over the edge it went.
            record["peak_vram_gb"] = _attempt_peak_gb()
            _record_phases(record)
            shortfall = estimator.diagnose_oom(exc, machine)
            # **The phase is the most actionable fact a failure carries, and it used to be
            # discarded.** The vendored code logs `Error in Phase 3 (Decoding)` and then re-raises
            # bare, so by the time the exception arrives here it is an anonymous OOM and the
            # ratchet has no choice but to change everything. `blame()` puts it back, with the
            # confidence attached: `named` when the vendored handler told us, `last_entered` when
            # it is inferred from the last banner.
            blame = _phase_blame()
            if blame:
                shortfall = dict(shortfall or {})
                shortfall.update(blame)
            if shortfall:
                record["shortfall"] = shortfall
            attempts.append(record)

            # **Derived from this failure, not looked up in a table.** This block used to read
            # `estimator.more_conservative(rung_index)` and replace the whole plan with
            # `RUNGS[nxt]` -- five levers moved at once, chosen by a list written before any of
            # this was measured, and answering none of the questions the failure had just
            # answered. It is what produced the run that OOMed at window 109 and then spent
            # 2,359 s at window 5: below the quality floor, with a chunk of 25 against a batch of
            # 5, from a shortfall block that already carried the phase, the failed allocation and
            # `needed_at_least_gb`.
            #
            # The replacement is the same walk the mid-stream ratchet uses -- one bound, measured
            # from the message, and the best-quality row under it. `MIN_WINDOW` is a hard
            # partition inside `frontier()`, so there is no row below the floor to step onto and
            # exhaustion is a terminal state rather than a quieter configuration.
            ratcheted = list(getattr(pipeline.run, "last_ratchet", []) or [])
            if ratcheted:
                record["ratchet"] = ratcheted
                # **A clip that already recovered in place is not restarted.** Everything this
                # loop could try has been tried, from a position that had frames written, which
                # this one does not.
                nxt_row = None
                walk = {"reason": "the stream already recovered in place; a restart would re-run "
                                  "the clip at a configuration it has already reached"}
            else:
                nxt_row, walk = solver.next_after_oom(
                    _job_shape_for(source, plan, estimated_frames, exact_size),
                    machine, plan, (rationale or {}).get("predicted_peak_vram_gb"),
                    phase=(shortfall or {}).get("phase"), shortfall=shortfall,
                    relax_swap=not request.get("pin"))
            record["walk"] = walk
            refusal = _refuse_retry(request, plan, nxt_row, shortfall, machine, source_path,
                                    exc, estimated_frames=estimated_frames)
            if refusal is not None:
                raise refusal

            # **The size is carried over rather than re-read from the request**: with an exact
            # canvas the model is asked for a short edge that covers the caller's width and
            # height, which is not `target_short_edge_px`. Re-reading it here would render the
            # retry at a different size from the attempt it replaced -- invisible on a request
            # whose two numbers happen to agree, which is the case this was written against.
            # **The stop this loop has always owed, and it is a clock rather than a counter**
            # (F-2026-08-18-17). Every OOM observed so far has fired within seconds, during VAE
            # encode, before a frame was produced — which is why walking has been cheap and why
            # bounding it by attempt count would be the wrong distinction: a count refuses cheap
            # recoveries and permits expensive ones. What has never been seen, and what the
            # correction's own repairs make survivable rather than impossible, is an OOM arriving
            # late in a long clip where each further attempt costs minutes.
            #
            # So: if the attempt that just failed would not fit again in the time this job has
            # left, there is no attempt to make. The budget is the caller's deadline where they
            # sent one and RunPod's execution ceiling where they did not — the same arithmetic
            # the frames refusal uses, and never a constant of this worker's own.
            spent = time.time() - (started or attempt_started)
            budget_s = (request.get("execution_timeout_ms")
                        or estimator.PLATFORM_EXECUTION_CEILING_MS) / 1000.0
            last_attempt_s = record.get("seconds") or 0.0
            if spent + last_attempt_s > budget_s:
                raise WorkerError(
                    errors.DEADLINE_EXCEEDED,
                    "out of memory after {:.0f}s, and the attempt that failed took {:.0f}s — "
                    "another would run past the {:.0f}s this job has. The next configuration "
                    "was computed and is reported below; resend with a larger "
                    "execution_timeout_ms to let it run.".format(
                        spent, last_attempt_s, budget_s),
                    shortfall=dict(shortfall or {},
                                   seconds_spent=round(spent, 1),
                                   seconds_per_attempt=round(last_attempt_s, 1),
                                   budget_seconds=round(budget_s, 1),
                                   next_window=nxt_row["window"],
                                   next_decode_grid=nxt_row["decode_grid"]),
                )

            rendered_at = plan["target_short_edge_px"]
            plan = dict(nxt_row["config"])
            plan["name"] = "replanned"
            plan["target_short_edge_px"] = rendered_at
            rationale = dict(rationale or {},
                             predicted_peak_vram_gb=nxt_row["predicted_peak_vram_gb"])
            progress.phase("retry", rung=plan["name"], force=True,
                           note="out of memory; re-planned under {:.1f} GB to window {} at decode "
                                "grid {}".format(walk["bound_gb"], nxt_row["window"],
                                                 nxt_row["decode_grid"]))


def _refuse_retry(request, plan, next_row, shortfall, machine, source_path, exc,
                  estimated_frames=None):
    """Whether to give up, and **why, in a form CF can act on**.

    "Did not retry" is not a result. The question CF is actually asking is whether sending this
    job anywhere again would help, and only the worker can answer it — it knows what it tried, on
    what card, and how far short it fell.
    """
    # **`pin` stops the job-level retry too, and it did not.** The switch was wired into
    # `_Ratchet.handles`, which governs the mid-stream chunk ratchet, and nothing checked it here
    # -- so a pinned run OOMed at exactly the configuration it was sent to measure and then
    # restarted the whole clip on a `RUNGS` bundle. Measured: a window-109 probe failed at
    # 92.21 GB in 228.5 s, then spent 2,359 s at window 5 and delivered a master the checks call
    # worse than a plain resize. Forty minutes of GPU, a bad deliverable, and the measurement the
    # run existed for buried under a second attempt nobody asked for.
    #
    # This is the exact failure `pin` was introduced to prevent, one level up from where it was
    # implemented. The two switches stay distinct: `allow_oom_retry=false` says the caller cannot
    # afford a second attempt; `pin=true` says a second attempt at a *different* configuration
    # would destroy the measurement.
    if request.get("pin"):
        return WorkerError(
            errors.OUT_OF_MEMORY,
            "out of memory at the pinned configuration; pin=true, so nothing else was attempted "
            "and the shortfall below is measured at exactly what was asked for",
            remedy=errors.Remedy.LARGER_GPU if shortfall else errors.Remedy.RETRY_SAME,
            shortfall=shortfall,
        )

    # CF's switch, and it is a business decision about a particular request rather than a
    # technical one here.
    if not request["allow_oom_retry"]:
        return WorkerError(
            errors.OUT_OF_MEMORY,
            "out of memory at a window of {}; allow_oom_retry is false so nothing was "
            "re-attempted".format(_effective_window(plan)),
            remedy=errors.Remedy.LARGER_GPU if shortfall else errors.Remedy.RETRY_SAME,
            shortfall=shortfall,
        )

    if next_row is None:
        # **Nothing fits above the quality floor.** Not "the bottom of a ladder" -- the frontier
        # stops at `MIN_WINDOW` by construction, so exhaustion means every configuration worth
        # delivering needs more memory than this card has. That is the only honest basis for
        # "a larger card would work", and the shortfall is measured at the configuration that
        # actually failed rather than at whichever rung happened to be last in a list.
        #
        # Running below the floor is available and is the caller's decision, not this function's:
        # under this shot's floor the model stops beating a plain resize, so shipping one quietly
        # would be a worse picture sold as a rescue. **The shot's floor, not the constant** — a
        # clip shorter than 21 frames is floored at its own length (CF, 2026-08-18), and quoting
        # 21 at a ten-frame clip names a window it could never have had.
        return WorkerError(
            errors.CAPACITY_EXCEEDED,
            "out of memory, and no configuration at or above the {}-frame quality floor fits in "
            "the memory this card has. A larger card is the remedy; a narrower window would be a "
            "worse picture than not running the model at all.".format(
                # **`or 1` produced a floor of 1 on a multi-frame clip**, on a video whose
                # container carries no duration or fps -- telling a caller about a one-frame
                # floor immediately after CF ruled that a window of 1 must never be offered.
                # Unchanged in behaviour from the old formula, and now actively contradicted by
                # the constant the same wave introduced. An unknown frame count is a video until
                # something says otherwise, so it takes the ladder's bottom rather than the
                # still's floor.
                planner.window_floor(int(estimated_frames)
                                     if estimated_frames else planner.LADDER_BOTTOM)),
            remedy=errors.Remedy.LARGER_GPU,
            shortfall=shortfall,
        )

    # **Check the preconditions; do not assume them.** If the source is gone the retry is not a
    # cheap continuation but a cold start wearing its clothes, and it should not happen.
    if not os.path.isfile(source_path):
        return WorkerError(
            errors.OUT_OF_MEMORY,
            "out of memory, and the source is no longer on local disk — a retry would be a cold "
            "start rather than a cheap continuation",
            remedy=errors.Remedy.RETRY_SAME,
            shortfall=shortfall,
        )
    return None


#: How many times a mid-clip recovery may narrow the window before giving up. Small on purpose:
#: each step is a real quality loss, and the levers above it are cheaper. Not reset by progress --
#: see `_Ratchet.__init__`.
WINDOW_STEP_BUDGET = 3


def _first_phase_closes_the_strip(on_batch, into=None):
    """Wrap the heartbeat so the first model phase stamps the end of the load strip.

    **The strip's second half can only be measured from the far side of it.** Its cost is the
    checkpoint read and the runner build, neither of which announces itself; what *is* observable
    is the moment the first vendored phase reports, because everything before that is preparation.
    Timed here rather than inside `pipeline`, which would have to learn about wall-clock reporting
    to say it, and rather than in `phasewatch`, whose subject is memory.

    Fires once. A ratchet re-enters the model and must not re-print a strip that closed long ago,
    and the figure worth having is the cold one — the first job on a fresh worker is the whole
    finding (F-2026-08-19-31).
    """
    state = {"opened": time.time(), "closed": False}

    def wrapped(phase, index, total):
        if not state["closed"]:
            state["closed"] = True
            prepare_s = time.time() - state["opened"]
            if into is not None:
                # Kept as well as printed: this half is the checkpoint read, which is the number
                # F-31's amended chain is waiting on and which no log survives to answer.
                into["prepare_s"] = round(prepare_s, 1)
            print("[load] strip closed: first model phase ({}) reached after {:.1f}s of "
                  "preparation".format(phase, prepare_s))
            _say_host("load-end")
        return on_batch(phase, index, total)

    return wrapped


def _output_pixels_for(source, plan):
    """The output frame's area, for sizing a tile against."""
    width, height = estimator.output_dimensions(
        source.get("width") or 1, source.get("height") or 1,
        plan.get("target_short_edge_px") or 1)
    return width * height


def plan_summary(rationale, machine=None, estimated_frames=None, host_ram_gb=None):
    """**The job's shape in one line, before any phase begins** (CF, 2026-08-21).

    Ruled during the seam retest, where the vendored banner's "Input: 231 frames" read as a
    single-chunk job to the person watching — it names the chunk it was handed and cannot know
    there is another. A reader should never have to infer the layout from a banner with no view
    of it.

    Built from the rationale rather than re-derived, so the line cannot disagree with the plan
    it describes. Absent fields degrade to a shorter sentence rather than to "None": a still has
    no chunks, a blind card has no name, and a worker with no cgroup has no slice.
    """
    frames = estimated_frames or rationale.get("estimated_frames")
    chunk = rationale.get("chunk_size")
    chunks = rationale.get("chunks")
    tail = rationale.get("tail_chunk")
    parts = []
    if frames:
        plural = "frame" if frames == 1 else "frames"
        if chunks and chunk:
            # **The sizes, not just the count**: "2 chunks" leaves the tail invisible, and the
            # tail is the span that runs at a different window. Written so the arithmetic checks:
            # `5x609+435` sums to the frame count on its left, and `231+9` does too.
            full = chunks - 1 if tail else chunks
            body = "{}x{}".format(full, chunk) if full > 1 else str(chunk)
            sizes = "{}+{}".format(body, tail) if tail else body
            parts.append("{} {} -> {} chunks ({})".format(frames, plural, chunks, sizes))
        else:
            parts.append("{} {} -> 1 chunk".format(frames, plural))
    if rationale.get("window"):
        parts.append("window {}".format(rationale["window"]))
    if rationale.get("residency"):
        parts.append("rung {}".format(rationale["residency"]))
    card = (machine or {}).get("gpu_name")
    if card:
        parts.append("card {}".format(card))
    if chunk and host_ram_gb:
        parts.append("host cap {} of {:.1f} GiB".format(chunk, host_ram_gb))
    return "[plan] " + " · ".join(parts) if parts else "[plan] (nothing settled)"


def _say_host(label, writer=None):
    """A `[host]` reading, carrying the peak as well as the moment.

    **`VmHWM` is the number the tail is for** (F-2026-08-18-23). This function was called from
    outside the writer's `with` while its own comment claimed inside, so it sampled `VmRSS`
    after ffmpeg had drained and freed — systematically understating the very peak the
    registry's provisional tail term is waiting to be fitted from. Moving the call is only half
    the repair: a point sample cannot see the drain peak from either side of it, because the
    peak happens between the two places anything is looking. `VmHWM` is monotone for the life of
    the process, so it reports the drain's high-water mark whenever it is read.
    """
    rss, hwm = phasewatch.host_rss_gb(), phasewatch.host_hwm_gb()
    if rss is None:
        return
    total = phasewatch.host_total_gb()
    # **Kept as well as printed** (F-2026-08-19-36). These banners are the only measurements of
    # the host the tail term is fitted from, and until now they existed solely as log lines —
    # readable by whoever was watching that worker's stream, and by nobody afterwards.
    # **Through `phasewatch.observe`, the one door** (CF, 2026-08-20). These four hand-banked
    # readings used to be the whole corpus while the per-phase banners were printed and dropped;
    # both now land in the same list, in order, with the killer's number and its anon/file split
    # beside ours on every sample.
    phasewatch.observe(label, rss, peak=hwm, total=total,
                       frames_fed=getattr(writer, "frames_written", None))
    print("[host] {:<12} rss {:6.2f} GiB   peak {:>8}{}{}".format(
        label, rss,
        "unknown" if hwm is None else "{:6.2f} GiB".format(hwm),
        "" if not total else "   of {:.1f} ({:.0%} peak)".format(
            total, (hwm or rss) / total),
        "" if writer is None else "   after {} frame(s) fed".format(
            getattr(writer, "frames_written", "?"))))


def _phase_watch():
    """The watch the last `pipeline.run` installed, or None.

    Read through `getattr` for the same reason `last_ratchet` is: `run` publishes its results as
    attributes on the function, and a still, a refusal or a job that failed before the model was
    reached leaves them unset. Absent is a normal state here, not an error.
    """
    return getattr(pipeline.run, "last_phases", None)


def _attempt_peak_gb():
    """The attempt's peak VRAM — **from the tap, because the tap now owns that counter.**

    `estimator.observed_peak_vram_gb()` reads `torch.cuda.max_memory_allocated`, and
    `PhaseWatch` resets exactly that counter at every phase boundary in order to produce per-phase
    figures. So after the tap landed, the "attempt peak" was really the peak of whichever phase
    happened to run last — post-processing, the cheapest of the four.

    Measured, and it is not a small error: a window-65 run at 4K reported **52.28 GB** where the
    true peak was the largest of its four phases. The instrument changed what it measured and
    nothing said so; the docstring in `phasewatch` even states that `max(peaks)` is the recovery,
    and this is that recovery, which was written down and then not wired.

    Falls back to the raw counter when there is no tap — a still, or a run that never reached the
    model — where nothing has been reset and the counter means what it always did.
    """
    watch = _phase_watch()
    if watch is not None and watch.peak_gb is not None:
        return round(watch.peak_gb, 2)
    return estimator.observed_peak_vram_gb()


def _record_phases(record):
    """Per-phase peak VRAM and which phase set the ceiling, on the attempt that produced them.

    **Recorded, never judged.** These are measurements; the decision they inform belongs to the
    estimator and to whoever reads a shortfall. Written onto the attempt so they reach manifest.json
    on success and the diagnostics bundle on a retry, which is the same route `peak_vram_gb` takes.
    """
    # **Above the early returns, because it is a host fact and not a phase fact.** It was below
    # them first, so a run with no vendored tap — a still, a rung-1 case, anything that never
    # reached the model — recorded no core count, which is exactly the run whose tail time would
    # later need one. Instrument-first, per the amendment: recorded, never fitted.
    record["cpu"] = phasewatch.cpu_configuration()
    watch = _phase_watch()
    if watch is None:
        record["phase_watch"] = {"installed": False, "why": "no watch on pipeline.run"}
        return
    if not watch.peaks:
        # **Say why, rather than say nothing.** A run that produces no per-phase figures looks
        # identical to a run on an image that has no tap, and the first GPU run to carry this
        # module was exactly that ambiguity.
        record["phase_watch"] = dict(watch.diagnosis)
        return
    record["phase_peaks_gb"] = {name: round(gb, 2) for name, gb in watch.peaks.items()}
    record["ceiling_phase"] = watch.ceiling
    record["phase_watch"] = dict(watch.diagnosis)
    # **The times the contract has promised since F-36 and never delivered** (F-2026-08-20-44).
    # Every boundary was stamped and the duration dropped, and the cost of that is now measured:
    # the B200 pair ran identical warm jobs whose phases matched to a couple of seconds and whose
    # tails differed 477 s against ~1340 s. The entire 17.9-to-21.7 s/frame spread was tail, and
    # with no per-phase times it had nowhere to be attributed and was read as host variance.
    if watch.durations:
        record["phase_seconds"] = dict(watch.durations)
    # **The volatile GPU series, beside the peaks and the seconds it has to be read against**
    # (item 10, CF 2026-08-28). Attached unconditionally once the watch produced peaks, because
    # its own "sampled nothing, and here is why" is a reading: a card whose clocks could not be
    # read for a whole run is a fact about that run, and an absent key would be indistinguishable
    # from a build that never sampled.
    record["gpu_series"] = watch.gpu_series()



def _phase_blame():
    """Which phase raised, and the levers that phase implicates. See `phasewatch.PhaseWatch`."""
    watch = _phase_watch()
    return watch.blame() if watch is not None else None


def _effective_window(plan):
    """How many frames the model actually attends to at once.

    `min(batch_size, chunk_size)`, because a batch larger than the chunk is silently the chunk —
    measured on `swapped`, where batch 21, 33, 49 and 65 all produced byte-identical masters at an
    identical peak. Anything reporting or comparing "the window" has to go through here, or it is
    reporting a number the model never used.
    """
    return min(int(plan["batch_size"]), int(plan["chunk_size"]))


class _Ratchet:
    """The OOM policy the stream consults, and the reason the window can be planned optimistically.

    **Policy lives here; mechanism lives in `pipeline._stream`.** The stream knows how to resume —
    what was written, what to re-read, how to keep the alpha aligned. It does not know what a rung
    is, how many steps are left, or what the calibration table says. Splitting it that way is what
    lets the resume be tested against a fake model with no rungs in sight.

    **One free retry at the same window before stepping down at all.** A mid-run OOM is often
    allocator fragmentation rather than a configuration that does not fit — 43.71 GB has succeeded
    where 43.01 GB failed on the same reported card. Emptying the cache and trying the same chunk
    again costs about one chunk; under the old all-or-nothing retry it would have cost the entire
    job, which is why it was never worth attempting. Once, not repeatedly: a second identical
    failure is a configuration that genuinely does not fit.
    """

    def __init__(self, estimated_frames, request, source_path, progress, warnings,
                 rebuild_args, reopen, source_pixels=None, output_pixels=None,
                 job_shape=None, snapshot=None, predicted_peak_gb=None):
        self._estimated_frames = estimated_frames
        self._request = request
        self._source_path = source_path
        self._progress = progress
        self._warnings = warnings
        self._rebuild_args = rebuild_args
        self._reopen = reopen
        # Tiling is sized against the frame it tiles: decode against the output, encode against
        # the input. Passing both is what lets relief pick a tile that is not larger than the
        # picture it is cutting up.
        self._source_pixels = source_pixels or output_pixels or 0
        self._output_pixels = output_pixels or source_pixels or 0
        # **What §5.3's walk needs, which the old ratchet had no way to ask for.** Stepping to
        # the next attempt is a *planning* decision -- it re-ranks the frontier under a bound
        # proven by the failure -- and planning needs the job's real dimensions, the card, and the
        # measurements. The ladder needed none of that, which is exactly why it stepped blind.
        self._job_shape = job_shape
        self._snapshot = snapshot or {}
        self._predicted_peak_gb = predicted_peak_gb
        self._retried_in_place = False
        #: Window steps are budgeted and **not** reset by progress. A free in-place retry is
        #: reset by a chunk landing, because sporadic fragmentation on a long clip should not
        #: accumulate into a window cut; a window step is monotonic and must stay bounded, or a
        #: clip that stumbles every tenth chunk walks itself to the floor.
        self._window_steps = 0
        self.steps = []

    def handles(self, exception):
        # CF's switch is respected here rather than in the stream: a caller who said "do not
        # retry" meant the job, and a ratchet is a retry however little it costs.
        #
        # `pin` is the calibration switch and is checked in the same place for the same reason.
        # The two are not the same request: `allow_oom_retry=false` says the caller cannot afford
        # a second attempt; `pin=true` says a second attempt at a *different* configuration would
        # destroy the measurement this run exists to take.
        if self._request.get("pin"):
            return False
        return bool(self._request["allow_oom_retry"]) and estimator.is_oom(exception)

    def step(self, plan, frames_written, exception):
        estimator.release_gpu_memory()
        shortfall = estimator.diagnose_oom(exception, hardware.read())

        # **One free retry, unless the message says it is not free.** The retry exists for
        # allocator fragmentation — 43.71 GB has succeeded where 43.01 GB failed on the same card
        # — and PyTorch's own text says whether that is what happened, by reporting how much was
        # reserved but unallocated. A gap that is noise against the failed allocation means the
        # configuration genuinely does not fit, and with the chunk set to the whole clip a wasted
        # attempt costs every minute the job has already spent rather than one chunk.
        if not self._retried_in_place and estimator.retry_is_worth_it(shortfall):
            self._retried_in_place = True
            record = {"kind": "same_window", "at_frame": frames_written,
                      "rung": plan["name"], "window": _effective_window(plan),
                      "batch_size": plan["batch_size"], "shortfall": shortfall}
            self.steps.append(record)
            self._announce(
                "out of memory at frame {} — retrying the same chunk at the same window after "
                "releasing cached memory. A mid-run OOM is often fragmentation rather than a "
                "configuration that does not fit, and the {} frames already written are kept."
                .format(frames_written, frames_written))
            return {"plan": plan, "args": self._rebuild_args(plan),
                    "capture": self._reopen(), "lead_in": plan["temporal_overlap"],
                    "record": record}

        # **The relief step is gone: the correction *is* the one lever** (F-2026-08-18-18).
        #
        # This asked `solver.relieve`, which was the pre-registry lever ladder and never got
        # re-founded when the heart did. What it reached for was measured, and it was bad: a
        # `TILE_LADDER` bottoming out at 384 px — 68.8% of every frame inside a cross-fade, far
        # past `MAX_BLEND_DECODE` — with no registry price anywhere in the decision, and encode
        # tiles sized against the *source* plane where the fitted formula prices them on R. So a
        # decode OOM at frame 400 finished the clip in a configuration that `planner.plan` and
        # `planner.correct` both refuse as terminal: the release's own quality floor, breached by
        # the single path the re-founding left behind.
        #
        # Deleting the step rather than repairing it, because the step below already does the
        # ruled thing — one lever, chosen by the failing phase, priced from the registry, floored
        # and blend-capped — and two mechanisms for "move one lever" is one that disagrees. The
        # whole-chunk restart from phase 1 is the ruled execution model either way.
        # **Which phase failed, which is what chooses the lever.** Read from the phase watch —
        # the vendored module names the phase it died in, and that name is the whole reason the
        # correction can move one lever instead of five.
        blame = _phase_blame() or {}

        # **Re-plan under a bound the failure proved, rather than step a ladder.**
        # The old code called `estimator.more_conservative` here and loaded a `RUNGS` bundle --
        # five knobs at once, chosen by a list written before any of this was measured. What
        # replaces it is `solver.next_after_oom`, which ranks the whole frontier under
        # `usable / f` where `f` comes out of the OOM message, and which lets the failing phase
        # override the ranking: a decode failure exits sideways to the same window at the next
        # grid instead of cutting the window it did not implicate.
        if self._window_steps >= WINDOW_STEP_BUDGET:
            return None
        if self._job_shape is None:
            # No dimensions to re-plan against — a caller that built this ratchet for the stream
            # alone. Step the window down the lattice on the configuration that failed, which is
            # weaker than a re-plan and is still the right direction.
            return self._step_window_only(plan, frames_written, shortfall)
        nxt_row, why = solver.next_after_oom(
            self._job_shape, self._snapshot, plan,
            self._predicted_peak_gb or self._peak_of(plan),
            phase=blame.get("phase"), shortfall=shortfall,
            relax_swap=not self._request.get("pin"))
        if nxt_row is None:
            return self._step_window_only(plan, frames_written, shortfall)
        self._window_steps += 1
        nxt_plan = dict(nxt_row["config"])
        nxt_plan["name"] = "replanned"
        nxt_plan["target_short_edge_px"] = plan["target_short_edge_px"]
        self._predicted_peak_gb = nxt_row["predicted_peak_vram_gb"]
        record = {"kind": "replan", "at_frame": frames_written,
                  "phase": blame.get("phase"), "confidence": blame.get("confidence"),
                  "from_window": _effective_window(plan), "window": _effective_window(nxt_plan),
                  "from_batch_size": plan["batch_size"], "batch_size": nxt_plan["batch_size"],
                  "from_chunk_size": plan["chunk_size"], "chunk_size": nxt_plan["chunk_size"],
                  "decode_grid": nxt_row["decode_grid"], "decode_tile": nxt_row["decode_tile"],
                  "predicted_peak_vram_gb": nxt_row["predicted_peak_vram_gb"],
                  "bound_gb": why["bound_gb"], "bound_basis": why["basis"],
                  "exited_sideways": why["exited_sideways"], "shortfall": shortfall}
        self.steps.append(record)
        self._announce(
            "out of memory at frame {} in {} — replanned under a bound of {:.1f} GB ({}). "
            "Window {} -> {}, decode grid {}, predicted {:.1f} GB. Where the window moved, "
            "temporal consistency changes mid-clip.".format(
                frames_written, blame.get("phase") or "an unattributed phase",
                why["bound_gb"], why["basis"], _effective_window(plan),
                _effective_window(nxt_plan), nxt_row["decode_grid"],
                nxt_row["predicted_peak_vram_gb"]))
        return {"plan": nxt_plan, "args": self._rebuild_args(nxt_plan),
                "capture": self._reopen(), "lead_in": nxt_plan.get("temporal_overlap", 0),
                "record": record}

        # window than the frames after it, and that is a real difference in the output the caller
        # receives. It beats losing the job, and it is not something to discover from a file.
        self._announce(
            "out of memory again at frame {} — stepping the temporal window down from {} to {} "
            "(rung '{}' to '{}') and continuing. The first {} frames were produced at the larger "
            "window and the rest will not be, so temporal consistency changes once, at that "
            "frame. The alternative was re-running the whole clip at the smaller window."
            .format(frames_written, was, now, plan["name"], nxt_plan["name"], frames_written))
        return {"plan": nxt_plan, "args": self._rebuild_args(nxt_plan),
                "capture": self._reopen(), "lead_in": nxt_plan["temporal_overlap"],
                "record": record}

    def _peak_of(self, plan):
        """What the registry says the failed configuration would need — its binding phase.

        Recomputed rather than remembered, because the number that matters is the one the *plan*
        implies — a plan reached by relief has been changed since it was priced, and using the
        original prediction would bound the walk against a configuration that is no longer
        running. Priced through `solver.plan_of_config`, so this figure and the walk's own are
        in one currency: §5.3 rule 4, two tables never meet in one inequality.
        """
        if self._job_shape is None:
            return None
        out = (self._job_shape.get("output_width") or 0) * (self._job_shape.get("output_height") or 0)
        if not out:
            return None
        priced = solver.plan_of_config(plan, self._job_shape)
        return round(max(priced["prices"].values()), 2)

    def _step_window_only(self, plan, frames_written, shortfall):
        """The degenerate walk: one rung down the lattice, everything else held."""
        window = _effective_window(plan)
        smaller = [w for w in solver.lattice(window) if w < window]
        if not smaller:
            return None
        self._window_steps += 1
        nxt_plan = dict(plan)
        nxt_plan["batch_size"] = smaller[0]
        nxt_plan["chunk_size"] = max(smaller[0], plan["chunk_size"])
        record = {"kind": "step_down", "at_frame": frames_written,
                  "from_window": window, "window": _effective_window(nxt_plan),
                  "from_batch_size": plan["batch_size"], "batch_size": nxt_plan["batch_size"],
                  "from_chunk_size": plan["chunk_size"], "chunk_size": nxt_plan["chunk_size"],
                  "rung": plan.get("name", "replanned"), "shortfall": shortfall}
        self.steps.append(record)
        self._announce(
            "out of memory at frame {} — stepping the temporal window from {} to {}. The frames "
            "before this point were made at the wider window, so temporal consistency changes "
            "mid-clip; it beats losing the job and it is not something to find out from a "
            "file.".format(frames_written, window, _effective_window(nxt_plan)))
        return {"plan": nxt_plan, "args": self._rebuild_args(nxt_plan),
                "capture": self._reopen(), "lead_in": nxt_plan.get("temporal_overlap", 0),
                "record": record}

    def _announce(self, message):
        self._warnings.append(message)
        self._progress.phase("ratchet", force=True, note=message)


def _upscale_once(cli, request, source, source_path, master_path, plan, progress,
                  estimated_frames, still=False, keep_alpha=False, exact_size=None,
                  warnings=None, load_strip=None, codec=None, encoder_out=None):
    args = pipeline.build_args(cli, plan, source_path, MODEL_BUILD,
                               request["color_correction"], debug=request["debug"])
    capture, shape = pipeline.open_source(cli, source_path, keep_alpha=keep_alpha)
    try:
        width, height = exact_size or estimator.output_dimensions(
            source["width"], source["height"], plan["target_short_edge_px"])
        identity = {
            "cf_request_id": request["request_id"],
            "cf_worker_version": WORKER_VERSION,
            "cf_model_build": MODEL_BUILD,
            "cf_output": "{}x{}".format(width, height),
        }
        progress.begin_phase()
        # A budget large enough never to truncate: the generator stops when the decode is
        # exhausted, so this bounds the read from above while the decode determines it below.
        budget = (estimated_frames or 0) * 2 + 10_000

        # **Audio is carried through by default, and this differs from the platform's
        # `keep_audio: false` on purpose.** That default exists because several generators invent
        # a soundtrack nobody asked for, so silence is the safe answer. Here the track is the
        # *customer's own source audio*, and an upscale that silently returned it muted would be
        # losing something the caller supplied rather than suppressing something a model made up.
        # The model itself carries no audio at all (`docs/decisions.md` 0.4), so this comes out
        # of the worker's own mux as a stream copy — no quality cost, near-zero time.
        #
        # Settled with CF 2026-08-12: `keep_audio` is a named field in `params`, default `true`
        # for this task type, and CF sends it explicitly. The default inverting the media
        # worker's is the same rule applied to a different question, not an inconsistency.
        audio_source = source_path if (source["has_audio"] and request["keep_audio"]) else None

        # A still gets a lossless, alpha-capable master; anything with a time axis gets the H.264
        # one. `keep_alpha` only ever reaches the still branch — the handler refuses to set it
        # otherwise — so the writer that cannot hold four channels is never handed four.
        if still:
            writer_cm = encoder.StillWriter(master_path, width, height, identity,
                                            channels=4 if keep_alpha else 3)
        else:
            # **`crf` is the caller's, and only on the master.** Default 12, which is what this
            # writer has silently used since the first commit — so an ordinary request encodes
            # exactly as it always did and a caller who names a value gets theirs. The derives
            # keep their own settings, which belong to the parked encoder track. One caveat rides
            # with it: the host-tail time and memory terms were measured at crf 12, and the
            # single-core encode drain lengthens as crf drops, so a CRF experiment reads the
            # `[host]` tail banner like any other probe.
            # **Handed to the caller as soon as it exists, so a FAILED attempt can still be
            # asked what the encoder cost.** `__exit__` drains and finishes sampling on the
            # exception path exactly as it does on the success path — but this function raises
            # before its `return`, so without this the number is measured and thrown away. The
            # instrument's own docstring cites "the 8K run died at ~46 GiB inside x264's working
            # set... the ceiling had to be inferred from a kernel kill": that run is a FAILURE,
            # and it was the only one that still recorded nothing.
            writer_cm = encoder.MasterWriter(master_path, width, height, shape["fps"] or 30.0,
                                             identity, audio_source=audio_source,
                                             audio_codec=source.get("audio_codec"),
                                             audio_limit_s=source.get("video_duration_s"),
                                             crf=request.get("crf", encoder.DEFAULT_CRF),
                                             preset=request.get("preset",
                                                                encoder.DEFAULT_PRESET),
                                             # **A LITERAL `False`, ON PURPOSE. Do not replace
                                             # it with `envelope.HEAD_KEYFRAMES_DEFAULT`, and
                                             # that is exactly what a later tidy-up will reach
                                             # for**, because `crf` and `preset` two lines above
                                             # import their defaults and this does not.
                                             #
                                             # The copy guards a request that reached the writer
                                             # without passing the surface — **and the guard
                                             # works only because the literal is NOT the
                                             # constant.** Import it, and the day the configured
                                             # default moves to `True` this line silently arms
                                             # the flag on precisely the requests that skipped
                                             # validation. **The SAFE value and the CONFIGURED
                                             # value are different things here, and this is a
                                             # copy of the safe one.**
                                             #
                                             # `crf` and `preset` are not the same case: their
                                             # defaults are values rather than switches, so a
                                             # bypass carrying `DEFAULT_CRF` delivers what
                                             # production delivers. `head_keyframes` is the only
                                             # field in this call where the two can diverge.
                                             head_keyframes=request.get(
                                                 "head_keyframes", False),
                                             # Same literal-default reasoning as above: the safe
                                             # value for each is the one that places no keyframe.
                                             keyframes=request.get("keyframes", "default"),
                                             keyframe_frames=request.get("keyframe_frames"),
                                             keyframe_seconds=request.get("keyframe_seconds"),
                                             # Already resolved in `handle`, right after the
                                             # probe — see the comment there for why it cannot
                                             # happen at the door and must not happen here.
                                             # **`or DEFAULT_CODEC`, because `codec` is None on
                                             # every path that does not resolve one.** A still
                                             # never reaches this writer, so today the fallback
                                             # cannot fire — but `CODEC_LIBRARIES[None]` is a
                                             # KeyError raised inside `_start()`, mid-encode,
                                             # with a delivered master's worth of work already
                                             # spent. A default that cannot be reached costs
                                             # nothing; the crash it prevents costs a job.
                                             codec=codec or encoder.DEFAULT_CODEC)
        if encoder_out is not None:
            encoder_out["writer"] = writer_cm
        # **The policy the stream consults when a chunk runs out of memory.** Built here because
        # this is where the rung ladder, the request and the source path all exist; consumed
        # inside the stream, which knows how to resume and nothing about rungs.
        #
        # `reopen` hands back a decoder at frame zero and the stream seeks it forward. Re-opening
        # rather than rewinding: the capture that OOMed is mid-read with the vendored reader's
        # state on it, and a decoder reset is one more thing that would have to be right.
        def _reopen():
            fresh, _shape = pipeline.open_source(cli, source_path, keep_alpha=keep_alpha)
            return fresh

        def _rebuild_args(new_plan):
            return pipeline.build_args(cli, new_plan, source_path, MODEL_BUILD,
                                       request["color_correction"], debug=request["debug"])

        # **Video only.** A still is one frame: there is nothing written to keep and nothing to
        # resume into, so the job-level retry is already the right and only mechanism.
        ratchet = None if still else _Ratchet(
            estimated_frames=estimated_frames, request=request,
            source_path=source_path, progress=progress,
            warnings=warnings if warnings is not None else [],
            rebuild_args=_rebuild_args, reopen=_reopen,
            source_pixels=(source.get("width") or 0) * (source.get("height") or 0),
            # Computed rather than taken from the rationale, which is not in scope here: relief
            # sizes a decode tile against the output frame, and a tile larger than the picture is
            # a tile that does nothing.
            output_pixels=_output_pixels_for(source, plan),
            # **The frontier's inputs, carried so the walk after an OOM can re-plan rather than
            # step a list.** Dimensions rather than pixel counts: a decode grid is chosen against
            # the real 16:9 frame, and a square of the same area gives a different grid.
            job_shape={
                "target_short_edge_px": plan["target_short_edge_px"],
                "source_width": source.get("width") or 0,
                "source_height": source.get("height") or 0,
                "output_width": (exact_size or estimator.output_dimensions(
                    source.get("width") or 0, source.get("height") or 0,
                    plan["target_short_edge_px"]))[0],
                "output_height": (exact_size or estimator.output_dimensions(
                    source.get("width") or 0, source.get("height") or 0,
                    plan["target_short_edge_px"]))[1],
                "estimated_frames": estimated_frames,
            },
            snapshot=hardware.read(),
            predicted_peak_gb=None)

        # **The guard is built from the plan and fed the container** (F-2026-08-20-42). Its whole
        # design is the planner's own law re-run at execution time with live numbers, so it is
        # constructed from exactly what the plan believed and nothing else — a guard carrying its
        # own private model would be a second thing to keep true.
        machine_now = hardware.read()
        # **One schedule, two parties.** The plan decides the rung; the guard may promote it
        # mid-run (amendment 9 makes rung 2 the guard's first remedy and refusal its fallback).
        # Both have to be looking at the same object or a promotion would arm a schedule nobody
        # consults — a rescue that narrates itself, which is the exact defect F-46 was filed for.
        schedule = pipeline.ResidencySchedule(plan)
        guard = hostguard.HostGuard(
            machine_now.get("host_ram_gb"),
            (source.get("width") or 0) * (source.get("height") or 0),
            _output_pixels_for(source, plan),
            estimated_frames,
            plan.get("batch_size") or 1,
            still=still,
            gpu_name=machine_now.get("gpu_name"),
            # **The schedule itself, not its `promote` alone** (F-2026-08-21-50): the guard has
            # to read `pending` as well as call `promote`, and two half-wired channels is how the
            # rescue got armed and then refused before it could run.
            schedule=schedule,
            # **The chunk the writer flushes at, not the clip** (F-2026-08-21-52). Canvases do
            # not accumulate across a hard cut, and a guard told only the job's frame count
            # projects a container no chunked run ever builds.
            chunk_frames=plan.get("chunk_size"))

        # **The heartbeat wraps the writer, and the nesting order is the whole point.** The drain
        # happens on `writer_cm.__exit__` — inside ffmpeg, with nothing calling back — and
        # context managers exit in reverse, so the timer is still running while it drains. An
        # 11-minute payload silence was measured exactly there on a healthy B200 tail, against
        # the worker's own 33-second promise; from outside that is indistinguishable from a
        # corpse, and three paid jobs have already been cancelled into that ambiguity.
        with progress.keeping_the_promise(), writer_cm as writer:
            # **Whose upscaler touches the alpha.** Off, the reader takes the channel out and
            # `resize_alpha` puts it back with Lanczos — measured correct at +1.000, and blind to
            # where the model moved the edges. On, the model keeps all four and runs
            # `edge_guided_alpha_upscale`, which follows the RGB's own edges and treats a binary
            # mask differently from a gradient alpha. That is the better answer for a cutout on
            # every reading of the code and is measured at nothing, which is why it is a flag.
            # `decisions.md` 4.9.
            written = pipeline.run(cli, capture, args, plan, budget, writer,
                                   on_chunk=progress.frames,
                                   # **The heartbeat, and it is not the same thing as `on_chunk`.**
                                   # `frames_done` cannot advance here -- frames are not written
                                   # until the chunk yields -- so this re-mints the payload with
                                   # the phase and batch it is on, and interpolates `pct`. What it
                                   # buys is the difference between a working job and a corpse,
                                   # which a 192-frame single-chunk run made indistinguishable for
                                   # 904 seconds.
                                   # **And the guard, on the same hook** (F-2026-08-20-42). A
                                   # pass boundary is the last point at which this process still
                                   # exists and can say something, and the point at which the
                                   # remaining work is known exactly. It raises
                                   # `host_capacity_exceeded` rather than letting the kernel take
                                   # the container with no exception, no bundle and a platform
                                   # retry that repeats it.
                                   on_batch=_first_phase_closes_the_strip(
                                       lambda phase, index, total: (
                                           guard.sample(frames_done=writer.frames_written,
                                                        phase=phase),
                                           progress.working(phase, index, total,
                                                            chunk_frames=plan["chunk_size"]))[1],
                                       into=load_strip),
                                   keep_alpha=keep_alpha,
                                   alpha_through_model=bool(
                                       request.get("keep_alpha_in_model")),
                                   exact_size=exact_size, ratchet=ratchet,
                                   schedule=schedule)
            # **Inside the `with`, where it says it is.** The model has finished and the encoder
            # still holds everything it was fed; the drain itself happens on the block's exit.
            _say_host("tail-in", writer)
            tail_opened = time.time()
        # And again once the drain is over — for `VmHWM`, which is the only reading that can
        # report a peak nothing was watching for. Everything between these two lines — the
        # multi-core assembly, the single-core encode drain, the 8-12-core segment that reached
        # ~110 GiB at 8K — is invisible to any exception, because a host breach is a cgroup
        # SIGKILL that writes no bundle and raises nothing.
        _say_host("tail")
        # **The tail, measured, because it is the time model's real liar** (F-2026-08-20-44 and
        # the B200 pair). It is not one of the vendored phases — it is everything between the
        # model finishing and the encoder draining — so nothing was timing it, and a per-frame
        # rate computed over the whole job silently carries it. Two identical warm runs differed
        # by 863 s here and by almost nothing anywhere else.
        tail_seconds = round(time.time() - tail_opened, 1)
        print("[host] tail drained in {:.1f}s".format(tail_seconds))
        # **Whatever decoder the stream finished on.** A ratchet re-opens the source, so asking
        # the original capture whether frames are left would be asking one that was abandoned
        # where the OOM happened — it would answer "plenty" and fail a job that delivered
        # everything.
        pipeline.assert_source_exhausted(
            cli, getattr(pipeline.run, "last_capture", capture), written)
    finally:
        capture.release()
        # **The model goes home with the job** (F-2026-08-20-45). Within-job reuse is the fix;
        # between-job residency is Build D's ruling, and a bug fix must not deliver a policy
        # nobody chose as a side effect. In the `finally` because the run that most needs the
        # memory back is the one that just failed to fit in it.
        # **Handed the job's own dict, or it clears only the far side** (found in review of
        # C+3). Without it, `runner_cache["runner"].dit` — 16.4 GiB — stayed reachable from
        # `pipeline.run.last_phases`, a function attribute, across the idle window and across a
        # retry: attempt 2 would materialise a second copy beside attempt 1's, on the run that
        # had just failed to fit. The release now collects and reads anon on both sides, like
        # every other eviction in this worker.
        pipeline.release_runner_cache(log=print)
    # What the model actually produced, on the 0-255 scale. **Reported, never judged.** A blank
    # output and a photograph of a white wall are indistinguishable from inside the worker, so
    # refusing on "looks flat" would reject legitimate work. Carrying the figure into the manifest
    # means a run that produced nothing is visible afterwards without a human having to download
    # and open every master to find out.
    # **When the model's own size differs from the one we computed, say so.** It is not an error
    # — the model's grid is the model's business and the writers now adopt it — but it is the
    # difference between the size the caller asked for and the size they got, and a caller who
    # laid out a page against `target_short_edge_px` deserves to see it rather than discover it.
    # ---- the keyframes the encode never reached -------------------------------------------
    #
    # **Detected here because here is where the truth is.** `probe_source` returns no frame count,
    # and `nb_frames` is a header claim this project already refuses to trust. `frames_written` is
    # counted by the writer, one per accepted frame, so a request naming frame 999 on a 121-frame
    # clip is only knowable now.
    #
    # **REFUSE, AND THE MASTER GOES WITH IT — CF, 2026-08-26.** I framed this as destroying a
    # correct master over a typo. CF's framing is the one that decides it: *"it might be that the
    # user wanted frame 99 and we didn't give it to him, failed product."* A caller who typed 999
    # for 99 takes delivery of a file that looks finished and discovers in an edit suite that it
    # does not cut where they asked. **That is this project's own defect class shipped to a
    # customer — a thing that looks correct and fails at the case it exists for.** A refusal costs
    # a re-run; a wrong master costs the belief that cut points are where the request put them.
    #
    # **The message must be enough to fix the request in ONE attempt**, because it is the only
    # thing a caller gets back for a whole job's spend. A bare "out of range" would make them
    # bisect a list at 25 minutes of H200 per guess.
    unplaced = (writer_cm.unplaced_keyframes()
                if hasattr(writer_cm, "unplaced_keyframes") else [])
    if unplaced:
        # **Reported from the SAME counter the refusal was decided by.** The check compares against
        # `writer_cm.frames_written`; formatting the message from `written` — `pipeline.run`'s
        # return — would tell a caller the last legal index against a quantity the decision never
        # used. They are expected to agree, and the message is the only thing a caller gets back
        # for a whole job's spend, so it is not the place to find out they do not.
        encoded = writer_cm.frames_written
        # **No "last legal index" when there is none.** `max(encoded - 1, 0)` would tell a caller
        # that index 0 is legal on a clip that encoded nothing at all.
        last = ("frames are numbered from 0, so the last one is {}".format(encoded - 1)
                if encoded else "this clip encoded nothing, so no frame number is legal")
        raise WorkerError(
            errors.INVALID_FIELD_VALUE,
            "keyframes requested at frame {} but this clip encoded {} frames, so they could not "
            "be placed — {}. Nothing was delivered: a master whose cut points are not where the "
            "request put them is worse than no master.".format(
                ", ".join(str(f) for f in unplaced), encoded, last),
        )

    return {"decoded_in": written, "written_out": written,
            "pixels": getattr(pipeline.run, "last_pixel_stats", None),
            "predicted_size": (width, height),
            # **Carried out of here because it is the term a per-frame rate must exclude.** The
            # tail is not a vendored phase and nothing was timing it; a rate computed over the
            # whole job silently includes it, and the B200 pair's whole 17.9-to-21.7 s/frame
            # spread was this number differing by 863 s between two otherwise identical runs.
            "tail_seconds": tail_seconds,
            # **The encoder's own high-water mark, carried out for the same reason `tail_seconds`
            # is: it is measured here and banked two functions away.** `phasewatch` reads
            # `/proc/self`, which is this worker with the model resident, so it can never say what
            # the ENCODER cost — a different process, not a different sampling strategy. Read off
            # the writer after its context manager has exited, which is where the drain sampling
            # finishes; reading it earlier would return the fed part of the encode only.
            "encoder_peak_rss_gb": getattr(writer_cm, "encoder_peak_rss_gb", None),
            "actual_size": getattr(pipeline.run, "last_output_size", None)}


def _write_diagnostics(request, machine, attempts, exception, captured, failed,
                       trace=None, warnings=None, job=None, started=None):
    """Never fails the job. The one outcome worse than losing the diagnostics is losing the
    result because the diagnostics could not be stored."""
    try:
        body = diagnostics.bundle(
            request["request_id"], machine, attempts,
            exception=exception, log_text=captured.text(),
            request=request, rationale=(trace or {}).get("rationale"),
            warnings=warnings, job=job, started=started,
            build=build_identity(),
            extra={"source": (trace or {}).get("source")})
        # The job's own destination first, the kept reserve second. They are not interchangeable:
        # the per-job URL is what CF correlates with the request, and the reserve exists for the
        # case where there is no per-job URL to use — a request that never carried one, or one
        # whose `diagnostics` could not be minted at submit.
        storage.put_diagnostics(
            request.get("diagnostics") or diagnostics.reserve(), body)
    except Exception:  # noqa: BLE001 — see the docstring
        pass


def _decorate(payload, machine, attempts, warnings, progress, started, debug=None):
    payload["hardware"] = machine
    # **`debug` is recorded, on every response shape** (api.md section 5, CF 2026-08-28). It
    # reaches two `colorfix` calls today and appeared in no manifest, no response and no
    # run-record, so a run that used a calibration lever was indistinguishable from one that did
    # not -- which would make the gate a label rather than a control.
    #
    # Set here rather than at the four call sites because this is the one function every shape
    # passes through: delivered, refused, internal and plan-only. Always present and never
    # omitted-when-false, so its absence in an old record stays distinguishable from a false.
    payload["debug"] = bool(debug)
    payload["execution_ms"] = int((time.time() - started) * 1000)
    if attempts:
        payload.setdefault("attempts", attempts)
    if progress.emitted:
        payload["progress_emitted"] = len(progress.emitted)
    return payload


def _write_to_the_reserve(exception, job=None, note=None):
    """The bundle for a failure with no job to attach it to. **Never raises.**

    This is the whole point of `diagnostics_reserve`. Everything `_write_diagnostics` covers has a
    validated request and a per-job URL that came with it; the failures CF cannot see at all are
    the ones that happen where neither exists — an escape past `handle`, or the process falling
    over outside any job. There is no `request` here by construction, so the bundle carries what
    can still be known: the exception, the hardware and the build.
    """
    try:
        destination = diagnostics.reserve()
        if not destination:
            return False
        body = diagnostics.bundle(
            "unattributed", hardware.read(), [], exception=exception,
            job=job, started=time.time(),
            build=build_identity(),
            extra={"note": note or "no request was in scope when this failed"})
        return storage.put_diagnostics(destination, body)
    except Exception:  # noqa: BLE001 — a last resort that raises is not one
        return False


def handler(job):
    """RunPod's entrypoint. **Errors ride the output envelope while the job still COMPLETES.**"""
    try:
        return handle(job.get("input") or {}, job)
    except Exception as exc:  # noqa: BLE001 — the last line of defence
        traceback.print_exc()
        # Past `handle`, so past everything that knew where this job's diagnostics go.
        _write_to_the_reserve(exc, job=job, note="escaped handle(); no validated request")
        return {"cf_error": {"code": errors.INTERNAL,
                             "message": "{}: {}".format(type(exc).__name__, exc)}}


if __name__ == "__main__":
    import runpod

    # **The serve loop itself, because a worker that dies here dies silently.** A driver mismatch
    # or a weight that will not load takes the process with it, and RunPod's answer to CF is a
    # worker that stopped — with the reason only in a log stream nothing scrapes.
    #
    # It reports only if a previous job left a reserve behind, so the very first container of a
    # broken build still cannot say anything. Accepted, and it is the stated limit of the design
    # rather than an oversight: an endpoint with zero successful jobs is not a subtle signal.
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _write_to_the_reserve(exc, note="the serve loop exited; no job was in scope")
        raise
