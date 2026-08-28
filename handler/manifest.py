"""The result manifest, written beside the output on every successful job.

**Not a diagnostic. A recovery path, and it exists because the job record does not outlive the
job.** RunPod retains an async result for **30 minutes** after completion and deletes the job
entirely at its `ttl`. On a job measured in hours, a poller that stops for half an hour — a
deploy, a restart, a network partition — comes back to a request it cannot ask about. The bytes
are in R2 and the job that made them is gone.

CF is not blind in that state, but it is blinder than on the image path: **it recorded the
prefix, not the keys**, so recovery starts with a `ListObjectsV2` and finds some set of files
whose names it did not choose. `keys.py` is what makes that listing predictable. This file is
what turns it into a completed request.

**A listing still cannot recover what only the worker knows** — the output's real dimensions,
frame count in and out, whether it tiled, whether it retried, the worker's own measured elapsed.
Without those, a recovered request is a set of files with no account of itself.

**The response stays the primary path and this changes nothing about it.** But a manifest in
durable storage survives the job record and the response does not.

One thing this carries that CF cannot recover any other way: **RunPod's `executionTime` is not
recoverable after the job record expires.** It exists only on the settled body, and the billing
API does not substitute — its records are aggregated per endpoint over a time bucket, carrying
`timeBilledMs`, `amount` and `gpuTypeId` with no job id at all, and `timeBilledMs` is
worker-alive milliseconds rather than execution time. So even a bucket narrow enough to contain
one job would return the wrong quantity. The worker's own measured elapsed is the closest thing
there is, and it goes in **labelled as the worker's** so CF can decide what to settle on.
"""

import json
import os

import planner

#: Bumped when a field changes meaning, never when one is added. A reader that finds an
#: unfamiliar version should say so rather than guess at the fields it recognises.
MANIFEST_VERSION = 1


def _delivered_sub_floor(result, attempts):
    """Did the attempt that produced the output run below this shot's quality floor?

    **Read off what ran, never off what was planned.** See the call site for why the plan is the
    wrong source. Returns None when it cannot be determined rather than guessing False.
    """
    frames = ((result or {}).get("frames") or {}).get("decoded_in")
    if not frames or frames <= 1:
        # A still has no temporal window, so the floor does not apply and the question is not
        # meaningful rather than answered False -- `planner.window_floor` says the same thing.
        return False if frames == 1 else None
    # **The attempt that PRODUCED THE OUTPUT, named by its own outcome rather than by its
    # position.** `handler.py` stamps `outcome: "ok"` on the attempt that succeeded, and the
    # calibration table already states this rule for `peak_vram_gb` -- "the attempt that produced
    # the output, not the first attempt tried". Taking the last entry would agree with that on
    # an ordinary walk and disagree on any path that appends after the win.
    ok = [a for a in (attempts or []) if (a or {}).get("outcome") == "ok"
          and (a or {}).get("window") is not None]
    if not ok:
        return None
    return int(ok[-1]["window"]) < planner.window_floor(int(frames))


def build(request, result, hardware, attempts, worker_version, model_build, job=None,
          build_identity=None):
    """The manifest body. Carries what the response envelope carries, plus the identity a
    recovered request needs to know what it is looking at."""
    body = {
        "manifest_version": MANIFEST_VERSION,
        "request_id": request["request_id"],
        "worker_version": worker_version,
        "model_build": model_build,
        # **Which image produced this, beside the two fields that only half-answered it.**
        # `worker_version` and `model_build` say what code and what checkpoint, and both were
        # already here; neither says which build, and a VRAM measurement is a measurement of a
        # build. Recorded here rather than in the response envelope because the envelope is CF's
        # contract and a field CF can see is a field CF may come to depend on — this is identity
        # for whoever reads the artefacts later, which is exactly what a manifest is for.
        "build": build_identity or {},
        # **The constants that planned this run, beside the build that carried them.** A
        # measurement is only meaningful against the numbers that produced it, and CF compares
        # the version its embedded predicate reports against this one to catch the two drifting
        # apart — which is the intended use rather than a nicety (CF, 2026-08-18).
        "registry_version": planner.REGISTRY_VERSION,
        # **RunPod's own handle for the job that wrote this.** The master lands at a prefix the
        # *caller* chose, so a caller who has lost the job id can always find the artefacts — but
        # not the running job: there is no endpoint that lists jobs, so without the id there is no
        # `/status` and no `/cancel`. Recording it here closes the loop in the other direction —
        # find the prefix, read the manifest, recover the handle.
        "runpod": {
            "job_id": (job or {}).get("id"),
            "worker_id": os.environ.get("RUNPOD_POD_ID"),
            "endpoint_id": os.environ.get("RUNPOD_ENDPOINT_ID"),
        },
        # Identity of the request, so a recovery can tell what was asked for rather than only
        # what came out. `source_url` is deliberately absent: it is a live credential.
        "requested": {
            "target_short_edge_px": request["target_short_edge_px"],
            "color_correction": request["color_correction"],
            "keep_audio": request["keep_audio"],
            "allow_oom_retry": request["allow_oom_retry"],
            # **What was ASKED, which is not the same fact as what happened.** A caller can send
            # this and still be planned at or above the floor, in which case the flag changed
            # nothing and the master carries the guarantee it always did. The outcome lives in
            # `estimate.sub_floor`, and marking a run on the request alone would flag jobs that
            # ran entirely normally. `estimator._terminal_options` promises the caller that "the
            # job is flagged in its manifest" -- this pair is that promise, kept honestly.
            "allow_below_quality_floor": request.get("allow_below_quality_floor", False),
            # **The two caller levers, recorded because they are part of what the output is.**
            # A master's CRF is a property of that master and cannot be recovered from the file
            # with any confidence; `tile_quality` decided the decode grid, and therefore the
            # seams, which is exactly the kind of thing a later A/B has to be able to key on.
            "crf": request.get("crf"),
            "tile_quality": request.get("tile_quality"),
            "schedule": request.get("schedule"),
            "derive": request["derive"],
        },
        "output": result["output"],
        "derived": result.get("derived", []),
        "frames": result["frames"],
        # **What actually RAN, derived from the attempt that produced the output rather than
        # from the plan.** `estimate.sub_floor` is the INITIAL plan's answer and it does not
        # follow an OOM correction: `solver.next_after_oom` can step the window below the floor
        # on a job whose first plan was in spec, the attempts change, and the rationale does
        # not. Reading the estimate would then tell a reader that a sub-floor master carries a
        # guarantee it does not have -- a number describing the plan that failed rather than the
        # one that delivered.
        #
        # Computed from `window` on the winning attempt, which every attempt carries as
        # `min(batch, chunk)`, against this shot's own floor. `None` rather than False when
        # there is no attempt or no window to read, because "we could not tell" and "it was in
        # spec" are different claims and only one of them is safe to make about a delivered
        # file.
        "sub_floor": _delivered_sub_floor(result, attempts),
        # Mean and spread of every pixel written, on the 0-255 scale. **Reported, never judged.**
        # Two real images once came back as white canvases with correct dimensions, correct keys,
        # a manifest and a COMPLETED status — nothing in the envelope said the picture was blank.
        # A standard deviation near zero is a broken run or a photograph of a white wall, and the
        # worker cannot tell those apart, so it records the figure and a human decides.
        "pixels": result.get("pixels"),
        "estimate": result["estimate"],
        "hardware": hardware,
        # What each attempt was configured with, what it cost and how it ended. On a job that
        # retried this is the pair the estimator's table is built from: the estimate that was
        # wrong and the configuration that worked.
        "attempts": attempts,
        "timings": result["timings"],
    }
    if result.get("warnings"):
        body["warnings"] = result["warnings"]
    return body


def serialise(body):
    return json.dumps(body, indent=2, sort_keys=False, default=str)


def identity_tags(request, result, worker_version, model_build):
    """The `-metadata` tags stamped into the video, in the mux already being done.

    **Identity only, and nothing a customer should not read** — this file is delivered. Timings,
    hardware, tiling configuration, worker ids and anything resembling a credential stay in the
    manifest and the diagnostics bundle.

    It is a recovery aid and never a source of truth: CF's standing rule is to read the worker's
    reported fields rather than re-probe the file, because a probe cannot reliably say which
    extent it is reporting — which is how a 342/360 disagreement reached production once. These
    are what someone falls back to when the response and the manifest are both gone, and CF
    should have to reach for them consciously.
    """
    return {
        "cf_request_id": request["request_id"],
        "cf_worker_version": worker_version,
        "cf_model_build": model_build,
        "cf_frames_in": result["frames"]["decoded_in"],
        "cf_frames_out": result["frames"]["written_out"],
        "cf_output": "{}x{}".format(result["output"]["width"], result["output"]["height"]),
    }
