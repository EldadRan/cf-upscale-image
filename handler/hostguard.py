"""Refuse while alive, rather than be killed silently. F-2026-08-20-42.

**A cgroup SIGKILL is the only failure this worker cannot report.** VRAM exhaustion raises, is
caught, is diagnosed and is re-planned; the host equivalent removes the process between one
instruction and the next — no exception, no bundle, no walk, no envelope — and RunPod then
restarts the container with no resume, so a deterministic host overrun becomes a retry loop that
bills repeatedly and writes nothing. F-41 died that way twice on the same plan, and the only
reason anyone knows what happened is that the platform's own log recorded `exit code 137`.

So the guard runs **the planner's own law at execution time**, with live numbers instead of
predicted ones. That is the whole design in one sentence, and it has two consequences worth
stating: when the table is right this never fires, and when it fires it produces a calibration
datum — a measured drift between what the model said and what the container is doing — instead
of an exit code.

**The trigger is a projection, not a percentage.** A bare threshold was explicitly rejected in
the ruling and the corpus says why: the F-41 staircase survived spikes at 97–99% of its limit
and kept going, so a naive 95% or 98% rule would have killed runs that were about to succeed.
What distinguishes a survivable spike from certain death is not the current reading, it is
whether the *remaining* work still fits — which is arithmetic this worker already owns.

A percentage does appear, but only as a backstop and only with a second condition attached: a
container sustaining ≥98% across consecutive samples **while still ramping** is not spiking, it
is arriving. That combination is what the staircase never showed.
"""

import errors


#: Sustained share of the slice that trips the backstop, and how many consecutive samples must
#: agree. Two, not one: a single sample at 98% is exactly the spike the staircase survived.
BACKSTOP_SHARE = 0.98
BACKSTOP_SAMPLES = 2

#: **Where a verdict may be reached** (F-2026-08-21-53). F-42's ruling is one phrase — "between
#: decode passes" — and this module's own docstring has quoted it since Build B/2 while the call
#: site fed it *every* phase boundary the vendored loop announces. The deviation was latent for
#: as long as the arithmetic was so wrong that it convicted early anyway; C+1's honest numbers
#: made it live, and both dit-entry refusals walked through this door.
#:
#: The ruling is not arbitrary about the phase. Between decode passes the remaining work is known
#: exactly and the peak that kills a container is the next thing to happen. At dit-entry the run
#: has not produced a canvas, the host-resident weights are streaming per batch, and the
#: "remaining work" the projection prices is separated from the reading by three phases of
#: allocation and release — a verdict there is a forecast about a container that does not exist
#: yet.
#:
#: **Other boundaries still sample.** The readings are the corpus — peaks, the anon/file split,
#: the transient's high-water mark — and none of that is a verdict. What they may not do is
#: convict.
CONVICTING_PHASES = ("vae_decode",)


class HostGuard:
    """Watches the host between decode passes and refuses before the kernel does.

    Constructed with what the plan believed; fed what the container is doing. **Never raises
    anything but `WorkerError`**, and returns silently when it cannot read the host at all — a
    guard that fails a job because it could not take a measurement is worse than no guard.
    """

    def __init__(self, limit_gb, source_pixels, output_pixels, frames, window,
                 still=False, gpu_name=None, log=print, schedule=None, chunk_frames=None):
        self.limit_gb = limit_gb
        self.source_pixels = source_pixels
        self.output_pixels = output_pixels
        self.frames = max(1, int(frames or 1))
        #: **The span the canvases are actually co-resident over** (F-2026-08-21-52). The guard
        #: was built with the *job's* frame count and projected the whole clip's canvases into
        #: one container — but the writer flushes at every hard cut, so a chunked run never holds
        #: more than one chunk's worth. On the 240-frame retest that was a 0.42 GiB overcharge
        #: and cost nothing; on the 8K flagship's 3 480-frame chunk-1 it charges 645 GiB, which
        #: refuses instantly and on every tier. `None` means the job is one chunk, where the two
        #: numbers are the same and nothing here changes.
        self.chunk_frames = None if not chunk_frames else max(1, int(chunk_frames))
        self.window = max(1, int(window or 1))
        self.still = still
        self.gpu_name = gpu_name
        self.log = log
        #: **Rung 2 as the first remedy, refusal as the fallback** (amendment 9). The residency
        #: schedule itself, not a bare callable: the guard both arms it (`promote()`) and reads
        #: it (`pending`), and those two have to be the same object or the guard credits an
        #: eviction that already happened. Handing over one object is what makes half-wiring
        #: impossible — F-2026-08-21-50 was the arming half working while the reading half did
        #: not exist. `None` means no ladder is wired, and the guard behaves exactly as it did
        #: before Build D.
        self.schedule = schedule
        #: Promotion is once per **chunk**, not once per run or once per sample. Once per sample
        #: would mean a first attempt that did not free what it claimed gets retried forever; once
        #: per run would mean a chunk inherits a judgement made from another chunk's readings,
        #: which is F-2026-08-21-54 in the guard rather than in the schedule. The seam resets it.
        self.promoted = False
        #: Which chunk the latch above belongs to, read from the schedule the hook maintains.
        self._promoted_in_chunk = None
        #: The largest boundary reading seen, on each axis. **Two, because a delta is only
        #: meaningful between readings of the same kind** (F-2026-08-21-49): `VmHWM` is a peak
        #: *RSS*, so the baseline subtracted from it must be an RSS baseline too.
        self.peak_boundary_gb = 0.0
        self.peak_boundary_rss_gb = 0.0
        #: **`VmHWM` as it stood the first time this guard looked** (F-2026-08-21-51). The
        #: monotone counter carries everything the process has ever touched, including the
        #: instant during load when the checkpoint exists as mmap'd file pages *and* as anon at
        #: once. C+1 fixed the axis and the contamination survived it, because cancelling a
        #: plateau by subtracting a boundary reading only works if a boundary lands on the
        #: plateau — and the materialisation overlap is an instant, not a plateau. Boundary
        #: samples structurally miss it. Differencing the counter against *itself* does not
        #: care: whatever the load spent is inside the baseline, and only growth after the
        #: baseline can be charged.
        self.hwm_baseline_gb = None
        self.observed_transient_gb = 0.0
        self.hot_samples = 0
        self.fired = False
        #: The axis the last sample charged, and the reclaimable cache beside it. Reported in the
        #: refusal so a reader can see which number convicted and what was discounted.
        self.charged_basis = None
        self.file_gb = None
        #: The standing verdict, so a latched refusal re-raises the original
        #: finding rather than re-deriving one from numbers that have moved on.
        self._verdict = None

    # ── the arithmetic ───────────────────────────────────────────────────────────────────────

    def chunk_span(self, frames_done=0):
        """`(frames in the chunk being built, how many of them are still to come)`.

        The writer flushes a chunk to disk at each hard cut, so co-residency resets there. A
        tail chunk is genuinely shorter than the full ones and gets its own, smaller answer —
        which matters twice over, because the postprocess transient is bounded by the frames
        held and a nine-frame tail does not hold a window's worth.
        """
        done = max(0, int(frames_done or 0))
        chunk = self.chunk_frames or self.frames
        start = (done // chunk) * chunk
        this_chunk = max(1, min(chunk, self.frames - start))
        return this_chunk, max(0, this_chunk - (done - start))

    def _next_phase_transient_gb(self, frames_done=0):
        """The margin: **what the next phase will ask for**, not a generic ripple.

        Ruled that way (amendment 5) because the 8K killer is a ~26 GiB postprocess working set
        that appears after decode finishes. A run can sit comfortably through every decode pass
        and still be dead, and a margin that does not price the specific thing coming next would
        not see it.
        """
        import planner  # noqa: PLC0415 — local, keeps this module importable without the chain

        held, _ = self.chunk_span(frames_done)
        return planner.postprocess_transient_gb(self.output_pixels, self.window, held,
                                                still=self.still)

    def _per_frame_gb(self):
        """**What a frame still to be produced adds — the canvas, and not the source stack**
        (F-2026-08-21-49, defect 2).

        The plan prices a chunk's whole resting ramp as `frames × (S*12 + O*6)`, and that is
        right: it is answering "how big does this container have to be". A live projection is
        answering a different question — "how much *more* will it hold" — from a `charged`
        reading that already contains the source stack in full. The acceptance run measured it:
        at dit-entry, charged 25.3 = the 18.4 constant + 5.14 of source + drift, for all 222
        frames. Adding `remaining × (S*12 + O*6)` then charged those same 5.14 GiB a second
        time.

        The condition this rests on, stated so it can be checked rather than assumed: the guard
        samples only at model-batch boundaries, and a chunk's frames are read before the model
        is handed any of them. A sample taken while the source was still arriving would
        under-project by whatever had not landed — there is no such sample, and if one is ever
        added this is the term that has to change with it.
        """
        import planner  # noqa: PLC0415

        return planner.host_canvas_per_frame_gb(self.output_pixels)

    def project(self, charged_gb, frames_done):
        """`(projection, headroom)` — what this run will reach, against what it may use.

        `charged + remaining accumulation + this run's own measured transient`, compared against
        the slice less the next phase's transient. Every term is either a live reading or a
        number fitted from this run, which is what makes a refusal here a measurement.

        **`charged` is anon, not `memory.current`** — see `sample`.
        """
        # **Remaining in the CHUNK, not in the job** (F-2026-08-21-52) — see `chunk_span`.
        _, remaining = self.chunk_span(frames_done)
        # **One spike, charged once** (F-2026-08-21-49, defect 3). This added the transient this
        # run has shown to the projection *and* subtracted the next phase's from the budget —
        # room demanded for two simultaneous spikes, when a spike is by definition the thing
        # that does not coexist with itself. The run holds one working set at a time, so the
        # term is the larger of what has been seen and what is modelled next: measurement where
        # it exists, the registry where it does not, and never their sum.
        spike = max(self.observed_transient_gb, self._next_phase_transient_gb(frames_done))
        projection = charged_gb + remaining * self._per_frame_gb() + spike
        # **A scheduled eviction is credited against the peak it is scheduled for**
        # (F-2026-08-21-50). Amendment 9 says "unload after each chunk's DiT", and that is what
        # the schedule does — so from the moment it is armed, the container that will meet the
        # peak phase is 16.4 GiB lighter than the one standing now. A projection that ignores
        # that refuses the job the rescue was arranged for, which is what the seam retest did:
        # promoted at batch 2 of 7, then refused at a later batch because the number had not
        # moved. **Only while the eviction is pending** — once it has run, the live reading is
        # already the truth and a second credit would under-project by the whole checkpoint.
        credit = self._eviction_credit_gb()
        return projection - credit, self.limit_gb

    def _eviction_credit_gb(self):
        """`MODEL_RESIDENT_GIB` while an eviction is scheduled and unspent, else zero."""
        if self.schedule is None or not getattr(self.schedule, "pending", False):
            return 0.0
        import planner  # noqa: PLC0415

        return planner.MODEL_RESIDENT_GIB

    # ── the sample ───────────────────────────────────────────────────────────────────────────

    def sample(self, frames_done=0, phase=None):
        """Take a reading. Raises `WorkerError` when the run is certainly dead ahead.

        **Called at every phase boundary, and convicting at one of them** (F-2026-08-21-53).
        This docstring used to say "called between decode passes", quoting F-42's ruling, while
        the call site handed it every boundary the vendored loop announces — so the sentence
        described the design and the code did something else, and nothing in the suite compared
        the two. Readings are taken everywhere because they are the corpus; verdicts are reached
        only at `CONVICTING_PHASES`.
        """
        # **Once fired, it keeps firing** (F-2026-08-20-46). This used to return silently after
        # the first refusal, which made the guard disarm itself: a single swallowed raise turned
        # it off permanently. Two observer layers were swallowing it, so the verdict printed once
        # and the run continued to completion — twice, deterministically. A latch that re-raises
        # means a refusal survives *any* single point that eats an exception, which is the only
        # safe assumption about a loop this worker does not own.
        if self.fired:
            self._raise(self._verdict)
        # **A new chunk may climb the ladder on its own account** (F-2026-08-21-54). The schedule
        # counts the seams; the guard only has to notice that the count moved.
        if self.schedule is not None:
            ordinal = getattr(self.schedule, "chunk_ordinal", None)
            if self._promoted_in_chunk is not None and ordinal != self._promoted_in_chunk:
                self.promoted = False
                self._promoted_in_chunk = None
        if not self.limit_gb:
            return None
        import hardware  # noqa: PLC0415 — stdlib-only

        current = hardware.memory_current_gb()
        if current is None:
            return None                    # no cgroup: nothing to guard against, silently

        # **Charge anon; treat file cache as soft** (CF, 2026-08-20, the anon/file caveat).
        # `memory.current` is what the kernel *watches*, which is why the model is priced against
        # it — but it is not what the kernel *cannot reclaim*. On a lazily-materialising card the
        # checkpoint read alone is ~16 GiB of file cache, and the corpus shows the kernel dropping
        # it under pressure: 16.18 -> 12.54 -> 4.88 across one job's boundaries.
        #
        # The refusal this fixes was right and its evidence was not. It reported "using 39.7" at
        # a boundary where anon was 23.71 — **40% of the reading was cache the kernel would hand
        # back**. Recomputed on anon the same refusal still fires (40.84 projected against 34.87
        # allowed), because that job was genuinely doomed; it was doomed by a second copy of the
        # weights (F-45), not by the ramp the projection was pricing.
        #
        # Falling back to `memory.current` when the split is unreadable is deliberate: over-
        # charging delays a job, under-charging loses a container.
        parts = hardware.memory_breakdown_gb()
        anon = parts.get("anon")
        self.file_gb = parts.get("file")
        charged = current if anon is None else anon
        self.charged_basis = "memory.current" if anon is None else "anon"

        self.peak_boundary_gb = max(self.peak_boundary_gb, charged)
        # **`VmHWM` sees what the boundaries cannot.** A per-pass working set is allocated and
        # released between two samples, so the only witness to its size is a monotone high-water
        # mark. This is the same reason the corpus fits the transient from `peak_gb`.
        #
        # **But it is an RSS peak, and it must be measured against an RSS baseline**
        # (F-2026-08-21-49, defect 1). This subtracted `peak_boundary_gb` — *anon* — from a
        # number that counts every resident page including the mmap'd `.safetensors` checkpoint
        # sitting in file cache. So the checkpoint appeared in the minuend and not the
        # subtrahend, and the whole of it could be read as "transient": the acceptance run
        # reported `observed_transient 15.32` on a box whose `reclaimable_file` was 16.71 at the
        # same instant. That is the checkpoint wearing a transient costume — the exact
        # dishonesty ad7707f purged from `charged`, surviving one term over.
        #
        # RSS on both sides, so the checkpoint cancels. The *delta* is then admissible against
        # an anon-charged base: a working set that appears and disappears between two passes is
        # anonymous memory by nature — it is the *level* that is unreadable across axes, never
        # the rise. Where RSS cannot be read the term is zero rather than contaminated: a
        # transient this guard could not measure is one it must not charge for.
        try:
            import phasewatch  # noqa: PLC0415

            rss = phasewatch.host_rss_gb()
            if rss:
                self.peak_boundary_rss_gb = max(self.peak_boundary_rss_gb, rss)
            hwm = phasewatch.host_hwm_gb()
            if hwm:
                # **The first look sets the floor.** By the time the guard is called at all the
                # model has loaded — it runs on model-batch boundaries — so whatever the load
                # peaked at belongs to the load. The retest is the case: file cache 0.69 GiB at
                # refusal, the checkpoint entirely in anon, and 14.87 GiB still charged as an
                # "observed" working set that no phase had allocated.
                if self.hwm_baseline_gb is None:
                    self.hwm_baseline_gb = hwm
                floor = max(self.hwm_baseline_gb, self.peak_boundary_rss_gb)
                self.observed_transient_gb = max(self.observed_transient_gb,
                                                 max(0.0, hwm - floor))
        except Exception:  # noqa: BLE001 — an unreadable counter is not worth failing a job over
            pass

        projection, allowed = self.project(charged, frames_done)
        # The backstop reads the same axis. A container at 98% of which two fifths is reclaimable
        # cache is not at 98% of anything that kills it, and firing there would be the false
        # positive the ruling spent a paragraph forbidding.
        share = charged / float(self.limit_gb)
        ramping = frames_done is not None and frames_done < self.frames

        # **Everything above this line is a reading; everything below it is a verdict**
        # (F-2026-08-21-53). The peaks, the split and the transient are banked at every boundary
        # the loop announces, because they are the corpus. The judgement happens only where
        # F-42's ruling put it. A latched verdict is exempt and re-raises anywhere — see the top
        # of this method: F-46 is about a verdict surviving every door, and this is about which
        # door it may be reached at.
        if phase not in CONVICTING_PHASES:
            self.hot_samples = 0
            return current

        if projection > allowed:
            # **The ladder's remedy comes before the ladder's refusal** (amendment 9): "when the
            # live projection says the resident peak will not fit, its FIRST remedy is rung 2 —
            # evict and continue — and refusal is the fallback, not the reflex."
            #
            # The arithmetic is the same subtraction the planner makes: the peak phase runs
            # without the checkpoint, so scheduling it out buys exactly `MODEL_RESIDENT_GIB`
            # against the projection. Promotion is offered only where it would actually close the
            # gap — arming an eviction that still leaves the container dead would spend a reload
            # to arrive at the same refusal one phase later, having narrated a rescue.
            if self._promote_instead(charged, projection, allowed, frames_done, phase):
                return current
            self._refuse("projection", charged, projection, allowed, frames_done, phase)
        # **The backstop, and both halves are required.** Sustained, because one sample at 98% is
        # the spike the F-41 staircase survived; ramping, because a container that has stopped
        # growing at 98% has arrived rather than overshot.
        if share >= BACKSTOP_SHARE and ramping:
            self.hot_samples += 1
            if self.hot_samples >= BACKSTOP_SAMPLES:
                # Same order here: a container sustaining 98% with a 16.4 GiB checkpoint still
                # home has somewhere to go, and taking it is cheaper than losing the run.
                if self._promote_instead(charged, projection, allowed, frames_done, phase):
                    self.hot_samples = 0
                    return current
                self._refuse("backstop", charged, projection, allowed, frames_done, phase)
        else:
            self.hot_samples = 0
        return current

    def _promote_instead(self, charged, projection, allowed, frames_done, phase):
        """Try rung 2. `True` if the run continues, `False` if the refusal stands.

        **Never during `dit_sample`, and that is not a limitation — it is the point.** The model
        is in use in that phase; what the guard arms is the *schedule*, which fires at the end of
        it. So a breach seen mid-DiT is answered by committing the rest of the run to rung 2, and
        the projection is judged against the container that will exist when the peak arrives
        rather than the one standing now. Evicting the weights out from under the sampler would
        be a crash dressed as a remedy.
        """
        if self.schedule is None or self.promoted:
            return False
        import planner  # noqa: PLC0415

        # `projection` arrives already credited if an eviction was pending, so this asks the
        # honest question either way: is there a checkpoint left to schedule out, and does
        # removing it close the gap?
        if projection - planner.MODEL_RESIDENT_GIB > allowed:
            return False
        self.promoted = True
        self._promoted_in_chunk = getattr(self.schedule, "chunk_ordinal", None)
        try:
            self.schedule.promote(phase)
        except Exception:  # noqa: BLE001 — a promotion that cannot be arranged is a refusal
            self.promoted = False
            return False
        # **The numbers, not the news.** F-42 was filed about a line that read like an action and
        # was not one; a promotion that printed "switching to rung 2" and nothing else would be
        # the same sentence with a happier verb.
        self.log("[host] projection {:.1f} GiB exceeds {:.1f} usable at frame {} of {} (phase "
                 "{}): promoting to rung 2 — the model leaves for each chunk's peak phase, which "
                 "prices the run at {:.1f} GiB. Refusal is what happens if that does not hold."
                 .format(projection, allowed, frames_done, self.frames, phase,
                         projection - planner.MODEL_RESIDENT_GIB))
        return True

    def _raise(self, verdict):
        """Raise the standing verdict. **Separate from deciding it**, so a latched refusal
        re-raises the original finding rather than re-deriving one from numbers that have moved
        on since."""
        raise errors.WorkerError(errors.HOST_CAPACITY_EXCEEDED, verdict["message"],
                                 remedy=errors.Remedy.LARGER_GPU,
                                 shortfall=verdict["shortfall"])

    def _refuse(self, trigger, current, projection, allowed, frames_done, phase):
        self.fired = True
        predicted = None
        try:
            import planner  # noqa: PLC0415

            held, _ = self.chunk_span(frames_done)
            # The chunk, because that is what the plan priced. Comparing a live chunk peak
            # against a whole-job prediction would report a drift that is really a scope
            # mismatch — the same confusion this finding is about, one field over.
            predicted = round(planner.host_peak_gb(
                self.source_pixels, self.output_pixels, held, self.window,
                still=self.still, gpu_name=self.gpu_name), 2)
        except Exception:  # noqa: BLE001
            pass
        drift = None if predicted is None else round(projection - predicted, 2)
        # **The sentence follows the arithmetic** (F-2026-08-21-49). It used to end "against N
        # usable once the next phase's M is reserved", which described a budget with the spike
        # subtracted from it — the double-charge, narrated. One spike, named once, on the side it
        # is now charged on.
        spike = max(self.observed_transient_gb, self._next_phase_transient_gb(frames_done))
        chunk_held, chunk_left = self.chunk_span(frames_done)
        message = (
            "this container will not hold the job it was given. At frame {} of {} it holds "
            "{:.1f} GiB of unreclaimable memory in its {:.1f} GiB slice; the {} canvases still "
            "to be produced plus the {:.1f} GiB working set the peak phase holds project to "
            "{:.1f} GiB, against the {:.1f} GiB the slice actually is. The plan priced this "
            "job at {} GiB{}. Refused here, alive, because the alternative is a cgroup SIGKILL: "
            "no error, no bundle, no partial output, and a platform retry that repeats it."
            .format(frames_done, self.frames, current, self.limit_gb, chunk_left,
                    spike, projection, allowed,
                    "unknown" if predicted is None else "{:.1f}".format(predicted),
                    "" if drift is None else ", drifting {:+.1f} GiB".format(drift)))
        self.log("[host] REFUSING while alive ({} trigger, phase {}): {}".format(
            trigger, phase, message))
        # **The drift, not just the deficit.** A refusal that says only "it did not fit"
        # explains itself and teaches nothing; the gap between what the model predicted and what
        # the container is doing is the number that reprices the constants.
        self._verdict = {
            "message": message,
            "shortfall": {
                "trigger": trigger,
                "phase": phase,
                # **Whether the ladder was tried, so a reader can tell a refusal that had no
                # remedy from one whose remedy was not enough.** Those are different findings:
                # the first reprices the constants, the second reprices the rung.
                "rung_two_attempted": self.promoted,
                "rung_two_available": self.schedule is not None,
                # **What the projection was already forgiven.** A refusal that stands after a
                # 16.4 GiB credit is a different finding from one that never had a rung to
                # climb, and the record must not make a reader guess which it is.
                "eviction_credited_gb": round(self._eviction_credit_gb(), 2),
                "frames_done": frames_done,
                "frames_total": self.frames,
                # **The chunk, named, so a reader can tell a job-scoped number from a
                # chunk-scoped one at a glance** — which is exactly what F-52 was.
                "chunk_frames": chunk_held,
                "chunk_frames_remaining": chunk_left,
                "host_limit_gb": round(self.limit_gb, 2),
                # **Which axis convicted, and what was discounted.** A refusal computed on
                # `memory.current` and one computed on anon can differ by 40% of the reading;
                # a corpus that cannot tell them apart cannot reprice anything from either.
                "charged_gb": round(current, 2),
                "charged_basis": self.charged_basis,
                "reclaimable_file_gb": (None if self.file_gb is None
                                        else round(self.file_gb, 2)),
                "projected_peak_gb": round(projection, 2),
                "allowed_gb": round(allowed, 2),
                # **Both halves of the one spike, so a reader can see which won and why.** The
                # term is `max(observed, modelled)`; a record carrying only the result cannot
                # say whether this refusal rested on a measurement or on the registry, and those
                # reprice different things.
                "spike_charged_gb": round(spike, 2),
                "spike_basis": ("observed" if self.observed_transient_gb
                                >= self._next_phase_transient_gb(frames_done) else "modelled"),
                "canvas_per_frame_gb": round(self._per_frame_gb(), 4),
                "predicted_peak_gb": predicted,
                "model_drift_gb": drift,
                "observed_transient_gb": round(self.observed_transient_gb, 2),
                "next_phase_transient_gb": round(
                    self._next_phase_transient_gb(frames_done), 2),
            },
        }
        self._raise(self._verdict)
