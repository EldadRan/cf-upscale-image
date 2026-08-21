"""Telling CF what is happening, on a job measured in hours.

Measured behaviour, not assumed — playbook §9 is the full account and this is what binds:

  `runpod.serverless.progress_update(job, payload)` writes into `/status`'s `output` field and
  **a plain handler is enough** — no generator, no `/stream`. The payload **keeps its type**, so
  send an object and spare CF a parser.

  **Every update replaces the last. There is no history.** A poller reads the current state and
  cannot tell what it missed between polls, so **anything that must survive belongs in the final
  result**, never here.

  **`executionTime` exists only on the settled body** and there is no running elapsed time. So an
  ETA is this worker's to measure and publish, or it does not exist.

  Updates become visible in under about two seconds, so emitting faster buys resolution the
  transport does not have. The payload is re-sent whole on every update and read whole by every
  poll, so it stays at a few kilobytes.

**The ETA works before any calibration exists, which is the point.** The first chunk measures the
real rate on the real card at the real size, and the remaining frames follow from it. A table
would give a better *first* estimate; nothing gives a better second one. That is also how the
table gets filled — a prediction that reports itself against the outcome improves, and one that
never does cannot.

`next_poll_s` costs nothing and is the same field the ETA travels in: the worker can simply say
*"nothing will change for 90 seconds"*. 30 s of encode with nothing to report should not be
polled twenty times. CF clamps what it honours at both ends, because a worker asking for an hour
and a worker that has hung look identical from outside — so it is a hint CF may shorten, never a
lease.
"""

import contextlib
import threading
import time

#: Emit no faster than this. Updates take ~2 s to become visible; sub-second emission is work
#: for nobody.
MIN_INTERVAL_S = 3.0

#: What CF is asked to wait when the worker knows nothing will change — bounded because a hint
#: CF has to clamp is a hint that was not worth sending.
MIN_POLL_S = 5

#: Shortest gap this worker will believe is one model pass. See `_note_pass`.
MIN_PASS_GAP_S = 1.0
MAX_POLL_S = 90


class Progress:
    """Phase, percentage and a self-calibrating ETA.

    Safe to construct when there is no job to report to: `progress_update` is skipped and the
    payloads are still recorded, which is what lets rung 1 assert the shape without RunPod.
    """

    def __init__(self, job=None, estimated_frames=None, enabled=True):
        self._job = job
        # **RunPod's retry counter, read off the job rather than counted here.** A worker that is
        # restarting cannot count its own restarts — the count died with the previous container —
        # so the only honest source is the platform's, which arrives on the job envelope. Absent
        # (rung 1, or a platform that stops sending it) it reports 0 rather than guessing.
        self._attempt = 0
        if isinstance(job, dict):
            try:
                self._attempt = int(job.get("retries") or 0)
            except (TypeError, ValueError):
                self._attempt = 0
        self._enabled = enabled
        self._estimated_frames = estimated_frames or None
        self._started = time.time()
        self._phase_started = self._started
        self._last_emit = 0.0
        self._seconds_per_frame = None
        #: A rate the planner already predicted, seeded before the run so an ETA exists from the
        #: first heartbeat rather than from the first written frame. Superseded the moment a
        #: chunk lands: this module's whole argument is that the first chunk measures the real
        #: rate on the real card at the real size, and nothing gives a better second estimate.
        self._expected_seconds_per_frame = None
        self._expected_basis = None
        #: Fraction of the clip's *work* completed, phase-weighted — the measure that moves while
        #: a whole-clip chunk is in the model and no frame has been written yet.
        self._work_fraction = 0.0
        #: **How long the last model pass took, measured** — the promise's real basis. A pass is
        #: the unit news arrives in, so the interval between two of them is the interval between
        #: two payloads, and this worker is the only party that can know it.
        self._pass_cadence_s = None
        self._last_pass_at = None
        #: A stretch with no passes to time — the fetch, the load strip, the encoder drain —
        #: still has to keep the promise it made, so `heartbeat()` re-publishes the standing
        #: payload with a fresh timestamp rather than letting `at` age past the interval the
        #: worker itself asked for.
        self._eta_basis = None
        # **State lives here, not in `emitted`.** `_emit` is rate-limited, so anything read back
        # out of the emitted history is missing whatever the limiter dropped. Reading
        # `frames_done` from that list made the ETA compute against zero frames done and report
        # the same figure at 10 frames and at 20 — plausible, wrong, and invisible.
        self._frames_done = 0
        #: The most recent payload, always current whether or not it was emitted.
        self.last = None
        #: What was actually sent. Kept for the final result, because the progress channel keeps
        #: no history and CF's only account of how the estimate behaved is what is returned.
        self.emitted = []

    def expect(self, seconds_per_frame, basis=None):
        """Seed the rate the planner already predicted, before any work has been done.

        **So that an ETA exists during the model stretch** (F-2026-08-19-29). `eta_s` answered
        `None` until a frame was written, and on a whole-clip chunk no frame is written until the
        final drain — so a 1,117 s job published no ETA for 1,061 s of it, and `next_poll_s`
        sat at its 5 s default the whole time because the ETA it keys off did not exist.

        Labelled, never disguised: a predicted rate and a measured one are different kinds of
        thing, and the payload says which is in hand.
        """
        if seconds_per_frame and seconds_per_frame > 0:
            self._expected_seconds_per_frame = float(seconds_per_frame)
            # **Normalised, because two vocabularies were about to share a word.** The
            # rationale's `prediction_basis` says where the *rate* came from — `measured` means
            # "from measured calibration rows", `approximate` means "scaled from another size".
            # `eta_basis` says where *this ETA* came from, and there `measured` means "this run
            # measured it". A rate off a table is a prediction for this run however it was
            # obtained, so it says so, and keeps the provenance as a suffix rather than losing it.
            self._expected_basis = ("predicted" if basis in (None, "measured")
                                    else "predicted_{}".format(basis))

    def phase(self, name, pct=None, force=False, **facts):
        payload = {"phase": name}
        if pct is not None:
            payload["pct"] = int(max(0, min(100, pct)))
        eta = self.eta_s()
        if eta is not None:
            payload["eta_s"] = int(eta)
            payload["eta_basis"] = self._eta_basis
        payload["next_poll_s"] = self._next_poll_s(eta)
        # **The basis of the promise, published beside it** (CF ruling, 2026-08-20). A client that
        # can see the cadence the worker measured can check the promise rather than trust it, and
        # the whole cadence investigation had to be run from outside precisely because the
        # payload said what to do and never why.
        if self._pass_cadence_s:
            payload["pass_cadence_s"] = round(self._pass_cadence_s, 1)
        # **When this was minted, and which run of the job minted it.** Every update replaces the
        # last and there is no history, so a payload that stops being replaced *keeps being
        # served* — indistinguishable from a live one. On 2026-08-14 a job died at 97% and
        # `/status` went on answering `IN_PROGRESS` with `frames_done: 903` from the attempt that
        # no longer existed. The status was not wrong about anything it said; it simply had no way
        # to say when it said it.
        #
        # `attempt` is the other half. RunPod's own `retries` counter had incremented to 1, which
        # was the only evidence anywhere that the work had restarted from frame zero — and it sat
        # beside a progress payload from the previous attempt saying 97%. With this, a reader
        # comparing the attempt it last saw against the attempt in hand sees the restart directly.
        payload["at"] = int(time.time())
        payload["attempt"] = self._attempt
        payload["elapsed_s"] = int(time.time() - self._started)
        payload.update(facts)
        self._emit(payload, force=force)
        return payload

    def frames(self, done, phase="upscale", boundary=True):
        """Called as frames are written. **The rate comes from chunk boundaries only.**

        Measured from the first completed chunk onward, so it reflects this card, this size and
        this configuration rather than an average over runs that were none of those things.

        **`boundary` is what keeps the rate honest** (F-2026-08-18-26). Since the count began
        advancing mid-chunk, this is also called every sixteen frames — and mid-chunk `elapsed`
        covers the whole chunk's model time while `done` counts only the frames written so far,
        so dividing one by the other inflates the rate by `chunk / 16`. On a 192-frame chunk the
        first update would have claimed a twelvefold slowdown, which the ETA and the deadline
        guard both consume. So the count advances on every call and the *rate* only where the two
        quantities are commensurable: at a boundary, where a whole chunk's frames have been
        written for a whole chunk's compute.

        The boundary call is **forced** past the rate limiter. It carries the only corrected rate
        in the chunk, and a three-second window that has just swallowed a mid-chunk update would
        otherwise swallow the correction with it.
        """
        elapsed = time.time() - self._phase_started
        self._frames_done = done
        if boundary and done > 0 and elapsed > 0:
            self._seconds_per_frame = elapsed / float(done)
        pct = None
        if self._estimated_frames:
            # **The published pct is the guarded one, not the raw frame ratio** (F-34). This line
            # used to publish `100 x done/estimated` directly, which collapsed the counter at the
            # drain seam: the model stretch climbs a phase-weighted pct to 78, then the first
            # frames are written and the raw ratio publishes 16/192 = 8. Observed live on the
            # first 8K customer delivery — 78 -> 8 -> 66 -> 83 — with no OOM and no restart, the
            # arithmetic identifying itself.
            #
            # The fix was already sitting on the next line. `_work_fraction` has always been
            # max()-guarded against exactly this regression, so the internal state knew better
            # than the display for as long as both existed; the display simply did not read it.
            # Now it does, and monotonicity is a property of one guarded number rather than a
            # coincidence between two unguarded ones.
            self._work_fraction = max(self._work_fraction,
                                      done / float(self._estimated_frames))
            pct = 100.0 * self._work_fraction
        return self.phase(phase, pct=pct, frames_done=done,
                          frames_expected=self._estimated_frames, force=boundary)

    #: Share of a chunk's wall time each model phase takes, measured once — window 49 at 4K,
    #: `decisions.md` 4.40. Used only to interpolate *within* a chunk, so being roughly right is
    #: worth a great deal and being exactly right is worth almost nothing. Decode dominates, which
    #: is why a heartbeat weighted evenly across four phases would sit at 50% for most of the run.
    PHASE_SHARE = {"vae_encode": 0.25, "dit_sample": 0.16, "vae_decode": 0.52, "postprocess": 0.02}

    #: The order the phases run in, so the shares of completed phases can be summed.
    PHASE_ORDER = ("vae_encode", "dit_sample", "vae_decode", "postprocess")

    def _note_pass(self):
        """Time the gap between two passes. **The measurement the promise is made of.**"""
        now = time.time()
        if self._last_pass_at is not None:
            gap = now - self._last_pass_at
            # The most recent pass, not an average: passes lengthen through a job as the canvas
            # accumulates, and a mean over a whole run promises a cadence that has already
            # stopped being true.
            # **A floor and a ceiling, and both are about what a pass *is*.** Below a second is
            # not a news arrival — it is a synthetic harness, or two callbacks from one pass — and
            # promising a cadence measured from it would ask a client to poll at the floor for
            # nothing. Above an hour is not a cadence either; it is a stall, and the stall watch
            # is what handles those.
            if MIN_PASS_GAP_S <= gap < 3600:
                self._pass_cadence_s = gap
        self._last_pass_at = now

    def heartbeat(self):
        """Re-publish the standing payload with a fresh timestamp. **Nothing new to say, said.**

        The promise is a contract: *ask again in N seconds and there will be something to read*.
        During a counterless stretch — the drain, the strip — there is no new number, and letting
        the payload go stale breaks the promise as surely as saying nothing would. The B200
        replication showed the shape: an 11-minute payload silence against the worker's own 33 s
        promise, during a perfectly healthy tail. A watcher cannot tell that from a corpse, and
        the `at` field exists precisely so it does not have to guess.

        Returns True if it published. Cheap by construction: it re-mints what is already there.
        """
        if not self._enabled or not self.last:
            return False
        due = self.last.get("next_poll_s") or MIN_POLL_S
        if time.time() - self._last_emit < due:
            return False
        # **A fresh timestamp on the standing numbers, and nothing else invented.** The figures
        # are whatever the worker last knew; only `at` moves, which is exactly the distinction
        # `at` was added to make — a payload minted a minute ago is a chunk in flight, one minted
        # an hour ago is a corpse.
        refreshed = dict(self.last)
        refreshed["at"] = int(time.time())
        refreshed["elapsed_s"] = int(time.time() - self._started)
        refreshed["heartbeat"] = True
        self._emit(refreshed, force=True)
        return True

    @contextlib.contextmanager
    def keeping_the_promise(self):
        """Hold the payload fresh across a stretch with no callbacks at all.

        **The drain has no hook and is where the silence was measured.** Once the model has
        finished, the encoder drains inside ffmpeg with nothing calling back into this class —
        the B200 replication showed an 11-minute payload silence there against the worker's own
        33-second promise, on a run that was perfectly healthy. From outside, that is
        indistinguishable from a dead worker whose `/status` is still being served.

        A daemon thread, because there is nothing else to hang a timer on: the alternative is a
        callback the vendored encoder does not offer. It re-mints the standing payload and
        invents nothing, so the worst it can cost is one small POST per promised interval.
        """
        stop = threading.Event()

        def tick():
            while not stop.wait(max(1.0, MIN_POLL_S / 2.0)):
                try:
                    self.heartbeat()
                except Exception:  # noqa: BLE001 — a heartbeat must never fail a job
                    pass

        thread = threading.Thread(target=tick, name="cf-progress-heartbeat", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()

    def working(self, phase, index, total, chunk_frames=None):
        """A heartbeat from inside a chunk, one per model batch.

        **`frames()` fires when a chunk lands, and a chunk can be the whole clip.** Once
        `chunk_size` was allowed to hold the entire shot — which is the right setting, because a
        chunk boundary is the only unblended seam in the system — a 192-frame job produced exactly
        one progress update, at the end. Measured: 904 seconds of silence while `/status` answered
        `IN_PROGRESS` with a payload nobody had touched since the estimate. That is
        indistinguishable from a dead worker still serving a stale body, which is the failure the
        `at` and `attempt` fields exist to expose and which this one actually caused.

        **`frames_done` still does not advance here, and `frames_in_flight` is why it does not
        have to** (F-2026-08-19-29). Frames are not written until the chunk yields, so counting
        them as done would be a lie the ETA and the deadline guard both consume — `frames_done`
        stays a count of frames that exist. But the work on them is demonstrably happening, and
        publishing nothing about it left a 1,117 s job reporting `0/192` for 1,061 s: four phases
        of real work, invisible, because the only number anyone displayed could not move.

        So the heartbeat now carries three things a caller can act on: `pct`, interpolated across
        the phase shares; `frames_in_flight`, the frames currently inside the model, which is a
        fact rather than an estimate; and — through `_work_fraction` — an ETA that exists during
        the model stretch instead of arriving with the drain.
        """
        # **Time the gap first**, so the payload built below already carries a promise measured
        # from this pass rather than from the one before it.
        self._note_pass()
        done = float(self._frames_done)
        fraction = self._chunk_fraction(phase, index, total)
        # **The frames in THIS chunk, not the planned chunk size** (F-2026-08-20, display). The
        # caller passes `plan["chunk_size"]`, which is what a *full* chunk holds; the last one
        # holds the remainder. On the 222-frame run at chunk 185 that printed `185+185/222` —
        # 370 frames in a 222-frame clip — and it did worse than look wrong: `ahead` is
        # `in_flight × fraction`, so `done + ahead` saturated the moment chunk 2 opened and `pct`
        # sat at 100% through the whole of it.
        #
        # F-34's mirror, at the boundary it never faced. That one was `pct` going *backwards* at
        # the drain seam; this is `pct` arriving early at the chunk seam. Both come from mixing a
        # chunk-shaped number with a clip-shaped one, and both are fixed by asking what is
        # actually in hand rather than what a plan said would be.
        planned = int(chunk_frames or 0)
        remaining = (max(0, self._estimated_frames - self._frames_done)
                     if self._estimated_frames else planned)
        in_flight = min(planned, remaining) if planned else 0
        pct = None
        if self._estimated_frames:
            ahead = in_flight * fraction
            pct = 100.0 * min(done + ahead, self._estimated_frames) / float(self._estimated_frames)
            # **The clip's work, not the chunk's.** A chunk's own progress says nothing about a
            # job of seven chunks, and the ETA is about the job.
            self._work_fraction = min(1.0, (done + ahead) / float(self._estimated_frames))
        return self.phase("upscale", pct=pct, frames_done=self._frames_done,
                          frames_expected=self._estimated_frames,
                          frames_in_flight=in_flight,
                          model_phase=phase, batch=index, batch_of=total,
                          chunk_pct=int(100 * fraction))

    def _chunk_fraction(self, phase, index, total):
        """How far through one chunk the run is, weighted by what each phase actually costs."""
        share_done = 0.0
        for name in self.PHASE_ORDER:
            if name == phase:
                break
            share_done += self.PHASE_SHARE.get(name, 0.0)
        within = (index / float(total)) if total else 0.0
        return min(1.0, share_done + self.PHASE_SHARE.get(phase, 0.0) * within)

    def begin_phase(self):
        """Reset the rate clock. Called when real frame work starts, so model load does not get
        amortised into the per-frame figure and flatter the ETA."""
        self._phase_started = time.time()

    def eta_s(self):
        """Seconds remaining, or None — **and never the optimistic one of two honest answers.**

        Three sources, in order of how much they know:

        *Measured.* A chunk has landed, so the real rate on the real card at the real size is in
        hand. This module's standing argument is that nothing beats it, and once it exists it is
        the answer.

        *Predicted.* The planner's own per-frame figure, seeded by `expect()`, priced over the
        whole clip and **decayed by the work fraction already done**. Available from the first
        heartbeat, which is what lets an ETA exist while the model is working and no frame has
        been written — and, since F-33, one that falls as that work proceeds rather than standing
        still beside a `pct` that climbs.

        *Observed.* Elapsed against the fraction of the clip's work completed, phase-weighted.
        Self-correcting and needs no table at all, but the shares are a model and a model can be
        wrong in the direction that flatters.

        **Where two estimates are available and neither is measured, the larger is published.**
        That is the no-optimism rule stated arithmetically: an ETA that runs short tells CF a job
        is nearly done when it is not, and CF's poll loop, its stale-request sweep and whatever an
        app shows a user all key off this number. `None` rather than a guess still holds when
        there is nothing at all to compute from.

        **What the rule never licensed** (CF, amending F-29's second design call at F-33): it
        stops this number guessing low; it does not permit ignoring the work the run has visibly
        done. An ETA and a `pct` minted in the same payload may not imply different completion
        fractions — that is not pessimism, it is one payload contradicting itself.
        """
        if not self._estimated_frames:
            return None
        if self._seconds_per_frame is not None:
            self._eta_basis = "measured"
            remaining = max(0, self._estimated_frames - self._frames_done)
            return remaining * self._seconds_per_frame

        candidates = []
        if self._expected_seconds_per_frame is not None:
            # **The seed decays against the same work fraction `pct` is built from** (F-33). It
            # used to be `remaining_frames x plan price`, and `frames_done` cannot move mid-chunk
            # by design — so on a whole-clip chunk the seeded candidate was a constant, the
            # `max()` published that constant for the entire run, and clip 1 reported
            # `78% ... eta 491 min` from one payload: three-quarters done and all of the time
            # left. Two numbers minted together, disagreeing about the same job.
            #
            # `_work_fraction` is the clip's phase-weighted progress and is exactly what `pct`
            # publishes, so scaling the full-job price by what remains of it keeps the two
            # arithmetically consistent by construction rather than by coincidence.
            #
            # **This does not repeal the no-optimism rule** (F-29 design call 2, as CF amended
            # it): the rule stops the ETA guessing *low*, and it never licensed ignoring measured
            # work. The `max()` below is untouched, so a run genuinely slower than its plan still
            # has its ETA pushed up by the observed candidate — which is the honest way for this
            # number to grow, since it is the one derived from this run's own wall clock.
            #
            # The seed's own accuracy is a separate matter and deliberately untouched here: the
            # 8K time seed runs ~5.7x hot, which is a §8b calibration question, not an arithmetic
            # one. A hot seed now decays from a hot start instead of standing still at it.
            full_price = self._estimated_frames * self._expected_seconds_per_frame
            candidates.append((full_price * max(0.0, 1.0 - self._work_fraction),
                               self._expected_basis))
        # A fraction this small is noise — one batch of a long chunk — and dividing by it turns
        # a plausible reading into a wild one.
        if self._work_fraction >= 0.02:
            spent = time.time() - self._phase_started
            candidates.append((spent * (1.0 - self._work_fraction) / self._work_fraction,
                               "observed"))
        if not candidates:
            self._eta_basis = None
            return None
        seconds, basis = max(candidates)
        self._eta_basis = basis
        return seconds

    def _next_poll_s(self, eta):
        """**When the next news arrives — not how much work is left** (F-2026-08-20, cadence).

        This asked for a tenth of the remaining time, which is a statement about the *end of the
        job* offered in answer to a question about the *next update*. The two are unrelated, and
        the corpus shows both failure directions in one run: during 8K decode, news arrived every
        ~240 s and the client was told to poll far more often, wasting requests; during
        postprocess, six passes completed in ~82 s and the promise was still minutes, so five of
        six payloads were overwritten before anyone looked. `/status` is a mailbox, not a stream
        — the latest post overwrites — so under-sampling loses payloads irrecoverably.

        The worker is the only party that knows the answer, and it knows it by measurement: it
        has just finished a pass and timed it. So the promise is the observed pass cadence, which
        makes it true by construction rather than by inference.

        The ETA survives as the fallback for the stretches with no passes to time — the fetch,
        the load strip, the drain — where a tenth of what is left is at least *a* number and the
        alternative is no promise at all.
        """
        if self._pass_cadence_s:
            # Half a pass: a client polling at this rate sees every payload at least once, which
            # is the property a mailbox needs. Clamped like everything else, because a cadence
            # measured from one very long pass is still a promise someone has to keep.
            return int(max(MIN_POLL_S, min(MAX_POLL_S, self._pass_cadence_s / 2.0)))
        if eta is None:
            return MIN_POLL_S
        return int(max(MIN_POLL_S, min(MAX_POLL_S, eta / 10.0)))

    def seconds_per_frame(self):
        return self._seconds_per_frame

    def elapsed_s(self):
        return time.time() - self._started

    def _emit(self, payload, force=False):
        # `last` is updated whichever way this goes: the rate limiter decides what is *sent*,
        # never what the worker knows.
        self.last = payload
        now = time.time()
        if not force and (now - self._last_emit) < MIN_INTERVAL_S:
            return
        self._last_emit = now
        self.emitted.append(payload)
        if not (self._enabled and self._job is not None):
            return
        try:
            import runpod
            runpod.serverless.progress_update(self._job, payload)
        except Exception:  # noqa: BLE001
            # An oversized or failed progress update cannot fail a job — the POST runs on a
            # daemon thread and its error is swallowed by the SDK. Swallowing it here too keeps
            # that true when the SDK is absent, which is how rung 1 runs.
            pass
