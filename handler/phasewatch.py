"""Which of the four phases is running, and what each one costs in VRAM.

**The vendored code has always known and always thrown it away.** Every phase opens with a banner
and every phase failure logs its own name before re-raising:

    ━━━━━━━━ Phase 3: VAE decoding ━━━━━━━━
    Error in Phase 3 (Decoding): CUDA out of memory. Tried to allocate 4.62 GiB ...

then `raise` — bare. By the time the exception reaches this worker it is an anonymous
`OutOfMemoryError`, and the single most actionable fact about the failure is gone. That is why the
ratchet steps a whole rung: it cannot tell a decode that needed a smaller tile from a DiT pass that
needed a smaller window, so it changes both and three other things besides.

**This costs nothing to fix.** Those lines are emitted with `force=True`, and `force` bypasses the
`enabled` check in `Debug.log` — they are written even with debug off, which is how this worker
runs in production. `inference_cli.debug` is a module-level singleton that this worker already
passes into `_stream_video_chunks`, so wrapping its `log` is a two-line install and needs no
vendored patch, no log file, no second thread and no stdout parsing.

**Per-phase peaks come from the same hook.** A banner is a phase boundary, so the peak counter is
read and reset there. Resetting is safe because nothing else reads that counter mid-attempt, and
the attempt-level figure is recovered exactly as `max(peaks)` — see `peak_gb`. What was one number
per run becomes four, which is the difference between a memory sweep that costs one run per point
and one that costs one run per four points.

**The phases repeat.** `_stream_video_chunks` runs all four for every chunk, so a phase is entered
once per chunk and the recorded figure is the maximum across them — the peak is what has to fit.
"""

import re
import time

#: Phase number to the name this worker uses. Deliberately not the vendored wording: these names
#: reach CF in a shortfall, and "upscale" beside a worker whose whole job is upscaling reads as the
#: job rather than the phase.
PHASES = {
    1: "vae_encode",
    2: "dit_sample",
    3: "vae_decode",
    4: "postprocess",
}

#: The lever that phase's failure actually implicates. Not consulted here — this module only
#: observes — but recorded beside the phase so the reader of a shortfall does not have to know the
#: vendored architecture to act on it.
LEVERS = {
    "vae_encode": ("vae_encode_tiled", "vae_encode_tile_size"),
    "dit_sample": ("batch_size", "blocks_to_swap", "swap_io_components"),
    "vae_decode": ("vae_decode_tiled", "vae_decode_tile_size"),
    "postprocess": ("tensor_offload_device", "chunk_size"),
}


#: `Encoding batch 3/10`, `Upscaling batch 3/10`, `Decoding batch 3/10`,
#: `Post-processing batch 3/10` — one per model batch, all four force-logged, so they arrive with
#: debug off exactly like the phase banners do.
_BATCH = re.compile(r"batch (\d+)\s*/\s*(\d+)")


def _batch_from(message):
    """The (index, total) a per-batch log line announces, or None."""
    found = _BATCH.search(message)
    if not found:
        return None
    return int(found.group(1)), int(found.group(2))


def _phase_from(message):
    """The phase a log line announces, or None.

    Matches the banner (`Phase 3: VAE decoding`) and the failure line (`Error in Phase 3
    (Decoding):`) with one rule, because both carry `Phase <n>` and nothing else force-logged
    does. Matching the digit rather than the wording is deliberate: the words are cosmetic and
    have already changed once upstream, the numbering is structural.
    """
    at = message.find("Phase ")
    if at < 0:
        return None
    digits = message[at + 6:at + 8].strip()
    if not digits[:1].isdigit():
        return None
    return PHASES.get(int(digits[0]))


def host_rss_gb():
    """The container's own resident set, in GiB, or `None` where /proc does not say.

    **`VmRSS` from `/proc/self/status`, read rather than modelled.** Host RAM is the wall that
    kills without an exception: a breach is a cgroup SIGKILL, which writes no bundle, raises
    nothing and offers no walk — so unlike VRAM there is no failure path that reports the number
    afterwards. It has to be sampled while the process is alive or it is never known at all.

    Linux-only by construction. On a laptop this returns `None` and the banners stay silent,
    which is the right behaviour for a figure that means nothing off the container.
    """
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001 — an unreadable /proc is a missing number, not a failure
        return None
    return None


def host_hwm_gb():
    """`VmHWM` — the highest resident set this process has *ever* reached, in GiB.

    **The instrument the tail actually needs.** A point sample of `VmRSS` cannot see the drain
    peak from either side of it: taken before the encoder finishes it is too early, taken after
    it is too late, and the peak sits in between where nothing is looking. `VmHWM` is the
    kernel's own high-water mark and is monotone for the life of the process, so a reading taken
    *after* the drain still reports what the drain reached.

    This is what the registry's provisional tail term (~5.3 B/px/frame, one anchor) is waiting
    to be fitted from, and it is why readings taken with the old point sample cannot be banked
    as drain peaks.
    """
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001 — an unreadable /proc is a missing number, not a failure
        return None
    return None


def host_total_gb():
    """The RAM this container may actually use — **the ceiling, not the machine's total**.

    This used to read `SC_PHYS_PAGES`, which is the physical host and was the third independent
    place doing so (F-2026-08-19-37). It feeds the `of X GiB (Y% peak)` figure on every `[host]`
    banner, so a sliced worker was reporting its tail as a comfortable fraction of a number it
    could never reach: 107.20 GiB of "3019.4" read as 4%, where against a real 377 GiB slice it
    is 28%. Same measurement, entirely different meaning.
    """
    try:
        import hardware  # noqa: PLC0415 — stdlib-only; the one choke point for this number
        return hardware.effective_ram_gb()
    except Exception:  # noqa: BLE001
        return None


def cpu_count():
    """Cores this container may actually use. **Delegates to `hardware.cpu_count`.**

    The reasoning and the reading both moved there on 2026-08-28, when `hardware.read()` began
    reporting the core counts per JOB rather than per attempt: `hardware` is the module this one
    already imports lazily to keep the cycle absent, so it is the only one of the two that can
    hold the single copy. This name stays because the boot banner and the per-attempt block both
    call it, and a second spelling of one number is how the pair drifts.
    """
    import hardware  # noqa: PLC0415 — stdlib-only; the one choke point for this number

    return hardware.cpu_count()


#: The environment variable the image bakes its commit into. Named here rather than spelled into
#: the format string so this banner and `handler.build_identity` cannot come to read different
#: names — the rung-1 witness asserts the two agree, and this constant is what makes that cheap.
BUILD_COMMIT_ENV = "BUILD_COMMIT"


#: Every `[host]` reading this job took, in order, wherever it was taken from. **The corpus, and
#: it used to be a keyhole** (CF, 2026-08-20): the run-record kept only the four readings
#: `handler` banked by hand — load, load-end, tail-in, tail — while `PhaseWatch` printed one at
#: every phase boundary and threw them away. The whole analysis that found the ~30 GiB
#: misassigned constant had to be inferred through that gap, because the sample that would have
#: settled it directly — the host *after* the first model phase, when file-backed weight pages
#: are materialised — was printed on a worker that no longer exists and recorded nowhere.
#:
#: It lives here rather than in `handler` because this is the module that takes the readings, and
#: the previous arrangement put the list one import away from the code that produced most of the
#: entries. Reset per job by `handler.handle`.
BANNERS = []


def observe(label, rss, peak=None, total=None, frames_fed=None):
    """Bank one host reading into the corpus, with the killer's number beside ours.

    Never raises: this is instrumentation on a path that has already spent GPU minutes, and a
    measurement that can fail a job is worse than no measurement. Same posture as the run-record
    and the progress emitter.
    """
    try:
        import hardware  # noqa: PLC0415 — stdlib-only; local to keep the import cycle absent

        current = hardware.memory_current_gb()
        parts = hardware.memory_breakdown_gb()
        BANNERS.append({
            "at": label,
            "rss_gb": None if rss is None else round(rss, 2),
            "peak_gb": None if peak is None else round(peak, 2),
            "total_gb": None if not total else round(total, 1),
            # **The split, on every sample and not only the four hand-banked ones.** It is what
            # separates weights that are materialised and pinned from cache the kernel could
            # drop — which is precisely the ~30 GiB's classification question, and it has to ride
            # the *same instant* as the RSS it disagrees with or the two cannot be compared.
            "cgroup_current_gb": None if current is None else round(current, 2),
            "cgroup_anon_gb": None if "anon" not in parts else round(parts["anon"], 2),
            "cgroup_file_gb": None if "file" not in parts else round(parts["file"], 2),
            "cgroup_slab_gb": None if "slab" not in parts else round(parts["slab"], 2),
            "frames_fed": frames_fed,
        })
    except Exception:  # noqa: BLE001 — see the docstring
        pass


#: **The one exception an observer must never swallow** (F-2026-08-20-46). This module's whole
#: posture is that watching must not break the run it watches, and two `except Exception: pass`
#: blocks enforced it — the wrapper around the vendored logger, and the per-batch dispatch inside
#: `_observe`. Then a *deliberate refusal* started travelling that path: the host guard raises
#: `WorkerError` at a pass boundary to stop a doomed run, and both blocks ate it. Twice,
#: deterministically, the verdict printed in full and the very next line was `Upscaling batch
#: 2/5`; the run went on to completion inside a container the kernel was about to kill.
#:
#: A refusal is not an observation failure. It is the point of observing. So it passes, and
#: everything else is still swallowed exactly as before.
def _is_a_refusal(exc):
    try:
        import errors  # noqa: PLC0415 — stdlib-only, and only reached on the failure path

        return isinstance(exc, errors.WorkerError)
    except Exception:  # noqa: BLE001 — if even this fails, keep the old posture
        return False


def _affinity_cores():
    """Cores in this process's mask. **Delegates to `hardware._affinity_cores`** — see
    `cpu_count` above for why the pair lives there now."""
    import hardware  # noqa: PLC0415 — stdlib-only

    return hardware._affinity_cores()


def cpu_configuration():
    """What the container was allowed to compute with, for the record (F-2026-08-20-44).

    **Instrument-first, and the amendment says so outright**: no coefficient is fitted from this
    until the corpus shows a shape, because one 36-vCPU data point is not a model. What is known
    is that the tail runs single-core for its encode segment while the assembly before it is
    parallel, so a tail time without a core count is a measurement of an unnamed machine.
    """
    import hardware  # noqa: PLC0415 — stdlib-only

    return {
        "usable_cores": cpu_count(),
        "cpu_quota": hardware.cpu_quota(),
        "affinity_cores": _affinity_cores(),
    }


def cgroup_note():
    """`   cgroup 39.54 (anon 28.30 / file 11.21)` — the killer's number beside our own.

    **Appended to every `[host]` line rather than printed on its own**, because the finding this
    exists to settle is a *disagreement between two readings of the same instant*, and two
    numbers on two lines minutes apart cannot be compared. On the F-41 staircase the platform
    read 39.5 GiB where our banner read 28.3; whether that 11.2 GiB is reclaimable page cache or
    memory that actually kills is the difference between under-chunking and dying, and no row in
    the corpus can answer it because no row has both figures.

    Silent where there is no cgroup — a laptop, a rung-1 run — for the same reason every other
    reading here is: an instrument that shouts about its own absence trains people to skip the
    line it is attached to.
    """
    import hardware  # noqa: PLC0415 — stdlib-only module, imported here to keep the cycle absent

    current = hardware.memory_current_gb()
    if current is None:
        return ""
    parts = hardware.memory_breakdown_gb()
    inside = "  ".join("{} {:.2f}".format(name, parts[name])
                       for name in ("anon", "file", "slab") if name in parts)
    return "   cgroup {:6.2f}{}".format(current, "" if not inside else " ({})".format(inside))


def build_banner():
    """Which build is running, in the worker's own log.

    **The image has always known this and the log has never said it.** `BUILD_COMMIT` is baked
    into every image's config by CI (`Dockerfile:223`, `docker-publish.yml:282`) and the gate
    reads it back off the registry blob on every verification — but that answers *what did we
    publish*, from outside. A worker log answers *what am I*, from inside, and until now it could
    not: a log pulled off a running endpoint named the host, the slice and the data centre, and
    left unstated the one fact that decides whether any of the others are worth reading. Ten
    calibration runs were once banked against an image reporting `"image": null`, and the only
    evidence they shared a build was that nobody remembered changing one.

    Beside the host lines deliberately: one glance at any worker log now names the build, the
    host slice and the DC together, which is the set a measurement has to be sorted by.

    **Absent is a supported state and says which name was tried** — the same convention the data
    centre line above already uses. A handler run locally has no build identity and must not
    crash for lacking one, and "the variable was never set" must not read identically to "we read
    the wrong name" for whoever finds the log later.
    """
    import os as _os  # noqa: PLC0415 — local, matching every other stdlib touch here

    commit = _os.environ.get(BUILD_COMMIT_ENV)
    if not commit:
        return ("[host] boot: build unknown — {} is not set. A CI image always carries it, so "
                "this is a local or hand-built one.".format(BUILD_COMMIT_ENV))
    # **The full sha as well as the short form.** This line is read beside a registry digest and
    # a `docs/deployment.md` lineage row, where an abbreviation is one collision away from naming
    # a different commit; the short form leads because that is the one a person types.
    return "[host] boot: build {} (sha-{})".format(commit[:7], commit)


def boot_banner():
    """The one-line CPU, host-RAM and data-centre statement every run should open with.

    **The data centre is on this line because the `[load]` strip is meaningless without it.** Two
    H200 data centres behaved differently on the same image on the same day — one could not pull
    from GHCR, another streams layers lazily and pays 6 to 12 minutes faulting the checkpoint in
    on a fresh worker. Every log now carries the axis those figures have to be sorted by, and an
    absent one says *which* names were tried, so "the platform stopped exposing it" and "we read
    the wrong key" stop looking identical.
    """
    import hardware  # noqa: PLC0415 — stdlib-only module; imported here to keep the cycle absent

    cores, total = cpu_count(), host_total_gb()
    dc, source = hardware.datacenter()
    physical = hardware.physical_ram_gb()
    limit = hardware.memory_limit_gb()
    # **Both numbers, always, so a sliced host is visible in every log** (F-2026-08-19-37). The
    # ceiling alone would be indistinguishable from a small machine, and the physical alone is
    # what this worker planned against for its whole life. Printed as a pair, the slice is a fact
    # anybody reading a log can see without knowing to go looking for it.
    return "[host] boot: {} core(s) usable, {} host RAM, {} resident, dc {}".format(
        cores if cores is not None else "?",
        "{:.1f} GiB".format(total) if total else "unknown",
        "{:.2f} GiB".format(host_rss_gb() or 0.0) if host_rss_gb() is not None else "unknown",
        "{} (from {})".format(dc, source) if dc
        else "not exposed — tried {}".format(", ".join(hardware.DATACENTER_ENV))) + (
        "\n[host] boot: cgroup slice {:.1f} GiB of {:.1f} GiB physical — the slice is what "
        "kills".format(limit, physical) if limit and physical
        else "\n[host] boot: no cgroup memory limit; the machine's {} is the ceiling".format(
            "{:.1f} GiB".format(physical) if physical else "unknown RAM")) + (
        "\n" + build_banner())


def _torch_peak_gb():
    """Peak bytes allocated since the last reset, in GB, or None off-GPU.

    Imported inside the function like every other torch touch in this package: `handler` must
    import with the model chain still lazy or the rung-1 contract suite stops running in CI.
    """
    try:
        import torch  # noqa: PLC0415 — deliberate lazy heavy import
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    except Exception:  # noqa: BLE001 — a counter we cannot read is not worth failing a job over
        return None


def _torch_reset_peak():
    """Zero the peak counter **and hand back everything the allocator is not using.**

    §5.1 asks for `empty_cache()` between phases, not only before a retry, and a phase boundary is
    the one moment in a run where that is nearly free: the phase that just ended has released its
    activations, the next one has not allocated yet, and whatever the allocator is still holding
    in cached blocks is fragmentation waiting to happen. Freeing it here is what keeps the
    `reserved - allocated` gap small enough that, when an OOM does arrive, the gap in its message
    means something.

    It costs a synchronise and the next phase re-acquiring its segments -- real, and small against
    a decode measured in minutes.
    """
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
    except Exception:  # noqa: BLE001
        pass


def _torch_reserved_gb():
    """Peak *reserved* bytes, and the allocator gap, in GB. `(reserved, gap)` or `(None, None)`.

    **Reserved minus allocated is the fragmentation number**, and it is the one figure that
    separates "this configuration does not fit" from "the allocator could not find a contiguous
    block for it". The OOM message carries the same quantity as `reserved but unallocated`, but
    only on the attempt that dies; reading it at every phase boundary means a *successful* run
    also reports how close to fragmented it was, which is what turns fragmentation from an
    after-the-fact excuse into something with a trend.
    """
    try:
        import torch  # noqa: PLC0415 — deliberate lazy heavy import
        if not torch.cuda.is_available():
            return None, None
        reserved = torch.cuda.max_memory_reserved(0) / (1024 ** 3)
        allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        return reserved, max(0.0, reserved - allocated)
    except Exception:  # noqa: BLE001 — a counter we cannot read is not worth failing a job over
        return None, None


class PhaseWatch(object):
    """Installs the tap for the duration of a `with` block and collects what it sees.

    `read_peak` and `reset_peak` are injected so the whole class is testable with no GPU and no
    torch — the tests drive it with a fake `Debug` and a counter they control, which is the only
    way to assert that a *specific* phase recorded a *specific* figure.

    **The wrapper must never break logging.** Every failure inside the bookkeeping is swallowed:
    an exception raised from a log call would surface as a model failure, at a point where the
    real exception is often already in flight, and the resulting traceback would blame this module
    for a decode OOM. Losing a phase label is a bad outcome; losing the job is worse.
    """

    def __init__(self, debug_holder, read_peak=None, reset_peak=None, on_batch=None,
                 announce=True, read_reserved=None):
        #: The object whose `.log` is wrapped — `inference_cli` itself in production, whose
        #: module-level `debug` is the singleton every vendored phase writes through.
        self._holder = debug_holder
        self._read = _torch_peak_gb if read_peak is None else read_peak
        self._reset = _torch_reset_peak if reset_peak is None else reset_peak
        #: `on_batch(phase, index, total)`, called once per model batch. **This is the only
        #: heartbeat a job has once the chunk holds the whole clip** — `on_chunk` then fires once,
        #: at the end. Optional, because the tap must stay useful to anything that only wants the
        #: peaks.
        self._on_batch = on_batch
        self._read_reserved = _torch_reserved_gb if read_reserved is None else read_reserved
        #: **Write the readings back into the worker's own log.** Everything this class records
        #: has until now reached only the JSON the handler returns, which means the person
        #: watching the RunPod log — the one who can see the run happening — was the only party
        #: without the numbers. Emitted through the *unwrapped* logger with `force=True`, so it
        #: arrives with debug off and cannot recurse back through `_observe`.
        self._announce = announce
        self._original = None
        self._debug = None
        self._was_instance_attr = False

        #: Phase currently running, by the time anything asks.
        self.phase = None
        #: Phase named by an `Error in Phase N` line. **Stronger than `phase`**: it is written by
        #: the vendored handler for the phase that actually raised, where `phase` is only the last
        #: one entered and would name phase 4 if a failure arrived between phases.
        self.failed_in = None
        #: name -> peak GB, the maximum over every chunk that ran that phase.
        self.peaks = {}
        #: **Why the tap produced nothing, when it produces nothing.** The first GPU run carrying
        #: this module returned no per-phase peaks at all, and the log proved the banners were
        #: there and the heartbeat was firing -- so the wrapper installed and `_observe` ran, and
        #: the peak *read* is what came back empty. Nothing recorded which, because every failure
        #: in here is swallowed by design. So the swallowing now leaves a note.
        self.diagnosis = {"installed": False, "banners": 0, "reads": 0, "read_failures": 0,
                          "last_error": None}
        #: name -> how many times the phase was entered. A phase count that is not the chunk count
        #: means the run did not do what the schedule said it would.
        self.entered = {}
        #: name -> seconds spent in it, summed across every entry (F-2026-08-20-44). **The
        #: contract promised these from the start** — F-36 lists "per-phase and per-frame times"
        #: among the record's contents — and every boundary was stamped and the duration dropped.
        #: What that cost is now measured: the B200 pair ran identical warm jobs whose phases
        #: matched to a couple of seconds and whose *tails* differed 477 s against ~1340 s, and
        #: the whole 17.9-to-21.7 s/frame spread was that tail. Without per-phase times the
        #: spread had nowhere to be attributed and was read as host variance.
        self.durations = {}
        self._phase_opened = None
        #: name -> peak *reserved* GB, and name -> the allocator gap at that peak. The gap is what
        #: §5.1 of the planner reads to tell a fragmentation OOM from a real one, and recording it
        #: on successes too is what gives that threshold something to be calibrated against.
        self.reserved = {}
        self.gaps = {}
        #: label -> peak host RSS in GiB observed at that boundary, including the tail. The tail
        #: entry is the one the registry's host term is waiting on.
        self.host_rss = {}

    # ── installation ────────────────────────────────────────────────────────────────────────
    def __enter__(self):
        self._debug = getattr(self._holder, "debug", None)
        if self._debug is None or not hasattr(self._debug, "log"):
            # Nothing to tap. Not an error: the contract suite runs with no vendored module at
            # all, and a watch that quietly observes nothing is better than an import-time
            # dependency on the GPU box's layout.
            return self
        self._original = self._debug.log
        # **Whether `log` was an instance attribute decides how to put it back.** It is normally a
        # bound method resolved from the class, so assigning the saved value on the way out would
        # leave a permanent instance-level shadow of the class method -- harmless until something
        # subclasses `Debug`, and invisible when it stops being harmless.
        self._was_instance_attr = "log" in vars(self._debug)
        original = self._original

        def log(message, *args, **kwargs):
            try:
                self._observe(message)
            except Exception as exc:  # noqa: BLE001 — see the class docstring
                # **Except a refusal.** See `_is_a_refusal`: the guard's verdict travels this
                # path, and swallowing it here let a doomed run continue to completion.
                if _is_a_refusal(exc):
                    raise
            return original(message, *args, **kwargs)

        self._debug.log = log
        self._reset()
        self.diagnosis["installed"] = True
        return self

    def __exit__(self, exc_type, exc, tb):
        # Close the phase that was running, so a failing run still reports what the phase that
        # died had reached. Without this the most interesting figure in the whole run is the one
        # that never gets recorded.
        try:
            self._close_current()
            # Said while the logger is still ours to write through, and said on the way out of a
            # *failed* attempt too -- that is the attempt whose numbers matter most.
            self._say(self.summarise())
        except Exception:  # noqa: BLE001
            pass
        if self._debug is not None and self._original is not None:
            if self._was_instance_attr:
                self._debug.log = self._original
            else:
                try:
                    del self._debug.log
                except AttributeError:      # someone replaced it underneath us
                    self._debug.log = self._original
        return False

    # ── observation ─────────────────────────────────────────────────────────────────────────
    def _observe(self, message):
        text = message if isinstance(message, str) else str(message)
        # Per-batch lines carry no phase number, so they are matched first and attributed to
        # whichever phase is currently open.
        batch = _batch_from(text)
        if batch is not None and self._on_batch is not None and self.phase is not None:
            try:
                self._on_batch(self.phase, batch[0], batch[1])
            except Exception as exc:  # noqa: BLE001 — see the class docstring
                # **The inner half of the same seam** (F-2026-08-20-46). Both blocks had to let
                # a refusal through: fixing one alone leaves the other holding the verdict.
                if _is_a_refusal(exc):
                    raise
            return

        name = _phase_from(text)
        if name is None:
            return
        if text.startswith("Error in Phase") or "Error in Phase" in text:
            # A failure line names the phase that raised and does *not* start a new one.
            self.failed_in = name
            return
        self._close_current()
        self.phase = name
        self._phase_opened = time.time()
        self.diagnosis["banners"] += 1
        self.entered[name] = self.entered.get(name, 0) + 1
        self._reset()

    def _close_current(self):
        if self.phase is None:
            return
        # **Stamped before anything that can fail.** Every read below is in a try; a duration
        # lost because a memory counter was unreadable would be the same defect this closes.
        if self._phase_opened is not None:
            self.durations[self.phase] = round(
                self.durations.get(self.phase, 0.0) + (time.time() - self._phase_opened), 1)
            self._phase_opened = None
        self.diagnosis["reads"] += 1
        try:
            peak = self._read()
        except Exception as exc:  # noqa: BLE001 — recorded rather than swallowed silently
            self.diagnosis["read_failures"] += 1
            self.diagnosis["last_error"] = "{}: {}".format(type(exc).__name__, exc)
            return
        if peak is None:
            self.diagnosis["read_failures"] += 1
            self.diagnosis["last_error"] = self.diagnosis["last_error"] or "read returned None"
            return
        previous = self.peaks.get(self.phase)
        self.peaks[self.phase] = peak if previous is None else max(previous, peak)

        reserved, gap = self._read_reserved()
        if reserved is not None:
            self.reserved[self.phase] = max(self.reserved.get(self.phase, 0.0), reserved)
            self.gaps[self.phase] = max(self.gaps.get(self.phase, 0.0), gap or 0.0)
        self._say("[mem] {:<12} peak {:6.2f} GB{}".format(
            self.phase, peak,
            "" if reserved is None else "   reserved {:6.2f}   gap {:5.2f}".format(reserved, gap)))
        self._say_host(self.phase)

    def _say_host(self, label):
        """The `[host]` banner for a phase boundary — VmRSS, and the share of the host it is.

        **F-2026-08-18-7, and it is load-bearing rather than a nicety.** The `[mem]` banners
        cover VRAM only, so host RSS during a tail was observable solely by watching the RunPod
        console, which the operator cannot always do — and the tail is where the container gets
        killed. Every run now measures the tail term for free, which is what turns the planner's
        provisional ~5.3 B/px/frame into a fitted registry line at the next re-key.
        """
        rss = host_rss_gb()
        if rss is None:
            return
        self.host_rss[label] = max(self.host_rss.get(label, 0.0), rss)
        total = host_total_gb()
        # **Banked, not only printed** — see `BANNERS`. The reading taken as the first model
        # phase opens is the post-materialisation floor CF's fit needs, and it was being written
        # to a log and discarded.
        observe(label, rss, peak=host_hwm_gb(), total=total)
        self._say("[host] {:<12} rss  {:6.2f} GiB{}{}".format(
            label, rss,
            "" if not total else "   of {:.1f}   ({:.0%})".format(total, rss / total),
            cgroup_note()))

    def _say(self, message):
        """Emit into the worker's own log, through the logger we wrapped.

        The **unwrapped** callable, deliberately: going through our own replacement would re-enter
        `_observe` on every line we write, and a line carrying the word "Phase" would then be read
        as a phase banner. `force=True` because production runs with debug off and these numbers
        are the whole point of the run.
        """
        if not self._announce or self._original is None:
            return
        try:
            self._original(message, category="vae", force=True)
        except TypeError:
            # A logger that does not take our keywords still takes the text.
            try:
                self._original(message)
            except Exception:  # noqa: BLE001 — see the class docstring
                pass
        except Exception:  # noqa: BLE001
            pass

    def summarise(self):
        """One line carrying the whole attempt: the peak, the ceiling, and every phase.

        Written at the end of the attempt rather than assembled by the reader, because the four
        per-phase lines are separated by minutes of decode in the log and nobody scrolls back.
        """
        if not self.peaks:
            return "[mem] no per-phase readings: {}".format(self.diagnosis)
        parts = " ".join("{} {:.2f}".format(name, gb)
                         for name, gb in sorted(self.peaks.items(), key=lambda kv: -kv[1]))
        worst_gap = max(self.gaps.values()) if self.gaps else None
        return "[mem] attempt peak {:.2f} GB, ceiling {}{} | {}".format(
            self.peak_gb, self.ceiling,
            "" if worst_gap is None else ", worst allocator gap {:.2f} GB".format(worst_gap),
            parts)

    # ── what it is for ──────────────────────────────────────────────────────────────────────
    @property
    def peak_gb(self):
        """The attempt's peak, recovered from the per-phase figures.

        Resetting the counter at each boundary is what makes the per-phase numbers real, and it
        would otherwise destroy the whole-attempt figure the calibration table has always banked.
        It does not: phases are sequential and the allocator is emptied between them, so the
        largest phase peak *is* the attempt peak.
        """
        return max(self.peaks.values()) if self.peaks else None

    @property
    def ceiling(self):
        """The phase that set the peak — the one a bigger card would be bought for."""
        if not self.peaks:
            return None
        return max(self.peaks, key=lambda name: self.peaks[name])

    def blame(self):
        """What to record about a failure: the phase, how sure we are, and the levers it names.

        `confidence` is not decoration. `named` means the vendored code told us which phase raised;
        `last_entered` means we are inferring it from the last banner, which is right in every
        case observed so far and wrong for a failure arriving between phases. A caller stepping a
        lever on this should know which it has.
        """
        name = self.failed_in or self.phase
        if name is None:
            return None
        return {
            "phase": name,
            "confidence": "named" if self.failed_in else "last_entered",
            "levers": list(LEVERS.get(name, ())),
            "phase_peaks_gb": {k: round(v, 2) for k, v in self.peaks.items()},
        }
