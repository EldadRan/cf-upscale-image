"""The interpolator shim: a frame stream in, a longer frame stream out.

**A stream transformer and nothing else.** It is not a pipeline stage, it owns no I/O, it opens
no file and it decides no policy. It consumes frames in order, holds exactly one frame of
lookahead because RIFE synthesises from the pair `(i, i+1)`, and emits the frames the plan calls
for. Wiring it to anything is somebody else's step.

Spec: `release-3-contract.md` §5 (the frame plan, the tail hold, the snap tolerance), §5b (the
float defect), §9 (what is portable from the reference script) and §9b (what is forbidden).
`fable/retime_oracle.py` is the same contract executable and is the authority: **if this file and
that one disagree, one of them is a bug and it gets a decision entry rather than a patch.** The
arithmetic here is written from the contract rather than copied from the oracle, because two
independent statements of one rule is the point — a plan that agreed with the oracle by sharing
its code would prove nothing.

**The plan half is stdlib-only and torch is imported lazily.** `Interpolator` needs a model and a
GPU; `build_plan` needs a frame count and two rates. Keeping the second free of the first is what
lets the acceptance cases run this file from a cold tree with no torch, no cv2 and no numpy, which
is where they run.
"""
from fractions import Fraction

#: Repairs binary floating point, and **is not the snap tolerance**. `k * src/dst` lands a hair
#: either side of a whole number; EPS decides that such a position IS the whole number. `tol`
#: changes policy — which frames are worth synthesising — and the two must never be conflated.
EPS = 1e-9

#: The padding contract (§9). Each dimension is padded up to a multiple of this, and the result
#: is cropped back to `[:h, :w]`. `128 / scale` because a model that upscales by `scale` needs its
#: INPUT aligned to the multiple its OUTPUT will be; `max` because no model wants less than 128.
PAD_BASE = 128


def target_count(n_in, src_fps, dst_fps):
    """How many output frames. **Half-up, in exact arithmetic, and both halves matter.**

    `n_in * dst / src` lands exactly on `.5` for every odd clip length at 24->60 — half of all
    inputs — so the tie is real and frequent rather than a corner. Python's `round()` ties to
    EVEN (302.5 -> 302, 303.5 -> 304) while C, JavaScript and Java round half away from zero, so
    a contract that leaves the tie unstated is one a reimplementation silently disagrees with on
    half the clips it sees.

    `Fraction` rather than float because the tie has to be DETECTED to be broken: `240 * 60 / 24`
    is exact in binary but `n * 60 / 23.976` is not, and a float that lands at `.4999999` takes
    the wrong branch of a rule whose whole purpose is to decide `.5`.

    Up rather than down: rounding up can only hold the last source frame one output frame longer,
    which invents nothing. Rounding down drops the tail of the source's last display interval.
    Never come up short of the source.
    """
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("both rates must be positive")
    src = Fraction(src_fps).limit_denominator(100000)
    dst = Fraction(dst_fps).limit_denominator(100000)
    exact = Fraction(n_in) * dst / src
    return int((exact + Fraction(1, 2)).__floor__())


def build_plan(n_in, src_fps, dst_fps, tol=0.0):
    """Return `(plan, stats)` — one entry per output frame, in order.

        ("copy", i)      deliver source frame i verbatim
        ("synth", i, t)  synthesise between source i and i+1 at timestep t
        ("hold", i)      deliver source frame i again; the position lies beyond it

    Output frame k maps to `pos = k * src/dst`, `i = floor(pos)`, `frac = pos - i`.

    **The nudge goes BEFORE the floor.** 22->60 at k=570 computes 208.99999999999997. Flooring
    first and repairing `frac` afterwards catches the low side only, so a position that lands just
    below its integer is classified as a synthesis at t~=1 — **the plan invents a frame it already
    had, and nothing shows it**: the count is right, the duration is right, the tail is right and
    the output plays. Four of 22->60's twenty originals go that way, silently, at full compute
    cost, replacing a perfect frame with an approximation of itself.

    **A hold is decided at CLAMP TIME, never inferred afterwards.** A position landing exactly on
    the last source frame is an ordinary copy; one lying strictly beyond it has nothing to
    interpolate toward and delivers that frame again. Counting holds by looking at the finished
    plan cannot separate those two and is right only at exact ratios.

    **The snap is applied before the clamp**, because snapping can reassign a position to the NEXT
    real frame and whether that frame exists is precisely what the clamp then decides.
    """
    if not 0.0 <= tol < 0.5:
        raise ValueError("tol is a fraction of one source interval, in [0, 0.5)")
    if n_in < 2:
        raise ValueError("a retime needs at least two source frames")

    n_out = target_count(n_in, src_fps, dst_fps)

    # **A plan that would deliver no frames is refused at the door** (contract §5, ruled
    # 2026-08-23). Half-up rounding can still land on zero — two frames of 240 fps source at an
    # 8 fps target rounds to none — and an empty delivery cannot satisfy §2's duration bound
    # against any source that has a duration, so returning an empty plan would push an
    # unanswerable job downstream. All three numbers are named because "there is no video to
    # deliver" is only actionable if you can see which of them made it so.
    #
    # Before the stats dict rather than inside it: `real_share` divided by this and raised a bare
    # `ZeroDivisionError` from the middle of a dict literal, which said nothing about rates. Found
    # while conforming this shim, and **held unpatched until the oracle carried it too** — both
    # files had it identically, so fixing one side would have manufactured a divergence out of an
    # agreement and hidden a contract question inside the mechanism built to surface one.
    if n_out < 1:
        raise ValueError(
            "a {} fps target on {} frames of {} fps source rounds to no output frames at all; "
            "there is no video to deliver, so this is refused rather than returned empty"
            .format(dst_fps, n_in, src_fps))

    ratio = src_fps / dst_fps
    last = n_in - 1

    plan = []
    n_copy = n_synth = n_hold = 0
    worst_snap = 0.0

    for k in range(n_out):
        pos = k * ratio
        i = int((pos + EPS) // 1)
        frac = pos - i
        if abs(frac) < EPS:
            frac = 0.0

        if frac > 0.0 and tol > 0.0:
            if frac <= tol:
                worst_snap = max(worst_snap, frac)
                frac = 0.0
            elif frac >= 1.0 - tol:
                worst_snap = max(worst_snap, 1.0 - frac)
                i += 1
                frac = 0.0

        if i > last or (i == last and frac > 0.0):
            plan.append(("hold", last))
            n_hold += 1
        elif frac == 0.0:
            plan.append(("copy", i))
            n_copy += 1
        else:
            plan.append(("synth", i, frac))
            n_synth += 1

    stats = {
        "n_out": n_out,
        "n_copy": n_copy,
        "n_hold": n_hold,
        "n_synth": n_synth,
        # A hold is a repeat, not new footage: the KPI counts frames that are really the source's.
        "real_frames": n_copy,
        "real_share": n_copy / n_out,
        "worst_snap_frac": worst_snap,
    }
    assert n_copy + n_hold + n_synth == n_out, "every output frame is classified exactly once"
    return plan, stats


def pad_multiple(scale=1):
    """The multiple each dimension is padded up to, per §9's padding contract."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    return int(max(PAD_BASE, PAD_BASE / scale))


class RetimeResult:
    """What `Interpolator.stream()` hands back: `.frames` and `.stats`, and **not iterable**.

    **Deliberately not a tuple, and the reason is one line of caller code** (contract §5d(a)).
    A tuple is iterable, so `for f in obj.stream(...)` succeeds — it yields the generator object
    and then the stats dict, and raises nothing at the line that made the mistake. Measured: a
    `.shape` access fails later and elsewhere with a confusing message, and a duck-typed writer
    writes two "frames" and never raises at all.

    **That mistake is not hypothetical.** `for f in obj.stream(...)` was CORRECT one wave ago,
    when `stream()` returned a generator directly, so it is the natural continuation of the shape
    that existed an hour before the first call site is written. This class turns it into an
    immediate `TypeError` at the exact line.

    **The divergence from `build_plan`'s `(plan, stats)` is deliberate.** The oracle is a pure
    function with one caller inside the kit; `stream()` is the API a pipeline consumes, and at an
    API boundary a mistake must fail where it was made. Consistency is worth less than that.

    No `__iter__` and no `__getitem__`, which is what makes `for` and unpacking both raise rather
    than one of them silently working — and no `__match_args__`, so a positional `case` pattern
    fails too. `__slots__` because a typo'd attribute on a result object is a silent None waiting
    to happen.

    One consequence of `__slots__` recorded rather than fixed: instances are not weak-referenceable,
    so a future consumer tracking open streams in a `WeakSet` would fail at construction. Nothing
    does that today, and `"__weakref__"` in the tuple is the one-line answer when something does.
    """

    __slots__ = ("frames", "stats")

    def __init__(self, frames, stats):
        self.frames = frames
        self.stats = stats

    def __repr__(self):
        # **A repr may not raise.** The `isinstance` guard covers `stats=None`, but `isinstance`
        # is true for dict SUBCLASSES too, so a mapping with a custom `get` — or an `n_out` whose
        # `__str__` raises — would take this down. Unreachable from `stream()`, whose stats are
        # always the plain dict literal `build_plan` returns; caught anyway, because the moment a
        # repr detonates is inside the debugger examining the failure it was meant to describe.
        try:
            n_out = self.stats.get("n_out") if isinstance(self.stats, dict) else None
            return "RetimeResult(frames=<stream>, stats={{n_out: {}}})".format(n_out)
        except Exception:  # noqa: BLE001 — see above; a repr that raises is worse than a vague one
            return "RetimeResult(frames=<stream>, stats=<unreadable>)"


class Interpolator:
    """Runs a plan against a frame stream. One frame of lookahead, one cached pair.

    **The model is cast once, here, and never through a global.** §9b: under torch 2.8.0 the
    reference script's `--fp16` takes `set_default_tensor_type(torch.cuda.HalfTensor)`, which is
    process-global — in-worker that makes every tensor created afterwards half-on-CUDA, SeedVR2's
    and the encoder's and the host guard's included. So the reference's live path is exactly the
    path this class may not take, and its other path is dead code at this pin. There is no working
    example of the correct form to copy: the module and its inputs are cast explicitly and nothing
    outside this object's own tensors is touched.

    **Copies and holds never enter the model.** A real frame is emitted as it arrived — not
    round-tripped through pad/synthesise/crop, which would spend compute to make a real frame
    slightly less real. That is the KPI the whole plan is shaped around.

    **This object publishes no state.** `stream()` hands its stats back inside a `RetimeResult`,
    so nothing per-call lives here and two concurrent streams share nothing but configuration.
    That is §5d(a).

    The one field that could be argued against that claim is `_model`, which `prepare()` rebinds
    rather than `__init__` setting once. It is still configuration and the exemption is written
    down here so a future §5d audit does not have to re-derive it: nothing reads it but
    `_synthesise`, there is no accessor, and casting an already-cast module is a no-op — so it
    cannot carry one stream's identity into another.

    **So the stream is not uniform in device or dtype, deliberately.** A synthesis comes back on
    `device` in `dtype`; a copy comes back exactly as the caller handed it in. Casting copies for
    tidiness would push real frames through an fp16 round trip to make them match the
    approximations — degrading the footage this design exists to preserve. The consumer converts
    if it needs one device, and it is told here rather than left to discover it.
    """

    def __init__(self, model, device="cuda", dtype=None, scale=1):
        self._model = model
        self._device = device
        self._dtype = dtype
        self._multiple = pad_multiple(scale)

    # ---- torch, imported where it is used ----------------------------------------------------

    @staticmethod
    def _torch():
        import torch  # noqa: PLC0415 — deliberate: the plan half of this module needs no torch
        return torch

    def prepare(self):
        """Cast the model once. Explicit, local, and never through a process-global default."""
        if hasattr(self._model, "to"):
            if self._dtype is not None:
                self._model = self._model.to(device=self._device, dtype=self._dtype)
            else:
                self._model = self._model.to(device=self._device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        return self

    # ---- padding -----------------------------------------------------------------------------

    def _pad(self, frame):
        """Pad H and W up to `self._multiple`; return the padded tensor and the original shape."""
        torch = self._torch()
        h, w = frame.shape[-2], frame.shape[-1]
        ph = (self._multiple - h % self._multiple) % self._multiple
        pw = (self._multiple - w % self._multiple) % self._multiple
        if ph == 0 and pw == 0:
            return frame, (h, w)
        return torch.nn.functional.pad(frame, (0, pw, 0, ph)), (h, w)

    @staticmethod
    def _crop(frame, geometry):
        h, w = geometry
        return frame[..., :h, :w]

    # ---- the stream --------------------------------------------------------------------------

    def _cast(self, frame):
        if self._dtype is not None:
            return frame.to(device=self._device, dtype=self._dtype)
        return frame.to(device=self._device)

    def _load_pair(self, cache, index, frame_a, frame_b):
        """Pad the pair for source index `index` into `cache`, once per pair.

        **`cache` belongs to one `stream()` call and never to the instance.** It used to live on
        `self`, keyed only by the plan-local index `i` — and a plan index carries no stream
        identity, so a second stream whose first synthesis named an index the instance still held
        was served the PREVIOUS clip's frames. Nothing raised: the count, the duration and the
        plan were all correct and two of the new clip's frames were interpolated from the old
        one's. Two generators from one instance interleave into the same defect, worse.
        """
        if cache.get("index") == index:
            return
        if frame_a.shape != frame_b.shape:
            # The geometry cached below is the LEFT frame's, and the crop is taken against it.
            # Refusing beats cropping frame i+1's output to frame i's size and calling it aligned.
            raise ValueError(
                "source frames {} and {} differ in shape ({} vs {}); this shim interpolates a "
                "pair and has no rule for a stream that changes size mid-clip"
                .format(index, index + 1, tuple(frame_a.shape), tuple(frame_b.shape)))
        a, geometry = self._pad(self._cast(frame_a))
        b, _ = self._pad(self._cast(frame_b))
        cache["index"], cache["padded"], cache["geometry"] = index, (a, b), geometry

    def _synthesise(self, cache, timestep):
        torch = self._torch()
        a, b = cache["padded"]
        with torch.inference_mode():
            out = self._model(a, b, timestep)
        # **Cropped and cloned out of inference mode.** A tensor produced inside `inference_mode`
        # keeps that status, and the stream would then emit two kinds of tensor: inference ones
        # for syntheses and ordinary ones for copies. A consumer doing anything in-place would
        # fail on some frames and not others, which is the worst shape a bug can have.
        return self._crop(out, cache["geometry"]).clone()

    def stream(self, frames, n_in, src_fps, dst_fps, tol=0.0):
        """Return a `RetimeResult` — `.frames` is the generator, `.stats` is the plan's stats.

        **The stats travel WITH the result and nothing is published on this object** (contract
        §5d(a)). They used to be an attribute, which made them last-writer-wins: a second
        `stream()` opened while a first generator was still undrained left `stats` reading the
        second plan while the first yielded the first clip's frames — every visible signal
        correct, the number belonging to a different clip. Invalidating on entry could not fix
        that; it only made the wrong number newer.

        **A result object rather than a tuple**, because a tuple is iterable and
        `for f in obj.stream(...)` — which was correct one wave ago — would then yield a
        generator and a dict without raising at the line that made the mistake. See
        `RetimeResult`.

        The consequence worth stating: this object now publishes nothing at all. Two interleaved
        calls cannot confuse their results because there is no shared place for a result to sit,
        and there is nothing to invalidate on entry because there is nothing to go stale.

        `n_in` is the source's frame count and comes from the container, not from counting the
        iterator: the plan is sized before the first frame is read, and a stream cannot be
        measured without consuming it. The stats are complete on return — the count is knowable
        before a frame is read, so a caller may size its loop from them without consuming anything.

        **One frame of lookahead, and the plan's `i` never goes backwards** — `pos = k * src/dst`
        is non-decreasing in k — so a single forward pass over the source suffices and nothing is
        buffered beyond the pair in hand.
        """
        plan, stats = build_plan(n_in, src_fps, dst_fps, tol=tol)
        return RetimeResult(self._emit(plan, frames, n_in), stats)

    def _emit(self, plan, frames, n_in):
        """The generator half of `stream`. Its state is per-call and none of it lives on self."""
        pair = {}
        source = iter(frames)

        held = {}          # source index -> frame, never more than two entries
        highest = -1       # the highest source index read so far

        def advance_to(index):
            """Read forward until source `index` is in hand. Frames behind the pair are dropped."""
            nonlocal highest
            while highest < index:
                try:
                    frame = next(source)
                except StopIteration:
                    raise ValueError(
                        "the stream ended after {} frames but the plan needs source frame {}; "
                        "n_in was given as {}. The count came from the container and the stream "
                        "disagrees with it — neither is guessed here."
                        .format(highest + 1, index, n_in))
                highest += 1
                held[highest] = frame
                for stale in [key for key in held if key < highest - 1]:
                    del held[stale]

        for entry in plan:
            if entry[0] == "synth":
                _, i, timestep = entry
                advance_to(i + 1)
                self._load_pair(pair, i, held[i], held[i + 1])
                yield self._synthesise(pair, timestep)
            else:
                _, i = entry
                advance_to(i)
                yield held[i]
