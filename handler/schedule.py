"""The batch list the vendored loop will actually run, worked out before a GPU is rented.

**This is layer 1: the schedule.** It replays `generation_phases.encode_all_batches` and
`inference_cli._stream_video_chunks` exactly -- stride `batch - overlap`, a trailing batch dropped
when its extent is no larger than the overlap, the overlap self-disabling at or above the batch,
uniform padding then 4n+1 padding, and chunk context concatenated on and later stripped. Every
branch here corresponds to a line in that source; where the two disagree, the source is right and
this is a bug.

**It lives in the image because the planner needs it.** Choosing the overlap is not a formula --
the landscape is jagged, and one frame of overlap either costs 2% of the sampling or spawns a
fifth pass with a four-frame tail depending on arithmetic no closed form captures. So the planner
runs the loop for every candidate and reads the answer off, which it can only do if the loop is
here rather than on a laptop. `scripts/simulate_schedule.py` imports it from here for the same
reason `tiles.py` is shared: two implementations of one piece of arithmetic is one that silently
disagrees.

**What it cannot tell you.** Nothing about VRAM and nothing about the picture. It counts passes,
frames and seams; peak memory is `solver.py` and quality is the creative team.
"""

def pad_to_4n1(count):
    """`pad_video_temporal` with `count=0, prepend=False` — the VAE's temporal constraint.

    Applied to every batch unconditionally, *after* any uniform padding, so it is the last thing
    that changes a batch's length.
    """
    if count % 4 == 1:
        return count
    return ((count - 1) // 4 + 1) * 4 + 1


class Batch(object):
    """One pass through the model, with the two padding kinds kept apart.

    `real` is frames that came from the source. `uniform` and `lattice` are frames the vendored
    code invented by reflecting the batch (`pad_video_temporal` extends with *reversed* frames,
    not a freeze), and they are paid for in sampler time and thrown away after decode.
    """

    def __init__(self, real, uniform=0, lattice=0, blended_into_previous=False):
        self.real = real
        self.uniform = uniform
        self.lattice = lattice
        self.blended_into_previous = blended_into_previous

    @property
    def window(self):
        """Frames the model sees at once — the temporal context, invented frames included."""
        return self.real + self.uniform + self.lattice

    @property
    def invented(self):
        return self.uniform + self.lattice

    def __repr__(self):
        if self.invented:
            return "{}+{}".format(self.real, self.invented)
        return str(self.real)


def schedule(frames, chunk, batch, overlap, prepend=0, uniform=False):
    """The exact batch list the pinned source will run, chunk by chunk.

    Mirrors `_stream_video_chunks` around `encode_all_batches`. Every branch here corresponds to a
    line in that source; where the two disagree, the source is right and this is a bug.

    `prepend` applies to the first chunk only — `_stream_video_chunks` zeroes it after chunk 1
    (L673-675) — and **is not removed on this worker's path**: removal lives in `_gpu_processing`
    (L1279), which is bypassed by calling `_stream_video_chunks` directly. The caller strips it.
    """
    if chunk < 1 or batch < 1 or overlap < 0 or frames < 1:
        raise ValueError("frames, chunk and batch must be >= 1 and overlap >= 0")

    step = batch - overlap
    effective_overlap = overlap
    if step <= 0:
        # `calculate_optimal_batch_params`: an overlap at or above the batch disables itself
        # rather than looping forever.
        step = batch
        effective_overlap = 0

    chunks = []
    read = 0
    previous_new = None
    while read < frames:
        new = min(chunk, frames - read)
        read += new

        # `prev_raw_tail = new_frames[-overlap:]` — bounded by the previous chunk's own length.
        context = 0 if previous_new is None or overlap <= 0 else min(overlap, previous_new)
        head = prepend if previous_new is None else 0
        total = new + context + head

        batches = []
        for start in range(0, total, step):
            if start == 0:
                first, last = 0, min(batch, total)
            else:
                first, last = start, min(start + batch, total)
                # `if end_idx - start_idx <= temporal_overlap: break` — a remainder no larger
                # than the overlap is already covered by the previous batch's tail.
                if last - first <= effective_overlap:
                    break
            real = last - first
            pad = batch - real if (uniform and real < batch) else 0
            lattice = pad_to_4n1(real + pad) - (real + pad)
            batches.append(Batch(real, pad, lattice, blended_into_previous=bool(batches)
                                 and effective_overlap > 0))

        chunks.append({
            "new": new, "context": context, "prepend": head, "total": total,
            "batches": batches,
            # What `_stream_video_chunks` yields: context stripped, prepend NOT stripped.
            "yielded": total - context,
            # What must reach the master, once this worker strips the prepend itself.
            "written": total - context - head,
        })
        previous_new = new

    return {
        "frames": frames, "chunk": chunk, "batch": batch, "overlap": effective_overlap,
        "prepend": prepend, "uniform": uniform, "step": step, "chunks": chunks,
    }


def measure(plan):
    """Structural quality and cost proxies. No VRAM, no seconds, no opinion about the picture."""
    chunks = plan["chunks"]
    every = [b for c in chunks for b in c["batches"]]
    if not every:
        raise ValueError("schedule produced no batches")

    window = plan["batch"] if plan["batch"] <= plan["chunk"] else plan["chunk"]
    real_windows = [b.window for b in every]

    # **Reflected padding is context, but it is not new information.** A pass of `2+3` hands the
    # model five frames of which three are the other two played backwards. Counting only `window`
    # would score that as a full pass, so the source count is kept beside it — the small rungs
    # score 100% on window precisely because 4n+1 rounding already pads them, and that flattery
    # is exactly what this second number refuses.
    source_windows = [b.real for b in every]

    # A seam is a boundary between two consecutive model passes. Inside a chunk it is blended;
    # between chunks the context frames are discarded and the join is a hard cut.
    blended = sum(1 for b in every if b.blended_into_previous)
    cut = len(chunks) - 1

    return {
        "batches": len(every),
        "chunks": len(chunks),
        "window": window,
        "min_window": min(real_windows),
        # The headline number: how far the worst pass falls below the window that was asked for.
        # 1.0 is a schedule where every pass got the full context.
        "worst_ratio": min(real_windows) / float(window),
        "min_source": min(source_windows),
        "worst_source_ratio": min(source_windows) / float(window),
        "starved": sum(1 for w in real_windows if w < window),
        "source_starved": sum(1 for w in source_windows if w < window),
        "blended_seams": blended,
        "cut_seams": cut,
        "invented_frames": sum(b.invented for b in every),
        # Sampler cost is ~fixed-per-batch + ~per-frame (decisions.md 4.40), so both terms matter.
        "model_frames": sum(b.window for b in every),
        # Frames the source contributed more than once: overlap recomputation plus chunk context.
        "recomputed": sum(b.window for b in every) - sum(b.invented for b in every) - plan["frames"],
        "yielded": sum(c["yielded"] for c in chunks),
        "written": sum(c["written"] for c in chunks),
        "correct_length": sum(c["written"] for c in chunks) == plan["frames"],
        # The trap: with prepend on, the stream is longer than the clip and nothing downstream
        # notices unless the worker strips it.
        "stream_overruns_by": sum(c["yielded"] for c in chunks) - plan["frames"],
    }


#: The widest overlap worth simulating. The request validator accepts up to 32 and the vendored
#: loop self-disables at or above the batch, so this is a search bound rather than a limit.
MAX_OVERLAP = 32


def choose_overlap(frames, chunk, batch, floor, prefer_up_to=4):
    """The overlap to run, found by simulating the loop rather than solving for it.

    **The landscape is jagged and no closed form survives it.** At a 49-frame window one frame of
    overlap costs about 2% more sampling, while two frames spawn an entire fifth pass with a
    four-frame tail. Which of those happens is a matter of where `frames`, `batch` and `step`
    land against each other, so the honest method is to run the loop for every candidate and read
    the answer.

    The rule, in order:

      *admissible*  every batch the schedule actually runs reaches `floor` real source frames.
                    Not the padded length -- reflected frames are context, not information, and a
                    tail of `3+2` is a three-frame pass wearing a five-frame coat.
      *cheapest*    fewest frames through the model. Overlap buys nothing but redundancy, so
                    among admissible schedules the least redundant one wins.
      *tie-break*   the larger overlap, up to `prefer_up_to`, because a wider blend hides a seam
                    better and the frames are already paid for.

    Returns `(overlap, why)`. `why` carries the whole admissible set so a plan can be argued with.

    **Zero is a real answer and it is not free.** The vendored decode blends only when the overlap
    is above zero, so `overlap=0` turns every batch join inside the chunk into an unblended cut.
    That is charged here as `unblended_joins` and left for the ranking to weigh -- this function
    reports it rather than ruling on it.
    """
    admissible, rejected = [], []
    for overlap in range(0, MAX_OVERLAP + 1):
        if overlap >= batch:
            break
        plan = schedule(frames, chunk, batch, overlap)
        measured = measure(plan)
        row = {
            "overlap": overlap,
            "model_frames": measured["model_frames"],
            "batches": measured["batches"],
            "min_source": measured["min_source"],
            "blended_seams": measured["blended_seams"],
            # Every join that is not blended: the chunk boundaries always, plus *all* of them
            # when the overlap is zero.
            "unblended_joins": measured["cut_seams"] + (
                measured["batches"] - measured["chunks"] if overlap == 0 else 0),
        }
        (admissible if measured["min_source"] >= min(floor, batch) else rejected).append(row)

    if not admissible:
        # Nothing reaches the floor. Return the schedule with the least-starved tail and say so --
        # the caller decides whether that is a plan or a refusal.
        best = max(rejected, key=lambda r: (r["min_source"], -r["model_frames"])) if rejected else \
            {"overlap": 0, "min_source": 0}
        return best["overlap"], {"reason": "no overlap reaches the {}-frame floor".format(floor),
                                 "floor_reached": False, "chosen": best, "admissible": [],
                                 "rejected": rejected}

    cheapest = min(r["model_frames"] for r in admissible)
    tied = [r for r in admissible if r["model_frames"] == cheapest]

    def rank(row):
        # **A wider blend is only worth preferring where there is something to blend.** A
        # single-batch schedule has no seam at any overlap, and every value ties at the same cost
        # -- so the tie-break would otherwise pick an arbitrary non-zero overlap and put a
        # meaningless number in the plan. There, take the smallest.
        if not row["blended_seams"]:
            return (0, 0, -row["overlap"])
        return (1, 1 if row["overlap"] <= prefer_up_to else 0, row["overlap"])

    chosen = max(tied, key=rank)
    return chosen["overlap"], {
        "reason": "{} of {} overlaps reach the {}-frame floor; cheapest costs {} model frames"
                  .format(len(admissible), len(admissible) + len(rejected), floor, cheapest),
        "floor_reached": True, "chosen": chosen, "admissible": admissible, "rejected": rejected,
    }
