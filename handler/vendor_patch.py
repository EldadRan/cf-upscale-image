"""The changes to the vendored source that cannot be made from outside it.

**Everything else this worker changes about SeedVR2 is a runtime swap** — `build_args` patches
`sys.argv`, `phasewatch` wraps `debug.log`, `pipeline` swaps `_read_frames_from_cap` and
`colorfix` swaps `wavelet_reconstruction`. Each of those is a module-level name, so rebinding it
is enough and the pinned source stays untouched on disk.

`_process_frames_core` is not that. Its last act is three lines *inside the function body*:

    if result_tensor.is_cuda or result_tensor.is_mps:
        result_tensor = result_tensor.cpu()
    if result_tensor.dtype in (torch.bfloat16, ...):
        result_tensor = result_tensor.to(torch.float32)

and the second `if` doubles the largest host allocation in the process. `ctx['final_video']` is
still holding the bfloat16 canvas when the float32 one is built, so the host carries 18 bytes a
pixel where 6 would do: 28.7 GB at 4K and 115 GB at 8K, for a widening whose only consumer
immediately narrows it to `uint8`. It is what killed the 8K container twice — `exit code 137`,
cgroup OOM, with 35 GB of the card still free.

There is no name to rebind. Patching `inference_cli.torch` so the dtype test misses would work
and would also break every other honest use of `torch.bfloat16` in that module. So the source is
edited, once, at build time.

**Exact-match or fail.** The replacement is anchored on the literal text at the pinned commit
(`SEEDVR2_COMMIT` in the Dockerfile). If a future bump moves that code by so much as a space
this raises and the build stops, which is the whole point: a patch that silently no-ops would
ship an image that looks patched, passes its own acceptance, and OOMs the first time anyone
sends it 8K.

**The second patch: the checksum of what content-addressing already proved.**
`download_weight()` validates an existing model file against a size+mtime cache, and on a miss
reads all 16.4 GiB of it through SHA256. Our image never seeds that cache — `bake_weights.py`
fetches through `hf_hub_download` deliberately, and says why in its own docstring — so *every
fresh worker's first job* pays that read before a single frame is touched. On a fully-materialized
NVMe host it costs seconds and nobody noticed for weeks; on a lazy-streaming host the layer is
faulted in through the read and it costs **6m06s and 12m29s, both measured** (F-2026-08-19-31),
in complete silence, which is what made it look like a stall in our own code.

The check is redundant by architecture, not merely expensive: the weights arrive inside a
digest-addressed image whose every blob the gate verifies before it is pinned. Re-hashing them at
runtime re-proves what content-addressing has already proved, once per worker, on the customer's
clock.

**Not the mtime cache.** Seeding the validation cache at build time is the obvious fix and it is
the wrong one: lazy-streaming filesystems need not preserve mtimes, so a fix keyed on mtime fails
in exactly the place the bug lives. The skip is keyed on `SEEDVR2_WEIGHTS_BAKED`, which the
Dockerfile sets immediately after the bake — a statement about how the image was built, which no
filesystem can contradict. The file-existence guard stays, so an image whose weights are somehow
absent still downloads them.

    python3 vendor_patch.py /app/SeedVR2
"""

import os
import sys

#: The environment flag the Dockerfile sets immediately after the bake. Read from the vendored
#: code, so its name is part of this patch's contract with the image.
BAKED_FLAG = "SEEDVR2_WEIGHTS_BAKED"

#: The runtime validation gate, verbatim at commit 4490bd1f — `src/utils/downloads.py`, inside
#: `download_weight`'s per-model loop. Matched as an exact substring, trailing whitespace and all.
_VALIDATE = """        ## Quick cache check first
        if is_file_validated_cached(filepath, cache_dir):
            # Debug log: Model already validated (using cache)
            if debug:
                debug.log(f"{model_type} model already validated (cache): {filepath}", category="setup")
            continue
"""

#: What replaces it. The vendored gate is left standing underneath, untouched: this adds a branch
#: above it and removes nothing, so an image *without* the flag behaves exactly as upstream does.
#: `os.path.exists` is kept deliberately — the flag says the build baked weights, not that this
#: particular filename is among them, and a missing file must still reach the downloader below.
_SKIP = """        # cf-upscale-worker (F-2026-08-19-31): baked weights are not re-hashed at runtime.
        # This file arrived inside a digest-addressed image, blob-verified before the tag was
        # pinned; the SHA256 below re-proves that at a cost of 16.4 GiB read per fresh worker —
        # seconds on materialized NVMe, 6-12 minutes measured on lazy-streaming hosts, silently,
        # on the customer's clock. The existence check stays: absent weights still download.
        if os.environ.get("{flag}") == "1" and os.path.exists(filepath):
            if debug:
                debug.log(f"{{model_type}} model baked into the image, validation skipped: {{filepath}}",
                          category="setup")
            continue

        ## Quick cache check first
        if is_file_validated_cached(filepath, cache_dir):
            # Debug log: Model already validated (using cache)
            if debug:
                debug.log(f"{{model_type}} model already validated (cache): {{filepath}}", category="setup")
            continue
""".format(flag=BAKED_FLAG)

#: The sentinel each patch leaves behind, and the file it belongs to. The Dockerfile asserts both
#: after the dependency install, where a vendored bump that moved either costs ten minutes of CI.
APPLIED_MARKERS = (
    ("inference_cli.py", "cf-upscale-worker (release item 3)"),
    (os.path.join("src", "utils", "downloads.py"), "cf-upscale-worker (F-2026-08-19-31)"),
    # Both C+3 patches leave the same marker, in two files. The Dockerfile asserts each file
    # separately, so one taking and the other silently missing is still caught.
    (os.path.join("src", "core", "generation_phases.py"),
     "cf-upscale-worker (F-2026-08-21-54)"),
    (os.path.join("src", "optimization", "memory_manager.py"),
     "cf-upscale-worker (F-2026-08-21-54)"),
)


#: The widening, verbatim at commit 4490bd1f. Indentation included — this is matched as an exact
#: substring, not a regex, so there is nothing here to get subtly wrong.
_WIDEN = """    if result_tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        result_tensor = result_tensor.to(torch.float32)
"""

#: What replaces it. **Not a deletion** — the narrow dtypes still have to reach a consumer that
#: understands them, and `float8` does not survive the arithmetic downstream the way `bfloat16`
#: does. So float8 is still widened and bfloat16 is handed over as it is; `pipeline._stream`
#: upcasts it a slice at a time, which is the same arithmetic on a buffer that is bounded instead
#: of proportional to the clip.
_KEEP = """    # cf-upscale-worker (release item 3): bfloat16 is handed to the caller as it is.
    # Widening the whole canvas here doubled the largest host allocation in the process while
    # `ctx['final_video']` still held the original, and the only consumer narrows it to uint8
    # immediately. `pipeline._stream` now upcasts one slice at a time — identical values, bounded
    # buffer. float8 is still widened: it has no such consumer.
    if result_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        result_tensor = result_tensor.to(torch.float32)
"""


#: **A DiT phase that finds no model stops at the door, by name** (F-2026-08-21-54, part 4).
#: The vendored line is `if runner.dit and ...` — an `and` that short-circuits past its own
#: materialisation call when `runner.dit` is None, walks into the batch loop, and dies fifty
#: lines later at `next(dit_model.parameters())` with `'NoneType' object has no attribute
#: 'parameters'`. That is how the seam retest ended.
#:
#: **This was first written as a reload, and the reload could not work** (found in review). Two
#: independent reasons, both in the vendored source: `materialize_model` begins by reading
#: `model = runner.dit` and returns immediately on None — "No DiT model structure found" — because
#: it fills weights into an existing meta-device structure and never builds one; and it clears
#: `runner._dit_checkpoint` itself the moment it succeeds, so the guard on the reload branch is
#: false after the chunk's own first materialisation anyway. The branch would have logged a
#: warning and fallen through to exactly the AttributeError it was written to prevent.
#:
#: Rebuilding the structure from here is not available — it needs `configure_runner`'s inputs,
#: which this scope does not have. So the honest patch converts an invariant violation into a
#: named one: something evicted the model in a phase that needs it, and the run says so at the
#: door instead of fifty lines into a loop. The ladder is what can null that pointer, and this
#: release stops it doing so mid-chunk; this is the net under that, and a net that reports
#: honestly beats one that pretends to catch.
#:
#: Not mirrored onto the VAE, deliberately. The same shape exists at `generation_phases.py:896`,
#: but the only thing that could null `runner.vae` mid-chunk was the eviction firing at a decode
#: boundary, and `MODEL_FREE_PHASES` no longer contains one. Each vendored anchor is a drift
#: liability that costs a build when upstream moves; one is worth carrying here, two is not.
_DITLESS = """        # Materialize DiT if still on meta device
        if runner.dit and next(runner.dit.parameters()).device.type == 'meta':"""

_RELOAD = """        # Materialize DiT if still on meta device
        # cf-upscale-worker (F-2026-08-21-54): fail at the door rather than walking a None into
        # the batch loop to die fifty lines down at next(dit_model.parameters()). No reload is
        # offered here on purpose -- materialize_model() returns on a None structure and clears
        # _dit_checkpoint once it has run, so a reload branch here would log a warning and fall
        # through to the very AttributeError it was written to prevent.
        if runner.dit is None:
            raise RuntimeError(
                "cf-upscale-worker: the DiT is absent at the start of the phase that needs it. "
                "Something released it mid-chunk -- the residency ladder may only evict in "
                "phases that reference no model.")
        if runner.dit and next(runner.dit.parameters()).device.type == 'meta':"""


#: **The cleanup path had the same bug and re-raised on the same None** (F-2026-08-21-54, part 5).
#: `cleanup_dit` reads `next(runner.dit.parameters()).device` under `except StopIteration`, which
#: an empty model raises and a *missing* one does not — `None.parameters()` is an AttributeError.
#: So the error handler for a dit-less runner died of the thing it was handling, and the original
#: failure reached the log wearing the cleanup's traceback.
#:
#: Widening the clause rather than returning early on purpose: the rest of `cleanup_dit` — the
#: BlockSwap unwind, the sampler and schedule teardown — still has work to do on a runner whose
#: model is gone, and skipping it would trade a crash for a leak.
#: **Anchored on the log line above each clause, because `except StopIteration: pass` appears
#: three times in that file** — `cleanup_dit`, `cleanup_vae`, and an unrelated device read. The
#: exact-match-or-fail rule caught the ambiguity on the first run rather than patching whichever
#: one came first, which is the entire reason that rule exists. Both cleanups are patched: the
#: eviction nulls the VAE beside the DiT, so the same handler would die the same way on it.
_CLEANUP_DIT = """            debug.log("DiT on meta device - keeping structure for cache", category="cleanup")
    except StopIteration:
        pass"""

_CLEANUP_DIT_WIDE = """            debug.log("DiT on meta device - keeping structure for cache", category="cleanup")
    except (StopIteration, AttributeError):
        # cf-upscale-worker (F-2026-08-21-54): StopIteration is an EMPTY model; a model that is
        # not there at all raises AttributeError, and this handler used to die of it.
        pass"""

_CLEANUP_VAE = """            debug.log("VAE on meta device - keeping structure for cache", category="cleanup")
    except StopIteration:
        pass"""

_CLEANUP_VAE_WIDE = """            debug.log("VAE on meta device - keeping structure for cache", category="cleanup")
    except (StopIteration, AttributeError):
        # cf-upscale-worker (F-2026-08-21-54): see the DiT cleanup above — the eviction nulls
        # both models, so both handlers have to survive finding one gone.
        pass"""


def _patch(target, before, after, what, consequence):
    """Replace `before` with `after` in `target`, exactly once, or stop the build.

    The two patches differ only in their text and in what an unpatched image does wrong, so the
    exact-match-or-fail rule lives here once. `after` containing itself is the idempotency test:
    a re-run in a shell, or a rebuilt layer, must not be an error.
    """
    with open(target) as handle:
        source = handle.read()

    if after in source:
        print("vendor_patch: {} already applied".format(what))
        return target

    found = source.count(before)
    if found != 1:
        raise SystemExit(
            "vendor_patch: expected {} exactly once in {}, found {}.\n"
            "The pinned SeedVR2 commit has moved under this patch. Re-read the vendored source "
            "and update the anchor; do not skip the patch, because {}".format(
                what, target, found, consequence))

    with open(target, "w") as handle:
        handle.write(source.replace(before, after))
    print("vendor_patch: {} patched in {}".format(what, target))
    return target


def apply(root):
    """Patch the vendored tree at `root`. Returns the paths written, in order."""
    written = [_patch(
        os.path.join(root, "inference_cli.py"), _WIDEN, _KEEP,
        "the bfloat16 handoff widening",
        "an unpatched image OOMs its cgroup at 8K with the card half empty.")]

    written.append(_patch(
        os.path.join(root, "src", "utils", "downloads.py"), _VALIDATE, _SKIP,
        "the runtime weight re-validation",
        "an unpatched image re-hashes 16.4 GiB on every fresh worker's first job — 6 to 12 "
        "minutes of silence on a lazy-streaming host, billed to whoever sent that job."))

    written.append(_patch(
        os.path.join(root, "src", "core", "generation_phases.py"), _DITLESS, _RELOAD,
        "the dit-less phase entry",
        "an unpatched image walks a None into the batch loop and dies at "
        "`next(dit_model.parameters())` with no indication of what took the model — which is "
        "how the 240-frame seam retest ended."))

    for what, before, after in (("the dit-less cleanup", _CLEANUP_DIT, _CLEANUP_DIT_WIDE),
                                ("the vae-less cleanup", _CLEANUP_VAE, _CLEANUP_VAE_WIDE)):
        written.append(_patch(
            os.path.join(root, "src", "optimization", "memory_manager.py"), before, after, what,
            "an unpatched image's cleanup path re-raises on the same missing model, so the "
            "original failure reaches the log wearing the cleanup's traceback."))

    return written


if __name__ == "__main__":
    for path in apply(sys.argv[1] if len(sys.argv) > 1
                      else os.environ.get("SEEDVR2_DIR", "/app/SeedVR2")):
        print("vendor_patch: wrote {}".format(path))
