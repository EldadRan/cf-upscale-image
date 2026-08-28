"""What machine this job landed on, read at runtime.

**None of this can come from configuration.** The endpoint offers several GPU types so a cold
start always finds a free card, which means the job does not know which card it has until it is
on it. A worker configured with "48 GB" would be wrong on the run that mattered.

Everything here degrades rather than raises. A missing figure is reported as `None` and the
estimator treats it as unknown — which sends it to the conservative end, where unknown belongs.
Guessing a number would be worse than not having one: the estimator's whole job is to decide
whether the work fits, and it cannot do that honestly against a fabricated total.

torch is imported lazily. The rung-1 contract suite runs in CI with no GPU and no torch, and it
exercises this module against injected readings rather than against a real card.
"""

import os
import shutil
import time

BYTES_PER_GB = 1024 ** 3


def _gpu():
    """(name, total_gb, free_gb) or (None, None, None).

    `torch.cuda.mem_get_info()` reports the **driver's** view of free memory, which is what
    matters: it accounts for other processes on the card, and a serverless GPU is not always
    exclusively this job's.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 — no torch is a valid state, not an error
        return None, None, None
    try:
        if not torch.cuda.is_available():
            return None, None, None
        properties = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        return properties.name, total / BYTES_PER_GB, free / BYTES_PER_GB
    except Exception:  # noqa: BLE001 — a driver that will not answer is 'unknown', not a crash
        return None, None, None


def _cuda_versions():
    """(driver_cuda, torch_built_for) — both as strings, or `None` where unknowable.

    **Recorded because the driver decides whether this image runs at all, and it was invisible.**
    The base image asks for a minimum host-driver CUDA version and the container is refused before
    any of this executes if the host is older. That bound is deliberately loosened in the
    Dockerfile to recover most of RunPod's fleet, which trades a clean refusal at container start
    for a possible failure at model load on a host that is genuinely too old.

    That trade is only acceptable if the driver is written down. Without it, "did this fail
    because the host was old?" is unanswerable after the fact — and the failure would be filed as
    an `internal` worker fault, which is the wrong table and the wrong owner.

    `torch.version.cuda` is what the build targets; the driver's own view comes from the runtime.
    They differ by design under minor-version compatibility, so both are recorded — a mismatch is
    normal, and only the pair says which.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None, None
    built = getattr(torch.version, "cuda", None)
    driver = None
    try:
        # Private, and guarded: an integer like 12040 for 12.4. There is no public API for the
        # driver's CUDA version, and shelling out to nvidia-smi on every job to read one number
        # costs a subprocess in the hot path.
        raw = torch._C._cuda_getDriverVersion()  # noqa: SLF001
        if raw:
            driver = "{}.{}".format(raw // 1000, (raw % 1000) // 10)
    except Exception:  # noqa: BLE001 — an unreadable driver version is 'unknown', not a failure
        driver = None
    return driver, built


def _compute_arch():
    """(device_capability, kernels_this_build_carries) — e.g. `"sm_122"`, `["sm_90", "sm_120"]`.

    **The question this answers is "will this card run at all", and it was unanswerable.** A torch
    build carries kernels for a fixed set of architectures. Run it on a newer card and it fails at
    the first kernel launch with *no kernel image is available for execution on the device* —
    immediately, and with nothing in the snapshot to explain why a card with plenty of free VRAM
    refused to compute.

    It matters right now for Blackwell. The RTX PRO 6000 reports `sm_122` while the cu128 wheels
    are built for `sm_120` plus PTX, so it should JIT forward — should, because PyTorch's own
    issue on exactly this closed without a documented resolution and the reports conflict. Rather
    than settle it with a paid job on one card, the worker states both halves on every job and
    every card answers it once, permanently.

    Read without allocating: `get_arch_list` reads the build, and the capability query touches the
    driver but launches nothing.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None, None
    arches = None
    try:
        arches = list(torch.cuda.get_arch_list()) or None
    except Exception:  # noqa: BLE001
        arches = None
    capability = None
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            capability = "sm_{}{}".format(major, minor)
    except Exception:  # noqa: BLE001 — an unreadable capability is 'unknown', not a failure
        capability = None
    return capability, arches


#: Environment names that have carried the data centre, most current first.
#:
#: **A list rather than a name, because the platform's is not ours to pin.** `RUNPOD_DC_ID` is the
#: documented one today; the others have appeared in worker environments and cost nothing to try.
#: Reading several and reporting *which* answered is the difference between "the platform stopped
#: exposing it" and "we were reading the wrong key" — two diagnoses that look identical from a
#: log line saying `unknown`, and this repo has paid for that confusion more than once.
DATACENTER_ENV = (
    "RUNPOD_DC_ID",
    "RUNPOD_DATACENTER_ID",
    "RUNPOD_DATACENTER",
    "RUNPOD_REGION",
)


def datacenter():
    """`(id, which env name carried it)`, or `(None, None)` when the platform says nothing.

    **Why a worker's data centre is a measurement axis and not trivia** (F-2026-08-19-37's wave).
    Two H200 data centres behaved differently on the same image on the same day: one could not
    pull from GHCR at all, and another streams image layers lazily — which is where the 6-to-12
    minute first-job silence lives, since the 16.4 GiB checkpoint is faulted in through whichever
    read touches it first. The `[load]` strip now measures that read, and a strip figure without a
    data centre beside it cannot be compared to any other strip figure. With one, the corpus sorts
    itself and the question answers itself on first contact with a slow DC.
    """
    for name in DATACENTER_ENV:
        value = os.environ.get(name)
        if value:
            return value, name
    return None, None


#: cgroup v2's memory ceiling, then v1's. Both are read because a host may run either, and the
#: one that exists is the one that kills.
CGROUP_V2_MEMORY_MAX = "sys/fs/cgroup/memory.max"
CGROUP_V1_MEMORY_MAX = "sys/fs/cgroup/memory/memory.limit_in_bytes"

#: cgroup v2's CPU quota: "<quota> <period>", or "max <period>" when unlimited.
CGROUP_V2_CPU_MAX = "sys/fs/cgroup/cpu.max"

#: **What the OOM killer actually watches**, and what this worker has never recorded
#: (F-2026-08-20-41, CF exemption 34b528d). Our `[host]` banners report VmRSS. The kernel decides
#: on `memory.current`, which is anon *plus page cache plus kernel memory* — and on the F-41
#: staircase the platform read 39.5 GiB where our banner read 28.3 on a comparable state. The
#: 11.2 GiB gap fits inside the 16.4 GiB checkpoint read exactly, which is a hypothesis, not a
#: finding, and the difference between those two is what this constant exists to close.
CGROUP_V2_MEMORY_CURRENT = "sys/fs/cgroup/memory.current"
CGROUP_V1_MEMORY_USAGE = "sys/fs/cgroup/memory/memory.usage_in_bytes"

#: The breakdown that settles the question the total cannot. **Clean page cache is reclaimed
#: before an OOM, not counted toward one** — so if the gap above is file-backed, charging it
#: would under-chunk as badly as `HOST_RESERVE = 4.0` over-chunked, with the sign flipped. But
#: F-41 *died* at 46.45 of 46.57 rather than reclaiming, which means either the cache was not
#: reclaimable or anon alone reached the wall. One run with these two lines recorded tells them
#: apart: `file` shrinking while `anon` climbs is reclaimable cache, and both climbing into the
#: limit is not.
#:
#: **The same seam wears a third mask** (CF): the A40 stills' load-end floors read 1.2–3.0 GiB,
#: which cannot contain 16.4 GiB of weights — so on that path the weights are mmap'd and
#: file-backed, invisible to RSS and visible to `memory.current`. Three anomalies, one
#: measurement family.
CGROUP_V2_MEMORY_STAT = "sys/fs/cgroup/memory.stat"
CGROUP_V1_MEMORY_STAT = "sys/fs/cgroup/memory/memory.stat"

#: The `memory.stat` keys worth carrying. Not the whole file: it runs to dozens of lines, most of
#: them irrelevant here, and a banner nobody can read at a glance is a banner nobody reads. `anon`
#: is what cannot be reclaimed, `file` is what can, `slab` and `kernel_stack` are the kernel's own
#: — together they account for nearly all of `memory.current`, and their sum against the total is
#: itself a check that the reading is intact. v1 spells the first two `rss` and `cache`.
MEMORY_STAT_KEYS_V2 = ("anon", "file", "slab", "kernel_stack", "sock", "shmem")
MEMORY_STAT_KEYS_V1 = ("rss", "cache", "shmem")


def _read_int(root, relative):
    try:
        with open(os.path.join(root, relative)) as handle:
            raw = handle.read().strip()
    except (OSError, ValueError):
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def memory_current_gb(root="/"):
    """This container's usage **as the killer counts it**, or None off-cgroup.

    Reported beside RSS rather than instead of it. Two numbers that disagree are a finding; one
    number that silently replaced another is how a model gets refitted against the wrong axis —
    and this project has already retired one set of constants for being measured on the wrong
    vintage. Both are recorded, the model is fitted afterwards, and the fit says which it chose.
    """
    for relative in (CGROUP_V2_MEMORY_CURRENT, CGROUP_V1_MEMORY_USAGE):
        value = _read_int(root, relative)
        if value:
            return value / BYTES_PER_GB
    return None


def memory_breakdown_gb(root="/"):
    """`{anon, file, slab, …}` in GiB from `memory.stat`, or `{}` where there is no cgroup.

    **The whole point is the anon/file split**, so the v1 spelling is normalised onto the v2 one
    (`rss` → `anon`, `cache` → `file`): a corpus whose rows use two vocabularies for one quantity
    is a corpus that has to be re-read before it can be fitted, and the re-reading is where the
    mistake goes in.
    """
    for relative, keys, rename in (
            (CGROUP_V2_MEMORY_STAT, MEMORY_STAT_KEYS_V2, {}),
            (CGROUP_V1_MEMORY_STAT, MEMORY_STAT_KEYS_V1, {"rss": "anon", "cache": "file"})):
        try:
            with open(os.path.join(root, relative)) as handle:
                lines = handle.read().split("\n")
        except (OSError, ValueError):
            continue
        found = {}
        for line in lines:
            parts = line.split()
            if len(parts) != 2 or parts[0] not in keys:
                continue
            try:
                found[rename.get(parts[0], parts[0])] = int(parts[1]) / BYTES_PER_GB
            except ValueError:
                continue
        if found:
            return found
    return {}


def physical_ram_gb(root="/"):
    """What the machine has. `/proc/meminfo` rather than psutil — one less dependency in an image
    already carrying 17 GB of weights, and this is the only fact needed from it.

    **This is no longer the number anything plans against**, and the rename is the point: it is
    the host's RAM, not this container's, and reading one as the other is F-2026-08-19-37.
    """
    try:
        with open(os.path.join(root, "proc/meminfo")) as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except (OSError, ValueError, IndexError):
        pass
    if root != "/":
        return None
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / BYTES_PER_GB
    except (ValueError, OSError, AttributeError):
        return None


def memory_limit_gb(root="/"):
    """This container's own memory ceiling, or None when it is not capped.

    **The number that actually kills.** A breach of this is a cgroup SIGKILL: no exception, no
    bundle, no walk — `planner.py` says so in its own comment and then planned against the
    physical host anyway. This worker has been allocated 24 vCPUs and 377 GB of a machine whose
    `/proc/meminfo` reports 3019.4 GiB, so the chunk cap has been defending a ceiling eight times
    higher than the one that exists.

    Unlimited is reported as None rather than as a huge number, in both cgroup dialects: v2 writes
    the literal `max`, and v1 writes a near-2^63 sentinel that is not a measurement of anything.
    A limit at or above the physical total is also None — it is not a slice, and carrying it would
    make every unsliced host look capped.
    """
    physical = physical_ram_gb(root)
    for relative in (CGROUP_V2_MEMORY_MAX, CGROUP_V1_MEMORY_MAX):
        try:
            with open(os.path.join(root, relative)) as handle:
                raw = handle.read().strip()
        except (OSError, ValueError):
            continue
        if not raw or raw == "max":
            continue
        try:
            limit = int(raw) / BYTES_PER_GB
        except ValueError:
            continue
        # v1's "unlimited" is 2^63-ish expressed in bytes; anything at or past the machine's own
        # RAM is not a constraint this worker has to plan around.
        if limit <= 0 or (physical and limit >= physical):
            continue
        return limit
    return None


def effective_ram_gb(root="/"):
    """**The one number anything may plan against**: the smaller of what the machine has and what
    this container is allowed.

    One choke point rather than three call sites each deciding, because that is exactly how this
    went wrong — `hardware.read`, `phasewatch.host_total_gb` and the planner's chunk cap all
    reached for the physical figure independently, so no amount of cross-reading two of them
    would have revealed the third.
    """
    physical = physical_ram_gb(root)
    limit = memory_limit_gb(root)
    if physical is None:
        return limit
    if limit is None:
        return physical
    return min(physical, limit)


def cpu_quota(root="/"):
    """Cores this container's CPU *quota* allows, or None when unquotaed.

    **Affinity cannot see this.** `sched_getaffinity` reports the CPUs a process may run on, which
    is the right answer when a container is pinned by mask — but a container throttled by quota
    still sees every CPU in its mask and is simply stopped when it has spent its slice. The two
    mechanisms are independent and a host may use either, so the usable count is the smaller.
    """
    try:
        with open(os.path.join(root, CGROUP_V2_CPU_MAX)) as handle:
            quota, _, period = handle.read().strip().partition(" ")
    except (OSError, ValueError):
        return None
    if quota == "max" or not period:
        return None
    try:
        allowed = float(quota) / float(period)
    except (ValueError, ZeroDivisionError):
        return None
    return allowed if allowed > 0 else None


def _host_ram_gb():
    """Kept as the name the snapshot reports, now answering with the effective ceiling."""
    return effective_ram_gb()


def _free_disk_gb(path):
    try:
        return shutil.disk_usage(path).free / BYTES_PER_GB
    except OSError:
        return None


def read(workdir="/"):
    """A snapshot of the machine. Reported on every job, whatever the outcome.

    It goes in the response and in the manifest because CF cannot otherwise explain why two
    identical requests cost differently, and it goes in the diagnostics bundle because an
    estimate that was wrong is only useful beside the hardware it was wrong about.
    """
    name, total_vram, free_vram = _gpu()
    driver_cuda, built_cuda = _cuda_versions()
    capability, arch_list = _compute_arch()
    return {
        "gpu_name": name,
        # Which driver actually ran this job, and what torch was built against. See _cuda_versions.
        "cuda_driver": driver_cuda,
        "cuda_built_for": built_cuda,
        # What the card is, and what this build has kernels for. A card outside the list either
        # JITs from PTX or fails at the first launch — see _compute_arch.
        "compute_capability": capability,
        "kernel_arch_list": arch_list,
        "vram_total_gb": _round(total_vram),
        "vram_free_gb": _round(free_vram),
        # **The effective ceiling, which is what the planner consumes.** Both halves ride beside
        # it so a sliced host is visible in every snapshot, every bundle and every run-record —
        # a slice that only shows up as a smaller number is a slice nobody can audit.
        "host_ram_gb": _round(_host_ram_gb()),
        "host_ram_physical_gb": _round(physical_ram_gb()),
        "host_ram_limit_gb": _round(memory_limit_gb()),
        "cpu_quota": _round(cpu_quota()),
        # **What the OOM killer counts, at snapshot time.** A single reading rather than a series
        # — the banners carry the series — but it lands in the manifest and the response, so even
        # a job whose banners are lost says once what the platform thought it was using.
        "host_memory_current_gb": _round(memory_current_gb()),
        "host_memory_breakdown_gb": {k: _round(v)
                                     for k, v in sorted(memory_breakdown_gb().items())} or None,
        "free_disk_gb": _round(_free_disk_gb(workdir)),
        # **MOVED UP from the attempt's `cpu` block** (CF, 2026-08-28). Both are per-JOB facts:
        # the mask a process may run on and the cores visible to it do not change between
        # attempts, so recording them per attempt put a constant in a varying record and left
        # the snapshot -- which is what the manifest, the response and every bundle carry --
        # unable to say how many cores the job had. `phasewatch` still reports them per attempt;
        # this is the copy a reader finds first.
        "usable_cores": cpu_count(),
        "affinity_cores": _affinity_cores(),
        # **WHICH cpu, not how many.** See `_cpu_model`.
        "cpu_model": _cpu_model(),
        # **The versions that actually ran**, rather than what the Dockerfile pinned: a wheel
        # resolved at build time and a wheel imported at run time are not the same claim, and a
        # measurement is a measurement of the stack that produced it.
        "libraries": _library_versions(),
        # The one-shot GPU telemetry. See `_gpu_topology` for why these and not current clocks.
        "gpu_topology": _gpu_topology(),
    }


def _affinity_cores():
    """Cores in this process's mask, before any quota is applied. `None` if unreadable.

    Reported beside the quota because they answer different questions and this project has
    already been bitten by conflating them (F-2026-08-19-37).

    **Lives here rather than in `phasewatch`**, which is where it was written: `phasewatch`
    imports this module lazily to keep the cycle absent, so the dependency only runs one way and
    the lower-level module is the only one that can be the single home. `phasewatch` delegates.
    """
    try:
        return len(os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        try:
            return os.cpu_count()
        except Exception:  # noqa: BLE001
            return None


def cpu_count():
    """Cores this container may actually use — the smaller of the mask and the quota.

    **`sched_getaffinity`, not `cpu_count`.** The second reports the machine's cores and the
    first reports the ones this process is allowed on, and a container pinned to a fraction of a
    large host is exactly the case worth knowing about: the phase-4 tail runs at one core's worth
    while thirty sit idle, and the first question that investigation has to answer is how many
    cores there were to be idle.

    **And the quota, which affinity cannot see** (F-2026-08-19-37). A container pinned by mask is
    visible to `sched_getaffinity`; one throttled by `cpu.max` sees every CPU in its mask and is
    simply stopped when its slice is spent. Independent mechanisms, so the usable figure is the
    smaller.
    """
    counts = [_affinity_cores()]
    try:
        quota = cpu_quota()
        if quota:
            counts.append(int(max(1, quota)))
    except Exception:  # noqa: BLE001
        pass
    counts = [c for c in counts if c]
    return min(counts) if counts else None


#: **The sentinel for "we looked and could not read it".** Distinct from `None`, which this
#: module already uses for "the mechanism does not apply here" — an unquotaed container's
#: `cpu_quota`, a machine with no cgroup. The gate's posture for telemetry was explicit: an
#: absent value must be distinguishable from a value we looked for under the wrong name, and two
#: nulls with different meanings are exactly the ambiguity that makes a corpus unanalysable.
UNREADABLE = "unreadable"


def _cpu_model(root="/"):
    """The CPU's marketing name from `/proc/cpuinfo`, or `UNREADABLE`.

    **Named rather than counted.** Cores are already recorded twice over; what no field carried
    was WHICH cpu, and the two runs that differ 21.6 against 40.7 s/frame on one worker id are
    the reason it matters -- a shared-tenancy neighbour is invisible, but a different host
    generation is not.
    """
    try:
        with open(os.path.join(root, "proc/cpuinfo")) as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip() or UNREADABLE
    except Exception:  # noqa: BLE001 — telemetry must never fail a job
        return UNREADABLE
    return UNREADABLE


def _library_versions():
    """`{torch, numpy, opencv}` — each a version string, `None` if absent, `UNREADABLE` on error.

    **Absent is a real state and is not an error.** Rung 1 runs in CI with no torch at all, so
    `None` there is the honest answer; `UNREADABLE` means the import raised for some other
    reason, which is a different fact and one somebody would want to see.
    """
    versions = {}
    for key, module in (("torch", "torch"), ("numpy", "numpy"), ("opencv", "cv2")):
        try:
            versions[key] = __import__(module).__version__
        except ImportError:
            versions[key] = None
        except Exception:  # noqa: BLE001
            versions[key] = UNREADABLE
    return versions


#: Read once and kept. **Every field in it is stable for the life of the container** -- link
#: topology, clock ceilings and which card this is do not change between attempts -- and
#: `hardware.read()` is called three or four times a job, once before validation and twice inside
#: the model attempt, two of those landing inside the phase stopwatch that feeds the ETA and the
#: rate. A per-read shell-out would put process-spawn latency inside a measured phase, which is
#: the opposite of what telemetry is for.
_TOPOLOGY_CACHE = None

#: The eight fields, in ONE query. `nvidia-smi --query-gpu` takes a comma-separated list and
#: returns a single CSV row, so this is one subprocess rather than eight for the same
#: information. The first draft spawned eight, each with its own 5 s timeout -- up to 40 s per
#: read on a wedged driver, and the read inside `_Ratchet.step()` happens straight after
#: `release_gpu_memory()`, which is the moment the driver is least likely to answer promptly.
_TOPOLOGY_FIELDS = ("pcie.link.width.current", "pcie.link.width.max",
                    "pcie.link.gen.current", "pcie.link.gen.max",
                    "clocks.max.sm", "clocks.max.memory", "serial", "uuid")

_TOPOLOGY_KEYS = ("pcie_link_width_current", "pcie_link_width_max",
                  "pcie_link_gen_current", "pcie_link_gen_max",
                  "clock_max_sm_mhz", "clock_max_mem_mhz", "gpu_serial", "gpu_uuid")


def _smi_row(fields, timeout=5):
    """One `nvidia-smi --query-gpu` call for device 0. A list of strings, or a string saying why
    not. **Never raises.**

    **Shelled out rather than read through torch**, because none of these is exposed there:
    torch reports what the allocator sees, and link topology and clock ceilings are properties of
    the machine rather than of the process.

    **The failure REASON is kept rather than collapsed.** An earlier draft returned a bare
    `UNREADABLE` for an absent binary, a non-zero exit, a timeout and an empty field alike --
    which defeats the sentinel's own purpose, since a field name a future driver stops
    recognising would be indistinguishable from a machine with no GPU. `stderr` carries that
    distinction and keeping it costs nothing.
    """
    try:
        import subprocess  # noqa: PLC0415
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields),
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return "{}: {}".format(
                UNREADABLE,
                (out.stderr or "").strip()[:120] or "exit {}".format(out.returncode))
        rows = (out.stdout or "").strip().splitlines()
        if not rows:
            return "{}: nvidia-smi returned no rows".format(UNREADABLE)
        return [part.strip() or UNREADABLE for part in rows[0].split(",")]
    except FileNotFoundError:
        return "{}: nvidia-smi is not installed".format(UNREADABLE)
    except Exception as exc:  # noqa: BLE001 — a timeout, a wedged driver: never fatal
        return "{}: {}".format(UNREADABLE, type(exc).__name__)


def _gpu_topology():
    """**The one-shot half of the GPU telemetry** (CF, 2026-08-28): what is STABLE for the life
    of the container -- link topology, clock ceilings, and which physical machine this is.

    **Why these and not current clocks.** Two runs at 13,608,000 px, window 1, same tiling, same
    worker id, same datacentre, same core count, same image measured 21.6 and 40.7 s/frame --
    encode x2.41, decode x2.40, sampling x1.62, postprocess x0.83, with CPU work flat. The
    bandwidth-bound phases were hit hardest, which points at shared-tenancy memory bandwidth or
    PCIe link degradation rather than at the host CPU or thermals. A DEGRADED LINK SHOWS UP HERE:
    a card negotiated at x8 gen3 when the machine offers x16 gen5 is visible in a single read.

    **The volatile half is deliberately NOT here.** Current clocks and throttle reasons are
    sampled at phase boundaries in a separate wave item, because a read at `hardware.read()` time
    samples the machine before the model has touched it -- the moment least likely to show the
    degradation this is hunting. Ruled by the gate, 2026-08-28.

    **Cached, because it cannot change and the read is not free.** See `_TOPOLOGY_CACHE`.
    """
    global _TOPOLOGY_CACHE
    if _TOPOLOGY_CACHE is not None:
        # A copy, so a caller mutating its snapshot cannot poison every later read.
        return dict(_TOPOLOGY_CACHE)

    row = _smi_row(_TOPOLOGY_FIELDS)
    if isinstance(row, str) or len(row) != len(_TOPOLOGY_KEYS):
        why = row if isinstance(row, str) else "{}: expected {} fields, got {}".format(
            UNREADABLE, len(_TOPOLOGY_KEYS), len(row))
        topology = {key: why for key in _TOPOLOGY_KEYS}
    else:
        topology = dict(zip(_TOPOLOGY_KEYS, row))
    # **Beyond RUNPOD_POD_ID, which names the worker and not the machine.** Two runs on one
    # worker id landed on hosts that behaved 1.9x apart; the GPU's own serial and uuid above are
    # the only identifiers here that follow the silicon.
    topology["host_id"] = os.environ.get("RUNPOD_POD_HOSTNAME") or UNREADABLE
    _TOPOLOGY_CACHE = topology
    return dict(topology)


#: The volatile fields, in ONE query. Current clocks and why they are where they are.
#:
#: **`clocks_throttle_reasons.active` is a hex bitmask**, not a list of names — the per-reason
#: boolean fields exist too but each is a separate column, and the whole point of this query is
#: that it costs one round trip. The mask is decoded by whoever reads the corpus; recording the
#: raw value is the honest thing anyway, since a decode table baked in here would go stale
#: against a driver nobody has seen yet.
_SAMPLE_FIELDS = ("clocks.current.sm", "clocks.current.memory",
                  "clocks_throttle_reasons.active", "utilization.gpu", "utilization.memory")

_SAMPLE_KEYS = ("clock_sm_mhz", "clock_mem_mhz", "throttle_reasons_hex",
                "util_gpu_pct", "util_mem_pct")

#: **The sampler's own budget, and a breaker rather than a timeout alone.** A 2 s timeout bounds
#: ONE read; four boundaries at 2 s each would still put 8 s inside the attempt this instrument
#: exists to measure. So the first read that costs more than `SAMPLE_BUDGET_S` disables sampling
#: for the rest of the run, and the record says it was disabled and why.
#:
#: **The number is deliberately far below anything that could matter.** The shortest phase this
#: has been observed against is postprocess at 13.8 s; a sampler allowed a tenth of that would be
#: shaping the measurement it reports.
SAMPLE_TIMEOUT_S = 2
SAMPLE_BUDGET_S = 0.25


def sample_gpu_state():
    """One volatile GPU reading. Returns `(values, elapsed_seconds)`. **Never raises.**

    **This is the half of the telemetry that measures the thing.** The one-shot half in
    `_gpu_topology` records link topology and clock ceilings, which are stable for the life of
    the container and therefore cannot show a run degrading; the two runs 1.9x apart had every
    STABLE variable identical. What differed had to be volatile, and this is the only reading of
    it — sampled where the phases already close, so encode's x2.41 and decode's x2.40 against
    postprocess's x0.83 can be checked against clocks and throttle state per phase rather than
    inferred from a wall clock.

    **It reports its own cost.** The caller records `elapsed` so the instrument's price is in the
    corpus beside its readings rather than asserted in a comment — which is what the one-shot
    half failed to do, and it shipped eight subprocesses per read into a timed phase before
    anyone noticed.
    """
    started = time.time()
    row = _smi_row(_SAMPLE_FIELDS, timeout=SAMPLE_TIMEOUT_S)
    elapsed = time.time() - started
    if isinstance(row, str) or len(row) != len(_SAMPLE_KEYS):
        why = row if isinstance(row, str) else "{}: expected {} fields, got {}".format(
            UNREADABLE, len(_SAMPLE_KEYS), len(row))
        return {key: why for key in _SAMPLE_KEYS}, elapsed
    return dict(zip(_SAMPLE_KEYS, row)), elapsed


def _round(value):
    return None if value is None else round(value, 2)


def usable_vram_gb(snapshot, reserve_gb=2.0):
    """What the estimator may plan against — free VRAM minus a reserve, or None if unknown.

    The reserve exists because free VRAM is a reading taken before the model loads, and
    allocator fragmentation, CUDA context and cuDNN workspaces all take memory that no plan
    accounts for. Planning to the last byte of a number measured at the wrong moment is how an
    estimate that says 'fits' produces an OOM at 90%.

    **Unknown is not treated as plentiful.** A `None` here sends the estimator to its floor
    configuration, because the alternative — assuming a large card — spends the job to find out.
    """
    free = snapshot.get("vram_free_gb")
    if free is None:
        return None
    return max(0.0, free - reserve_gb)
