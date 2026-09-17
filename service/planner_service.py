"""Will it fit, and how long will it run — per tier, by the worker's own planning code.

`cf-planner.md` (cf-upscale-project) is the spec. This module is the per-tier core that
`POST /estimate` projects from; `app.py` is only the door.

**Every number is the worker's own output.** What this module adds is where each input came from.
The one field it ever rewrites is `prediction_basis`, from `measured` to `borrowed`, on
`nearest_memory` only (§4).
"""

import json
import os
import re

from service import worker_path  # puts handler/ first on the import path

import estimator  # noqa: E402
import planner  # noqa: E402
import validation  # noqa: E402
from errors import CAPACITY_EXCEEDED, WorkerError  # noqa: E402

for _module in (estimator, planner, validation):
    if os.path.dirname(os.path.abspath(_module.__file__)) != worker_path.HANDLER:
        raise ImportError("{} loaded from {}, not the resolved worker at {}".format(
            _module.__name__, _module.__file__, worker_path.HANDLER))

VRAM_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vram_table.json")

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

#: The four fields `estimator.plan` reads from a hardware snapshot (§1).
HARDWARE_FIELDS = ("gpu_name", "vram_total_gb", "vram_free_gb", "host_ram_gb")


class Refusal(Exception):
    """An input the service cannot use, named. Never defaulted (§3b)."""

    def __init__(self, field, message):
        super().__init__("{}: {}".format(field, message))
        self.field = field
        self.message = message


def load_vram_table(path=VRAM_TABLE_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def service_commit(environ):
    """The commit this service runs: a full 40-hex sha, or None where it cannot be established.

    **A malformed value is a deployment error and raises** rather than reading as unknown — an
    unknown commit is a legitimate state, a mistyped one is a misconfiguration nobody would see.
    """
    value = environ.get("CF_PLANNER_COMMIT")
    if not value:
        return None
    if not _FULL_SHA.match(value):
        raise ValueError("CF_PLANNER_COMMIT {!r} is not a full 40-hex sha".format(value))
    return value


# ------------------------------------------------------------------------------------------------
# The door's checks. Each refuses by the field's full name.
# ------------------------------------------------------------------------------------------------

def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_number(body, key, where):
    if key not in body or body[key] is None:
        raise Refusal("{}.{}".format(where, key), "required")
    if not _number(body[key]) or body[key] <= 0:
        raise Refusal("{}.{}".format(where, key), "must be a positive number")
    return body[key]


def _positive_int(body, key, where):
    value = body.get(key)
    if key not in body or value is None:
        raise Refusal("{}.{}".format(where, key), "required")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise Refusal("{}.{}".format(where, key), "must be a positive integer")
    return value


def _object(body, key, where, nullable=False):
    if key not in body:
        raise Refusal("{}.{}".format(where, key), "required" + (" (null allowed)" if nullable
                                                                  else ""))
    value = body[key]
    if value is None and nullable:
        return None
    if not isinstance(value, dict):
        raise Refusal("{}.{}".format(where, key), "must be an object")
    return value


def _gpu_name(body, where):
    value = body.get("gpu_name")
    if not isinstance(value, str) or not value:
        raise Refusal("{}.gpu_name".format(where), "required, a non-empty string")
    return value


def _optional_number(body, key, where):
    value = body.get(key)
    if value is None:
        return None
    if not _number(value) or value <= 0:
        raise Refusal("{}.{}".format(where, key), "must be a positive number when present")
    return value


def read_job(body):
    job = _object(body, "job", "request")
    width = _positive_int(job, "source_width", "job")
    height = _positive_int(job, "source_height", "job")
    frames = _positive_int(job, "frames", "job")
    if not isinstance(job.get("is_still"), bool):
        raise Refusal("job.is_still", "required, a boolean")
    still = job["is_still"]
    if still and frames != 1:
        raise Refusal("job.frames", "an image source is 1 frame; is_still is true with frames "
                                    "{}".format(frames))

    has_edge = job.get("target_short_edge_px") is not None
    has_canvas = job.get("output_size") is not None
    if has_edge and has_canvas:
        raise Refusal("job.output_size", "exactly one of target_short_edge_px and output_size; "
                                         "both were sent")
    if not has_edge and not has_canvas:
        raise Refusal("job.target_short_edge_px", "exactly one of target_short_edge_px and "
                                                  "output_size; neither was sent")
    canvas = None
    if has_canvas:
        size = _object(job, "output_size", "job")
        canvas = (_positive_int(size, "width", "job.output_size"),
                  _positive_int(size, "height", "job.output_size"))
        target = estimator.short_edge_covering(width, height, *canvas)
    else:
        target = _positive_int(job, "target_short_edge_px", "job")

    tile_quality = job.get("tile_quality", validation.DEFAULT_TILE_QUALITY)
    if tile_quality not in validation.TILE_QUALITIES:
        raise Refusal("job.tile_quality", "must be one of {}".format(
            ", ".join(validation.TILE_QUALITIES)))
    schedule = job.get("schedule", validation.DEFAULT_SCHEDULE)
    if schedule not in validation.SCHEDULES:
        raise Refusal("job.schedule", "must be one of {}".format(", ".join(validation.SCHEDULES)))

    return {
        "source_width": width, "source_height": height, "frames": frames, "still": still,
        "target": target, "canvas": canvas, "tile_quality": tile_quality, "schedule": schedule,
    }


def read_tier(body, index):
    where = "tiers[{}]".format(index)
    if not isinstance(body, dict):
        raise Refusal(where, "must be an object")
    if "tier" not in body or body["tier"] is None:
        raise Refusal("{}.tier".format(where), "required")
    host_ram_gb = _positive_number(body, "host_ram_gb", where)

    cards = body.get("cards")
    if not isinstance(cards, list) or not cards:
        raise Refusal("{}.cards".format(where), "required, a non-empty list")
    read_cards = []
    for n, card in enumerate(cards):
        card_where = "{}.cards[{}]".format(where, n)
        if not isinstance(card, dict):
            raise Refusal(card_where, "must be an object")
        read_cards.append({"gpu_name": _gpu_name(card, card_where),
                           "vram_total_gb": _positive_number(card, "vram_total_gb", card_where)})

    if "worker_commit" not in body:
        raise Refusal("{}.worker_commit".format(where), "required (null allowed)")
    worker_commit = body["worker_commit"]
    if worker_commit is not None and (not isinstance(worker_commit, str)
                                      or not _FULL_SHA.match(worker_commit)):
        raise Refusal("{}.worker_commit".format(where), "must be a full 40-hex sha, or null")

    idle = _object(body, "idle", where, nullable=True)
    if idle is not None:
        idle_where = "{}.idle".format(where)
        idle = {"gpu_name": _gpu_name(idle, idle_where),
                **{k: _optional_number(idle, k, idle_where)
                   for k in ("vram_total_gb", "vram_free_gb", "host_ram_gb")}}

    last_run = _object(body, "last_run", where, nullable=True)
    if last_run is not None:
        last_where = "{}.last_run".format(where)
        last_run = {"gpu_name": _gpu_name(last_run, last_where),
                    **{k: _optional_number(last_run, k, last_where)
                       for k in ("vram_total_gb", "vram_free_gb")}}

    return {"tier": body["tier"], "host_ram_gb": host_ram_gb, "cards": read_cards,
            "worker_commit": worker_commit, "idle": idle, "last_run": last_run}


# ------------------------------------------------------------------------------------------------
# §4: resolving the four hardware fields from what CF sent.
# ------------------------------------------------------------------------------------------------

def nearest_measured(nominal_gb, table_cards):
    """The measured card closest in total memory; **an exact tie takes the lower memory**."""
    return min(table_cards, key=lambda name: (abs(table_cards[name]["vram_total_gb"] - nominal_gb),
                                              table_cards[name]["vram_total_gb"], name))


def _both(reading):
    return (reading is not None and reading.get("vram_total_gb") is not None
            and reading.get("vram_free_gb") is not None)


def resolve_hardware(tier, table_cards, index):
    where = "tiers[{}]".format(index)
    idle, last_run = tier["idle"], tier["last_run"]

    host_ram_gb = (idle or {}).get("host_ram_gb") or tier["host_ram_gb"]

    if idle is not None:
        card, card_source = idle["gpu_name"], "idle"
        # Nominal memory from cards[] (the least, where the name repeats), else the idle
        # worker's own total; neither is a refusal, but only if resolution gets that far (§4).
        named = [c["vram_total_gb"] for c in tier["cards"] if c["gpu_name"] == card]
        nominal = min(named) if named else idle.get("vram_total_gb")
    else:
        # The first card holding the least nominal memory, in the order CF sent. Its nominal is
        # THAT entry's, never another entry's under the same name.
        worst = min(tier["cards"], key=lambda c: c["vram_total_gb"])
        card, nominal, card_source = worst["gpu_name"], worst["vram_total_gb"], "worst_card"

    resolved_from = None
    if card_source == "idle" and _both(idle):
        planned_name, total, free, vram_source = card, idle["vram_total_gb"], idle[
            "vram_free_gb"], "idle"
    elif _both(last_run) and last_run["gpu_name"] == card:
        planned_name, total, free, vram_source = card, last_run["vram_total_gb"], last_run[
            "vram_free_gb"], "last_run"
    elif card in table_cards:
        planned_name, vram_source = card, "table"
        total, free = table_cards[card]["vram_total_gb"], table_cards[card]["vram_free_gb"]
    else:
        if nominal is None:
            raise Refusal("{}.idle.gpu_name".format(where),
                          "{!r} is not in cards[] and carries no VRAM of its own, so it has no "
                          "nominal memory to resolve against".format(card))
        planned_name = nearest_measured(nominal, table_cards)
        total = table_cards[planned_name]["vram_total_gb"]
        free = table_cards[planned_name]["vram_free_gb"]
        vram_source = "nearest_memory"
        resolved_from = {"card": card, "measured": planned_name}

    return {
        "hardware": {"gpu_name": planned_name, "vram_total_gb": total, "vram_free_gb": free,
                     "host_ram_gb": host_ram_gb},
        "card_source": card_source,
        "vram_source": vram_source,
        "resolved_from": resolved_from,
    }


# ------------------------------------------------------------------------------------------------
# The core.
# ------------------------------------------------------------------------------------------------

def _plan_tier(job, resolved):
    """One tier through `estimator.plan`, unchanged. Returns the core entry's planning fields."""
    snapshot = resolved["hardware"]
    worker_job = {
        "target_short_edge_px": job["target"],
        "source_width": job["source_width"],
        "source_height": job["source_height"],
        "estimated_frames": job["frames"],
        "still": job["still"],
        "tile_quality": job["tile_quality"],
        "schedule": job["schedule"],
    }
    if job["canvas"] is not None:
        delivered = job["canvas"]
    else:
        delivered = estimator.output_dimensions(job["source_width"], job["source_height"],
                                                job["target"])
    frames = 1 if job["still"] else job["frames"]
    try:
        _chosen, rationale = estimator.plan(worker_job, snapshot)
    except WorkerError as refusal:
        if refusal.code != CAPACITY_EXCEEDED:
            raise
        # §3b: estimator.plan carries no residency on a refusal. The same verdict, read from
        # planner.plan called with estimator's own arguments (estimator.py, the planner.plan call).
        verdict = planner.plan(
            (job["source_width"], job["source_height"]), frames, job["target"],
            usable_gb=estimator._usable_vram(snapshot), host_ram_gb=snapshot.get("host_ram_gb"),
            tile_quality=job["tile_quality"], schedule=job["schedule"],
            gpu_name=snapshot.get("gpu_name"))
        # **The same verdict, checked rather than assumed** — a second call that refused on another
        # constraint would pair this refusal's reason with that one's residency.
        if (verdict.get("action") == "plan"
                or estimator._refusal_text(verdict.get("reason") or "") != refusal.message):
            raise RuntimeError("planner.plan did not reach the verdict estimator.plan refused on")
        return {
            "fits": False, "predicted_seconds": None, "reason": refusal.message,
            "residency": verdict.get("residency"),
            "output_width": delivered[0], "output_height": delivered[1],
            "best_window": None, "ideal_window": planner.ideal_window(frames),
            "binding_phase": None, "anchored": None, "prediction_basis": None,
            "rationale": None, "planner_verdict": verdict,
        }
    return {
        "fits": True,
        "predicted_seconds": rationale.get("predicted_seconds"),
        "reason": None,
        "residency": rationale.get("residency"),
        "output_width": delivered[0], "output_height": delivered[1],
        "best_window": rationale.get("window"),
        "ideal_window": rationale.get("ideal_window"),
        "binding_phase": rationale.get("binding_phase"),
        "anchored": rationale.get("anchored"),
        "prediction_basis": rationale.get("prediction_basis"),
        "rationale": rationale, "planner_verdict": None,
    }


def estimate_core(body, commit, vram_table=None):
    """Every tier's core entry, in the order sent. Raises `Refusal` for an unusable input.

    The whole request is read before anything is planned, so a refusal on the last tier costs no
    planning on the first.
    """
    if not isinstance(body, dict):
        raise Refusal("request", "must be a JSON object")
    table_cards = (vram_table or load_vram_table())["cards"]
    job = read_job(body)
    tiers = body.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise Refusal("tiers", "required, a non-empty list")
    read = [read_tier(t, i) for i, t in enumerate(tiers)]
    resolved = [resolve_hardware(t, table_cards, i) for i, t in enumerate(read)]

    entries = []
    for tier, where in zip(read, resolved):
        planned = _plan_tier(job, where)
        if where["vram_source"] == "nearest_memory" and planned["prediction_basis"] == "measured":
            # §4: the one label the service rewrites. The rate is the measured card's, not this one's.
            planned["prediction_basis"] = "borrowed"
        worker_commit = tier["worker_commit"]
        entries.append(dict(
            planned,
            tier=tier["tier"],
            planned_short_edge_px=job["target"],
            hardware_used=dict(where["hardware"]),
            card_source=where["card_source"],
            vram_source=where["vram_source"],
            resolved_from=where["resolved_from"],
            registry_version=planner.REGISTRY_VERSION,
            commit=commit,
            commit_match=(None if worker_commit is None or commit is None
                          else commit == worker_commit),
        ))
    return entries


#: §3b, and nothing else goes on the wire.
WIRE_FIELDS = ("tier", "fits", "predicted_seconds", "reason", "residency", "output_width",
               "output_height", "best_window", "ideal_window", "binding_phase", "anchored",
               "prediction_basis", "hardware_used", "card_source", "vram_source", "resolved_from",
               "registry_version", "commit", "commit_match")


def project(entry):
    return {field: entry[field] for field in WIRE_FIELDS}


def version(commit, vram_table=None):
    table = vram_table or load_vram_table()
    return {
        "commit": commit,
        "registry_version": planner.REGISTRY_VERSION,
        "calibration_rows": len(estimator.load_calibration()),
        "vram_table_corpus": table["corpus"],
    }
