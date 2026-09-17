"""Will it fit, and how long will it run — per CARD, by the worker's own planning code.

`cf-planner.md` (cf-upscale-project) is the spec. This module is the per-tier core that
`POST /estimate` projects from; `app.py` is only the door.

**One ordered `cards[]` in, one answer per card out, in CF's order** (§3a, §4). The service
reorders nothing, chooses no card and ranks nothing: which card CF expects, and what it pays for
placement risk, is CF's policy.

**Every number is the worker's own output.** What this module adds is where each input came from.
The one field it ever rewrites is `prediction_basis`, from `measured` to `borrowed`, wherever the
planned card is not the card CF named — `nearest_memory` and `pool_floor` (§4a-i).
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

HERE = os.path.dirname(os.path.abspath(__file__))
VRAM_TABLE_PATH = os.path.join(HERE, "vram_table.json")
HANDLER_HISTORY_PATH = os.path.join(HERE, "handler_history.json")

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


class TableUnusable(RuntimeError):
    """The service's own committed table cannot answer. **A deploy fault, never the caller's** —
    §4's refusals are input faults, and a 400 naming `service.vram_table` would tell CF it sent
    something wrong about a file CF has never seen."""


class Refusal(Exception):
    """An input the service cannot use, named. Never defaulted (§4)."""

    def __init__(self, field, message):
        super().__init__("{}: {}".format(field, message))
        self.field = field
        self.message = message


def load_vram_table(path=VRAM_TABLE_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_handler_history(path=HANDLER_HISTORY_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def service_handler_tree(history):
    """The handler/ tree this service plans with: the history table's newest entry (§5).

    The kit holds that this is HEAD's tree — no commit after `newest` touches handler/.
    """
    return history["commits"][history["newest"]]


def service_commit(environ):
    """The commit this service runs: a full 40-hex sha, or None where none can be established.

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


def _positive_int(body, key, where):
    value = body.get(key)
    if key not in body or value is None:
        raise Refusal("{}.{}".format(where, key), "required")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise Refusal("{}.{}".format(where, key), "must be a positive integer")
    return value


def _optional_number(body, key, where):
    value = body.get(key)
    if value is None:
        return None
    if not _number(value) or value <= 0:
        raise Refusal("{}.{}".format(where, key), "must be a positive number when present")
    return value


def _object(body, key, where):
    if key not in body or body[key] is None:
        raise Refusal("{}.{}".format(where, key), "required")
    if not isinstance(body[key], dict):
        raise Refusal("{}.{}".format(where, key), "must be an object")
    return body[key]


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
    host_ram_gb = _optional_number(body, "host_ram_gb", where)

    if "worker_commit" not in body:
        raise Refusal("{}.worker_commit".format(where), "required (null allowed)")
    worker_commit = body["worker_commit"]
    if worker_commit is not None and (not isinstance(worker_commit, str)
                                      or not _FULL_SHA.match(worker_commit)):
        raise Refusal("{}.worker_commit".format(where), "must be a full 40-hex sha, or null")

    cards = body.get("cards")
    if not isinstance(cards, list) or not cards:
        raise Refusal("{}.cards".format(where), "required, a non-empty ordered list")
    read_cards = []
    for n, entry in enumerate(cards):
        card_where = "{}.cards[{}]".format(where, n)
        if not isinstance(entry, dict):
            raise Refusal(card_where, "must be an object")
        gpu_name = entry.get("gpu_name")
        if not isinstance(gpu_name, str) or not gpu_name:
            raise Refusal("{}.gpu_name".format(card_where), "required, a non-empty string")
        label = entry.get("label")
        if label is not None and not isinstance(label, str):
            raise Refusal("{}.label".format(card_where), "must be a string when present")
        card = {"gpu_name": gpu_name, "label": label,
                **{k: _optional_number(entry, k, card_where)
                   for k in ("vram_total_gb", "vram_free_gb", "host_ram_gb")}}
        # **Host RAM is per card, and neither source refuses THAT CARD by name** (§4): the host
        # slice decides the chunk, and there is nothing conservative to assume.
        card["host_ram_used_gb"] = card["host_ram_gb"] or host_ram_gb
        if card["host_ram_used_gb"] is None:
            raise Refusal("{}.host_ram_gb".format(card_where),
                          "no host RAM for this card: neither the card nor the tier carries one")
        read_cards.append(card)

    return {"tier": body["tier"], "host_ram_gb": host_ram_gb, "cards": read_cards,
            "worker_commit": worker_commit}


# ------------------------------------------------------------------------------------------------
# §4a-i: each card's VRAM, the first source that covers THAT card.
# ------------------------------------------------------------------------------------------------

def nearest_measured(nominal_gb, table_cards):
    """The measured card closest in total memory; **an exact tie takes the lower memory**."""
    return min(table_cards, key=lambda name: (abs(table_cards[name]["vram_total_gb"] - nominal_gb),
                                              table_cards[name]["vram_total_gb"], name))


def _copy_stats(row):
    """A COPY of the table row's statistics. A live reference would be process-global shared
    mutable state the moment anyone caches the table, which `vram_table=` exists to allow."""
    stats = row.get("stats")
    return dict(stats) if isinstance(stats, dict) else None


def _worst_measured(names, table_cards):
    """The worst card the table measures among `names`, or None where it measures none."""
    measured = [name for name in names if name in table_cards]
    if not measured:
        return None
    return min(measured, key=lambda name: (table_cards[name]["vram_total_gb"], name))


def resolve_card(card, siblings, table_cards):
    """The four fields to plan this card against, and where each came from (§4, §4a-i)."""
    name = card["gpu_name"]
    total, free = card["vram_total_gb"], card["vram_free_gb"]

    if total is not None and free is not None:
        # A reading, used as sent. **A free without a total is not one**, and a total without a
        # free is a nominal — neither is planned against directly.
        planned_name, vram_source, resolved_from, stats = name, "given", None, None
    elif name in table_cards:
        planned_name, vram_source, resolved_from = name, "table", None
        total, free = table_cards[name]["vram_total_gb"], table_cards[name]["vram_free_gb"]
        stats = _copy_stats(table_cards[name])
    elif total is not None:
        # A nominal: it SELECTS a measured card and is never planned against.
        planned_name = nearest_measured(total, table_cards)
        vram_source, resolved_from = "nearest_memory", {"card": name, "measured": planned_name}
        total, free = (table_cards[planned_name]["vram_total_gb"],
                       table_cards[planned_name]["vram_free_gb"])
        stats = _copy_stats(table_cards[planned_name])
    else:
        # **`pool_floor`, deliberately pessimistic** (§4a-i): the worst measured card in THIS
        # list, else the table's worst. An unnamed card is usually better than the list
        # advertised, so the floor under-promises — the ruled error direction.
        planned_name = (_worst_measured([c["gpu_name"] for c in siblings], table_cards)
                        or _worst_measured(list(table_cards), table_cards))
        vram_source, resolved_from = "pool_floor", {"card": name, "measured": planned_name}
        total, free = (table_cards[planned_name]["vram_total_gb"],
                       table_cards[planned_name]["vram_free_gb"])
        # §3b lists vram_stats on table and nearest_memory; pool_floor names its card instead.
        stats = None

    return {
        "hardware": {"gpu_name": planned_name, "vram_total_gb": total, "vram_free_gb": free,
                     "host_ram_gb": card["host_ram_used_gb"]},
        "vram_source": vram_source,
        "vram_stats": stats,
        "resolved_from": resolved_from,
    }


# ------------------------------------------------------------------------------------------------
# The core.
# ------------------------------------------------------------------------------------------------

#: §3d — window, tail and tiling. The chunk and the blocks swapped are scheduling facts and say
#: nothing about quality, so they are not on the wire.
QUALITY_FROM_RATIONALE = (
    ("best_window", "window"), ("ideal_window", "ideal_window"), ("passes", "passes"),
    ("shortest_pass", "shortest_pass"), ("decode_grid", "decode_grid"),
    ("decode_tile", "decode_tile"), ("encode_grid", "encode_grid"),
    ("encode_tile", "encode_tile"),
)


def _quality_of_refusal(frames):
    """A refused card still says what the CLIP could use (§3b, ruled on C16).

    `ideal_window` is a property of the clip rather than of the card, so a tier walk that refuses
    everywhere still learns it; the rest of §3d needs a plan and stays null.
    """
    return {field: (planner.ideal_window(frames) if field == "ideal_window" else None)
            for field, _key in QUALITY_FROM_RATIONALE}


def _plan_card(job, snapshot):
    """One card through `estimator.plan`, unchanged."""
    worker_job = {
        "target_short_edge_px": job["target"],
        "source_width": job["source_width"],
        "source_height": job["source_height"],
        "estimated_frames": job["frames"],
        "still": job["still"],
        "tile_quality": job["tile_quality"],
        "schedule": job["schedule"],
    }
    frames = 1 if job["still"] else job["frames"]
    try:
        _chosen, rationale = estimator.plan(worker_job, snapshot)
    except WorkerError as refusal:
        if refusal.code != CAPACITY_EXCEEDED:
            raise
        # §3b: estimator.plan carries no residency on a refusal. The same verdict, read from
        # planner.plan called with estimator's own arguments.
        usable = estimator._usable_vram(snapshot)
        verdict = planner.plan(
            (job["source_width"], job["source_height"]), frames, job["target"],
            usable_gb=usable, host_ram_gb=snapshot.get("host_ram_gb"),
            tile_quality=job["tile_quality"], schedule=job["schedule"],
            gpu_name=snapshot.get("gpu_name"))
        # **The same verdict, checked rather than assumed** — a second call that refused on
        # another constraint would pair this refusal's reason with that one's residency.
        if (verdict.get("action") == "plan"
                or estimator._refusal_text(verdict.get("reason") or "") != refusal.message):
            raise RuntimeError("planner.plan did not reach the verdict estimator.plan refused on")
        return {
            "fits": False, "predicted_seconds": None, "prediction_basis": None,
            "reason": refusal.message,
            # P1d/P1a (§3b): both labels as planner.fits reads a refusal (planner.py, fits).
            "residency": verdict.get("residency", planner.ROUTE_UP),
            "anchored": usable <= planner.ANCHORED_MAX_USABLE,
            "binding_phase": None, "quality": _quality_of_refusal(frames),
            "rate_from": None,
            "rationale": None, "planner_verdict": verdict,
        }
    return {
        "fits": True,
        "predicted_seconds": rationale.get("predicted_seconds"),
        "prediction_basis": rationale.get("prediction_basis"),
        "reason": None,
        "residency": rationale.get("residency"),
        "anchored": rationale.get("anchored"),
        "binding_phase": rationale.get("binding_phase"),
        "quality": {field: rationale.get(key) for field, key in QUALITY_FROM_RATIONALE},
        # §4a-ii: whose measurement the time is. The worker's own two keys, carried through
        # unchanged — null means the rate is this card's own rows.
        "rate_from": _rate_from(rationale),
        "rationale": rationale, "planner_verdict": None,
    }


def _rate_from(rationale):
    """`timing_from_another_card`, plus the tiling where that differs too (§4a-ii).

    **THE SAME NUMBER COMES BACK ON EVERY CARD IN A TIER** when only one card has rows in the
    pixel band — `estimator._attach_timing` falls back to `same_card or comparable` — and three
    identical numbers side by side look like three measurements. This is what says they are not.
    """
    other_card = rationale.get("timing_from_another_card")
    other_tiling = rationale.get("timing_from_another_tiling")
    if not other_card and not other_tiling:
        return None
    rate_from = dict(other_card or {})
    if other_tiling:
        rate_from["tiling"] = dict(other_tiling)
    return rate_from


def estimate_core(body, commit, vram_table=None):
    """Every tier's answer, in the order sent, each with one answer per card.

    The whole request is read before anything is planned, so a refusal on the last tier costs no
    planning on the first.
    """
    if not isinstance(body, dict):
        raise Refusal("request", "must be a JSON object")
    job = read_job(body)
    tiers = body.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise Refusal("tiers", "required, a non-empty list")
    read = [read_tier(t, i) for i, t in enumerate(tiers)]

    # Tables after the door: a malformed request is refused by name whatever state they are in.
    table_cards = (vram_table or load_vram_table())["cards"]
    if not table_cards:
        # Before any card is planned: every fallback below ends in a measured card.
        raise TableUnusable("the service's VRAM table measures no card")
    history = load_handler_history()
    handler_tree = service_handler_tree(history)

    if job["canvas"] is not None:
        delivered = job["canvas"]
    else:
        delivered = estimator.output_dimensions(job["source_width"], job["source_height"],
                                                job["target"])

    entries = []
    for tier in read:
        answers = []
        for card in tier["cards"]:
            resolved = resolve_card(card, tier["cards"], table_cards)
            planned = _plan_card(job, resolved["hardware"])
            if resolved["resolved_from"] and planned["prediction_basis"] == "measured":
                # §4a-i: the one field the service rewrites, and only for the MEMORY sense. The
                # time sense needs no rewrite — the worker already labels a rate from another
                # card or another tiling `borrowed` itself, and `rate_from` beside it says which
                # card and which tiling. A condition on rate_from here would never fire, and an
                # unfirable condition is one no witness can hold.
                planned["prediction_basis"] = "borrowed"
            answers.append(dict(
                planned,
                gpu_name=card["gpu_name"],
                label=card["label"],
                hardware_used=dict(resolved["hardware"]),
                vram_source=resolved["vram_source"],
                vram_stats=resolved["vram_stats"],
                resolved_from=resolved["resolved_from"],
            ))
        worker_commit = tier["worker_commit"]
        # §5: matched on handler/'s tree. A commit the table does not hold is unknown, never a
        # mismatch.
        worker_tree = (None if worker_commit is None
                       else history["commits"].get(worker_commit))
        entries.append({
            "tier": tier["tier"],
            "cards": answers,
            "fits_any": any(a["fits"] for a in answers),
            "fits_all": all(a["fits"] for a in answers),
            "output_width": delivered[0],
            "output_height": delivered[1],
            "planned_short_edge_px": job["target"],
            "registry_version": planner.REGISTRY_VERSION,
            "commit": commit,
            "handler_tree": handler_tree,
            "worker_handler_tree": worker_tree,
            "handler_match": (None if worker_tree is None or handler_tree is None
                              else worker_tree == handler_tree),
        })
    return entries


#: §3b, and nothing else goes on the wire.
TIER_WIRE_FIELDS = ("tier", "fits_any", "fits_all", "output_width", "output_height",
                    "registry_version", "commit", "handler_tree", "worker_handler_tree",
                    "handler_match")
CARD_WIRE_FIELDS = ("gpu_name", "label", "fits", "predicted_seconds", "prediction_basis",
                    "rate_from", "reason", "residency", "anchored", "binding_phase", "quality",
                    "hardware_used", "vram_source", "vram_stats", "resolved_from")


def project(entry):
    wire = {field: entry[field] for field in TIER_WIRE_FIELDS}
    wire["cards"] = [{field: answer[field] for field in CARD_WIRE_FIELDS}
                     for answer in entry["cards"]]
    return wire


def version(commit, vram_table=None):
    table = vram_table or load_vram_table()
    history = load_handler_history()
    return {
        "commit": commit,
        "handler_tree": service_handler_tree(history),
        "handler_history_newest": history["newest"],
        "registry_version": planner.REGISTRY_VERSION,
        "calibration_rows": len(estimator.load_calibration()),
        "vram_table_corpus": table["corpus"],
    }
