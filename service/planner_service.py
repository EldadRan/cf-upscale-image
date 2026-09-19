"""Will it fit, and how long will it run — per CARD, by the worker's own planning code.

`cf-planner.md` (cf-upscale-project) is the spec. This module is the per-tier core that
`POST /estimate` projects from; `app.py` is only the door.

**One ordered `cards[]` in, one answer per card out, in CF's order** (§3a, §4). The service
reorders nothing, chooses no card and ranks nothing: which card CF expects, and what it pays for
placement risk, is CF's policy.

**Every number is the worker's own output.** What this module adds is where each input came from.
Memory is planned against the card §4a-i resolves; **time is judged by the worker itself on the
name CF SENT** (§4a-ii, W6), so a card the card table holds is priced at its own rate whatever its
memory was resolved to, and J5's and F4's absences name CF's card with no rewrite here. The one
field this module rewrites is `prediction_basis`, `measured` to `borrowed`, wherever the planned
card is not the one CF named — `nearest_memory` and `pool_floor` (§4a-i, C14): `borrowed` covers
both senses, and `resolved_from` beside it says it was the memory.
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
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


class TableUnusable(RuntimeError):
    """The in-image card table cannot answer for memory. **A deploy fault, never the caller's** —
    §4's refusals are input faults, and a 400 naming the table would tell CF it sent something
    wrong about a file CF has never seen."""


class Refusal(Exception):
    """An input the service cannot use, named. Never defaulted (§4)."""

    def __init__(self, field, message):
        super().__init__("{}: {}".format(field, message))
        self.field = field
        self.message = message


def startup_check():
    """Raise `TableUnusable` unless the in-image card table loads and gives some card's memory.

    **Before the port opens** (app.main). `estimator.load_card_table` returns None on a missing
    or malformed file rather than raising — right for the worker, which must keep running — so
    without this a broken table would deploy, start, and answer 503 to every request.
    """
    if not memory_cards(estimator.load_card_table()):
        raise TableUnusable("the in-image card table is absent, malformed, or gives no card's "
                            "memory")


def memory_cards(card_table):
    """`{gpu_name: {vram_total_gb, vram_free_gb}}` — the cards the card table gives BOTH memory
    figures, a copy (W6: the memory half of the one curated table). **A card with a rate and no
    memory is not measured for memory**, and resolves like any other card the table lacks.

    **The plan uses these figures, and nothing else rides with them**: `vram_stats` was dropped by
    CF on 2026-09-19 (W6 Q4) — nothing read it, and §4a always said the statistics were
    information, not an input.
    """
    if not card_table or estimator.card_table_problem(card_table):
        return {}
    return {name: {"vram_total_gb": entry["vram_total_gb"],
                   "vram_free_gb": entry["vram_free_gb"]}
            for name, entry in card_table["cards"].items()
            if entry.get("vram_total_gb") is not None and entry.get("vram_free_gb") is not None}


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
    # **`worker_commit` is not read** (W6 item 3, ruled by CF 2026-09-19): the service no longer
    # compares commits, and a caller still sending one has it ignored, whatever it holds. CF
    # compares its pinned image tag with /version's `commit` itself.

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

    return {"tier": body["tier"], "host_ram_gb": host_ram_gb, "cards": read_cards}


# ------------------------------------------------------------------------------------------------
# §4a-i: each card's VRAM, the first source that covers THAT card.
# ------------------------------------------------------------------------------------------------

def nearest_measured(nominal_gb, table_cards):
    """The measured card closest in total memory; **an exact tie takes the lower memory**."""
    return min(table_cards, key=lambda name: (abs(table_cards[name]["vram_total_gb"] - nominal_gb),
                                              table_cards[name]["vram_total_gb"], name))


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
        planned_name, vram_source, resolved_from = name, "given", None
    elif name in table_cards:
        planned_name, vram_source, resolved_from = name, "table", None
        total, free = table_cards[name]["vram_total_gb"], table_cards[name]["vram_free_gb"]
    elif total is not None:
        # A nominal: it SELECTS a measured card and is never planned against.
        planned_name = nearest_measured(total, table_cards)
        vram_source, resolved_from = "nearest_memory", {"card": name, "measured": planned_name}
        total, free = (table_cards[planned_name]["vram_total_gb"],
                       table_cards[planned_name]["vram_free_gb"])
    else:
        # **`pool_floor`, deliberately pessimistic** (§4a-i): the worst measured card in THIS
        # list, else the table's worst. An unnamed card is usually better than the list
        # advertised, so the floor under-promises — the ruled error direction.
        planned_name = (_worst_measured([c["gpu_name"] for c in siblings], table_cards)
                        or _worst_measured(list(table_cards), table_cards))
        vram_source, resolved_from = "pool_floor", {"card": name, "measured": planned_name}
        total, free = (table_cards[planned_name]["vram_total_gb"],
                       table_cards[planned_name]["vram_free_gb"])

    return {
        "hardware": {"gpu_name": planned_name, "vram_total_gb": total, "vram_free_gb": free,
                     "host_ram_gb": card["host_ram_used_gb"]},
        "vram_source": vram_source,
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


def _plan_card(job, snapshot, card_table, timed_as):
    """One card through `estimator.plan`, unchanged — memory against `snapshot`, TIME under
    `timed_as`, the name CF sent (§4a-ii). **A memory substitution is not a time substitution**:
    planning under the substitute's name priced a card the table DOES hold at the substitute's
    rate, hidden only while the two tables covered the same cards (W6)."""
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
        _chosen, rationale = estimator.plan(worker_job, snapshot, card_table=card_table,
                                            timed_as=timed_as)
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
            "max_target": _max_target(refusal, job, snapshot),
            # P1d/P1a (§3b): both labels as planner.fits reads a refusal (planner.py, fits).
            "residency": verdict.get("residency", planner.ROUTE_UP),
            "anchored": usable <= planner.ANCHORED_MAX_USABLE,
            "binding_phase": None, "quality": _quality_of_refusal(frames),
            "rate_from": None,
            "timing_unavailable": None,
            "rationale": None, "planner_verdict": verdict,
        }
    return {
        "fits": True,
        # §3e: only a refusal has a largest-that-plans to report.
        "max_target": None,
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
        # J5: a card no row was measured on gets no time and says so — a named absence. The
        # card still fits and still has quality; a null time is not a refusal.
        "timing_unavailable": rationale.get("timing_unavailable"),
        "rationale": rationale, "planner_verdict": None,
    }


#: The worker's own option, `resend with target_short_edge_px=N` (estimator._terminal_options).
_REDUCE_HOW = re.compile(r"target_short_edge_px=(\d+)$")


def _max_target(refusal, job, snapshot):
    """§3e: the largest target that plans on this card, or None when nothing smaller does.

    **THE WORKER'S OWN WALK, NOT A SECOND ONE.** `estimator.plan`'s capacity refusal already
    carries `_terminal_options`' answer in its shortfall — down the 32 px grid, even-snapped,
    floor 64, `planner.plan` in a loop — so the service reads it rather than walking again. The
    one number is taken from the worker's option, and the rest is re-derived with the walk's own
    arguments and checked against the worker's sentence: **a changed option fails the answer
    rather than guessing.** Reported, never taken; an `output_size` request gets a short edge.
    """
    shortfall = refusal.shortfall or {}
    # **A moved list or a renamed option is loud too** (review, P2): read as null it would say
    # "nothing smaller plans" on the wire, which is a false answer, not a missing one.
    if "options" not in shortfall:
        raise RuntimeError("the worker's capacity refusal carries no options list")
    options = [o for o in shortfall["options"] or []
               if o.get("option") == "reduce_target_resolution"]
    if shortfall["options"] and not options:
        raise RuntimeError("the worker's options carry no reduce_target_resolution: {!r}".format(
            shortfall["options"]))
    if not options:
        return None
    matched = _REDUCE_HOW.search(options[0].get("how") or "")
    if not matched:
        raise RuntimeError("the worker's reduce_target_resolution option changed shape: {!r}"
                           .format(options[0]))
    edge = int(matched.group(1))
    width, height = job["source_width"], job["source_height"]
    out_w, out_h = estimator.output_dimensions(width, height, edge)
    # **Exactly the walk's call** (estimator._terminal_options): its frames, its usable VRAM,
    # and no tile_quality or schedule — the window is the one the worker's sentence names.
    answer = planner.plan(
        (width, height), max(1, int(job["frames"] or 1)), edge,
        usable_gb=estimator._usable_vram(snapshot), host_ram_gb=snapshot.get("host_ram_gb"),
        gpu_name=snapshot.get("gpu_name"))
    if answer.get("action") != "plan" or "delivers {}x{} ".format(out_w, out_h) not in \
            options[0].get("cost", "") or \
            "at a window of {} frames".format(answer["w"]) not in options[0].get("cost", ""):
        raise RuntimeError("the worker's smaller target does not re-derive: {!r}".format(
            options[0]))
    return {"target_short_edge_px": edge, "output_width": out_w, "output_height": out_h,
            "best_window": answer["w"]}


def _rate_from(rationale):
    """`timing_from_another_card`, plus the tiling where that differs too (§4a-ii).

    **THE SAME NUMBER CAN COME BACK ON SEVERAL CARDS IN A TIER** when they borrow one lender's
    rate, and identical numbers side by side look like several measurements. This is what says
    they are not.
    """
    other_card = rationale.get("timing_from_another_card")
    other_tiling = rationale.get("timing_from_another_tiling")
    if not other_card and not other_tiling:
        return None
    rate_from = dict(other_card or {})
    if other_tiling:
        rate_from["tiling"] = dict(other_tiling)
    return rate_from


def estimate_core(body, commit, card_table=None):
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
    # **ONE card table for the whole request, both halves** (W6): memory resolves against it
    # here, and the worker times every card against the same object below. **The body's
    # `card_table` is that table when sent and valid** (item 2) — the worker's own resolution,
    # so a table CF sends to both surfaces makes them agree by construction.
    #
    # **KNOWN, TEMPORARY SILENCE:** a refused override falls back to the in-image table and the
    # warning is DROPPED here, until CF rules the /estimate `warnings` field (Q5b, 2026-09-19).
    # The worker says it; the service cannot yet.
    sent = card_table is None and "card_table" in body
    if card_table is None:
        card_table, _unsaid = estimator.resolve_card_table(body.get("card_table"),
                                                           "card_table" in body)
        sent = sent and _unsaid is None
    table_cards = memory_cards(card_table)
    if not table_cards and sent:
        # The caller's valid table carries no memory at all: the caller's input, named.
        raise Refusal("card_table", "carries no card's vram_total_gb and vram_free_gb, so no "
                                    "card without a reading can be resolved")
    if not table_cards:
        # Before any card is planned: every fallback below ends in a measured card.
        raise TableUnusable("the card table gives no card's memory")

    if job["canvas"] is not None:
        delivered = job["canvas"]
    else:
        delivered = estimator.output_dimensions(job["source_width"], job["source_height"],
                                                job["target"])

    # Time is judged against the same table on the name CF SENT, so J5's and F4's absences come
    # back naming CF's card from the worker itself. **Memory and time are different facts read
    # from different halves of one table.**

    entries = []
    for tier in read:
        answers = []
        for card in tier["cards"]:
            resolved = resolve_card(card, tier["cards"], table_cards)
            planned = _plan_card(job, resolved["hardware"], card_table, card["gpu_name"])
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
                resolved_from=resolved["resolved_from"],
            ))
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
        })
    return entries


#: §3b, and nothing else goes on the wire.
TIER_WIRE_FIELDS = ("tier", "fits_any", "fits_all", "output_width", "output_height",
                    "registry_version", "commit")
CARD_WIRE_FIELDS = ("gpu_name", "label", "fits", "max_target", "predicted_seconds",
                    "prediction_basis",
                    "rate_from", "timing_unavailable", "reason", "residency", "anchored", "binding_phase", "quality",
                    "hardware_used", "vram_source", "resolved_from")


def project(entry):
    wire = {field: entry[field] for field in TIER_WIRE_FIELDS}
    wire["cards"] = [{field: answer[field] for field in CARD_WIRE_FIELDS}
                     for answer in entry["cards"]]
    return wire


def version(commit, card_table=None):
    if card_table is None:
        card_table = estimator.load_card_table()
    # **The commit and nothing tree-shaped** (W6 item 3, ruled 2026-09-19): the handler/ tree's
    # only source was the retired history table, and CF's comparison is its pinned image tag
    # against this `commit` — both full shas.
    return {
        "commit": commit,
        "registry_version": planner.REGISTRY_VERSION,
        # W6: the in-image card table's own stamp, and how many cards it prices — the one
        # number a caller can act on. A row count meant nothing once the table was curated.
        "card_table_generated_utc": (card_table or {}).get("generated_utc"),
        "card_table_cards_priced": len(estimator.cards_with_rates(card_table)),
    }
