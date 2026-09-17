"""Generate `vram_table.json` from the delivered run records. **Never hand-edit the output.**

`cf-planner.md` §4a, ruled 2026-09-17 (461df2f). Per `gpu_name`:

    vram_total_gb   the MINIMUM total over the included readings
    vram_free_gb    the MINIMUM free among included readings AT that total

Every record status counts, because the hardware is read before the outcome exists.

**The leak is excluded by an explicit list, never a heuristic.** And the generator STOPS — it
neither includes nor drops — when any other record reads total minus free above 1 GiB, or when one
listed id is no longer in the corpus. Either is a question for the gate, not for this script.

Usage:
    python3 service/generate_vram_table.py [RUNS_DIR]           write service/vram_table.json
    python3 service/generate_vram_table.py --check [RUNS_DIR]   exit 1 if the committed table differs

RUNS_DIR defaults to `cf-upscale-project/records/runs` beside this repository. **It searches for
nothing:** one path, or the argument, and a missing directory is a stop.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_RUNS = os.path.join(os.path.dirname(REPO_ROOT), "cf-upscale-project", "records", "runs")
TABLE_PATH = os.path.join(HERE, "vram_table.json")

#: The 2026-08-23 leak: a previous job's memory still resident at snapshot (`predictor.md` §2c).
LEAK_EXCLUDED = (
    "01REAL20260823T18121000", "01REAL20260823T18170000", "01REAL20260823T18215100",
    "01REAL20260823T19044600", "01REAL20260823T22425500", "01REAL20260823T22484100",
    "01REAL20260823T22541500",
)

#: Above this gap a reading is not the driver context alone, and must be accounted for.
RESIDENT_GAP_LIMIT_GB = 1.0

#: The one reading above the limit that is ruled NOT a defect: Blackwell, 1.23 GiB, unexplained.
GAP_ALLOWED = ("01REAL20260820T09291300",)

_ID = re.compile(r"(01REAL\d{8}T\d{8})\.json$")


class Stop(Exception):
    """The corpus says something the rule does not cover. Goes to the gate."""


def record_id(filename):
    match = _ID.search(filename)
    if not match:
        raise Stop("{}: filename carries no record id".format(filename))
    return match.group(1)


def read_corpus(runs_dir):
    """`{record_id: (gpu_name, total, free, utc)}`, one entry per id.

    **Two files with one id count once when byte-identical and stop when not** — the corpus holds
    one such pair, and a reading counted twice is harmless to a minimum but not to a count.
    """
    if not os.path.isdir(runs_dir):
        raise Stop("no run records at {}".format(runs_dir))
    readings, raw = {}, {}
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith(".json"):
            continue
        rid = record_id(name)
        with open(os.path.join(runs_dir, name), "rb") as handle:
            body = handle.read()
        if rid in raw:
            if raw[rid] != body:
                raise Stop("{}: two files carry this id and differ".format(rid))
            continue
        raw[rid] = body
        record = json.loads(body)
        hardware = record.get("hardware") or {}
        gpu_name = hardware.get("gpu_name")
        total, free = hardware.get("vram_total_gb"), hardware.get("vram_free_gb")
        if not gpu_name or not _number(total) or not _number(free):
            raise Stop("{} ({}): no usable hardware block".format(rid, name))
        readings[rid] = (gpu_name, total, free, record.get("utc"))
    return readings


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def build(runs_dir):
    readings = read_corpus(runs_dir)

    missing = [rid for rid in LEAK_EXCLUDED + GAP_ALLOWED if rid not in readings]
    if missing:
        raise Stop("listed ids no longer in the corpus: {}".format(", ".join(missing)))

    included = {rid: r for rid, r in readings.items() if rid not in LEAK_EXCLUDED}
    violators = sorted(rid for rid, (_, total, free, _utc) in included.items()
                       if total - free > RESIDENT_GAP_LIMIT_GB and rid not in GAP_ALLOWED)
    if violators:
        raise Stop("readings above {} GiB resident, not ruled: {}".format(
            RESIDENT_GAP_LIMIT_GB, ", ".join(violators)))

    by_card = {}
    for rid, (gpu_name, total, free, _utc) in included.items():
        by_card.setdefault(gpu_name, []).append((total, free))
    cards = {}
    for gpu_name in sorted(by_card):
        rows = by_card[gpu_name]
        total = min(t for t, _ in rows)
        at_total = [f for t, f in rows if t == total]
        cards[gpu_name] = {
            "vram_total_gb": total,
            "vram_free_gb": min(at_total),
            "readings": len(rows),
            "readings_at_total": len(at_total),
        }

    utcs = sorted(u for (_, _, _, u) in readings.values() if u)
    return {
        "generated_by": "service/generate_vram_table.py — never hand-edited",
        "rule": ("per gpu_name: minimum vram_total_gb over included readings, then minimum "
                 "vram_free_gb at that total (cf-planner.md §4a)"),
        "corpus": {
            "source": "cf-upscale-project/records/runs",
            "records": len(readings),
            "included": len(included),
            "excluded": list(LEAK_EXCLUDED),
            "gap_allowed": list(GAP_ALLOWED),
            "first_record": min(readings),
            "last_record": max(readings),
            "first_utc": utcs[0] if utcs else None,
            "last_utc": utcs[-1] if utcs else None,
        },
        "cards": cards,
    }


def render(table):
    return json.dumps(table, indent=2, ensure_ascii=False) + "\n"


def main(argv):
    check = "--check" in argv
    args = [a for a in argv if a != "--check"]
    runs_dir = args[0] if args else DEFAULT_RUNS
    try:
        text = render(build(runs_dir))
    except Stop as stop:
        print("STOP: {}".format(stop), file=sys.stderr)
        return 2
    if check:
        try:
            with open(TABLE_PATH, encoding="utf-8") as handle:
                committed = handle.read()
        except OSError:
            committed = None
        if committed != text:
            print("vram_table.json differs from what the corpus generates", file=sys.stderr)
            return 1
        print("vram_table.json matches the corpus")
        return 0
    with open(TABLE_PATH, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("wrote {}".format(TABLE_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
