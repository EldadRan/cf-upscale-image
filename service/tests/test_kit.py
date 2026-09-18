"""The CF planner service's kit — `cf-planner.md` §7, one class per case.

Run from the repository root:  python3 -m unittest discover -s service/tests -t . -v

**Parity runs against `handler/` as imported, never against a fixture of its output.** It reads
the banked run record from `cf-upscale-project/records/runs`, which is local-only; where that
directory is absent the case FAILS rather than skips, because a skipped parity case is a green kit
that certified nothing.
"""

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(SERVICE)
RUNS = os.path.join(os.path.dirname(REPO_ROOT), "cf-upscale-project", "records", "runs")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from service import planner_service as ps  # noqa: E402

import estimator  # noqa: E402  (on the path through service.worker_path)
import planner  # noqa: E402

SHA_A = "a" * 40
SHA_B = "b" * 40
#: 26294cc is e499dd5's parent, and e499dd5 changed handler/estimator.py — so its tree is not
#: HEAD's on any later commit either.
OTHER_TREE_COMMIT = "26294cc5cb1f90fdaabe84f59223c1215a6f3fc1"

with open(os.path.join(SERVICE, "vram_table.json"), encoding="utf-8") as _handle:
    TABLE = json.load(_handle)["cards"]

A40 = "NVIDIA A40"
H200 = "NVIDIA H200"
B200 = "NVIDIA B200"
BLACKWELL = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
MIG = "NVIDIA RTX PRO 6000 Blackwell MIG 2g.48gb"

#: Card answer field -> the rationale key it is the worker's own output of.
PROJECTED_FROM_RATIONALE = (
    ("predicted_seconds", "predicted_seconds"), ("residency", "residency"),
    ("binding_phase", "binding_phase"), ("anchored", "anchored"),
    ("prediction_basis", "prediction_basis"),
)
#: `quality` (§3d) -> the rationale key each field is the plan's own.
QUALITY_FROM_RATIONALE = (
    ("best_window", "window"), ("ideal_window", "ideal_window"), ("passes", "passes"),
    ("shortest_pass", "shortest_pass"), ("decode_grid", "decode_grid"),
    ("decode_tile", "decode_tile"), ("encode_grid", "encode_grid"),
    ("encode_tile", "encode_tile"),
)


def job(**overrides):
    body = {"source_width": 1920, "source_height": 1080, "frames": 90, "is_still": False,
            "target_short_edge_px": 1480, "tile_quality": "default", "schedule": "max_window"}
    body.update(overrides)
    return body


def card(gpu_name=A40, **fields):
    return dict({"gpu_name": gpu_name}, **fields)


def tier(**overrides):
    body = {"tier": "t1", "host_ram_gb": 46.57, "worker_commit": None, "cards": [card()]}
    body.update(overrides)
    return body


def request(job_body=None, tiers=None):
    return {"job": job_body or job(), "tiers": tiers or [tier()]}


def one(body, commit=None):
    """The first tier's answer."""
    return ps.estimate_core(body, commit=commit)[0]


def answer(body, index=0, commit=None):
    """One card's answer, in CF's order."""
    return one(body, commit=commit)["cards"][index]


def refused_field(test, body, field):
    with test.assertRaises(ps.Refusal) as caught:
        ps.estimate_core(body, commit=None)
    test.assertEqual(caught.exception.field, field, caught.exception.message)


def _git(*args):
    return subprocess.run(["git", "-C", REPO_ROOT] + list(args), capture_output=True,
                          text=True, check=True).stdout


def _same_tree_commit():
    """A commit on main, not HEAD, whose `handler/` tree is HEAD's — read from the table.

    **Called per test, never at import** (review, W1): a stale table must fail the tests that
    need it and leave `HistoryTable` to name the cause, not error the whole module before any
    test runs.

    **Derived, not pinned.** It was pinned to e499dd5 when P1 was written, and the first wave to
    touch `handler/` after it (W1 J7) made that commit's tree someone else's. Every wave that
    touches `handler/` regenerates the table, so the table always holds one: its `newest`, when
    HEAD is the regeneration commit after it.
    """
    with open(os.path.join(SERVICE, "handler_history.json"), encoding="utf-8") as handle:
        commits = json.load(handle)["commits"]
    head = _git("rev-parse", "HEAD").strip()
    tree = _git("rev-parse", "HEAD:handler").strip()
    for sha, handler_tree in commits.items():
        if sha != head and handler_tree == tree:
            return sha
    raise AssertionError(
        "no commit in handler_history.json other than HEAD shares HEAD's handler/ tree — the "
        "table is stale or uncommitted; regenerate it and commit it (HistoryTable names which)")


class Parity(unittest.TestCase):
    """cd622500: the job's request, its hardware block as its one card, reproduces its rationale."""

    RECORD = "2026-08-31_01REAL20260831T15562300.json"
    #: The two keys the handler adds after `estimator.plan` returns (§7, ruled on C3).
    HANDLER_ADDED = ("deadline", "effective_temporal_window")

    def test_cd622500(self):
        path = os.path.join(RUNS, self.RECORD)
        self.assertTrue(os.path.isfile(path), "parity record not found at {}".format(path))
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        self.assertTrue(record["runpod"]["job_id"].startswith("cd622500"))
        source, req, hw = record["source"], record["request"], record["hardware"]
        body = request(
            job(source_width=source["width"], source_height=source["height"],
                frames=source["estimated_frames"], is_still=source["still"],
                target_short_edge_px=req["target_short_edge_px"],
                tile_quality=req["tile_quality"], schedule=req["schedule"]),
            [tier(host_ram_gb=hw["host_ram_gb"],
                  cards=[card(hw["gpu_name"], vram_total_gb=hw["vram_total_gb"],
                              vram_free_gb=hw["vram_free_gb"], host_ram_gb=hw["host_ram_gb"],
                              label="idle")])])
        entry = one(body)
        got = entry["cards"][0]
        fresh, recorded = got["rationale"], record["rationale"]

        self.assertEqual(got["vram_source"], "given")
        self.assertIsNone(got["resolved_from"])
        compared = sorted(k for k in recorded if k not in self.HANDLER_ADDED)
        mismatched = {k: (recorded[k], fresh.get(k, "<absent>"))
                      for k in compared if fresh.get(k, object()) != recorded[k]}
        self.assertEqual(mismatched, {}, "recorded vs fresh")
        only_fresh = sorted(set(fresh) - set(recorded))
        print("\n  parity: {} recorded keys equal; excluded {}; only in fresh output: {}".format(
            len(compared), list(self.HANDLER_ADDED), only_fresh))
        self.assertEqual(got["predicted_seconds"], 897.1)
        self.assertEqual(recorded["predicted_seconds"], 897.1)
        # Every field the wire carries is the worker's own, key for key (review F2, F3 on P1).
        for field, key in PROJECTED_FROM_RATIONALE:
            self.assertEqual(got[field], recorded[key], field)
        for field, key in QUALITY_FROM_RATIONALE:
            self.assertEqual(got["quality"][field], recorded[key], field)
        self.assertEqual(got["prediction_basis"], "measured")
        self.assertEqual((entry["output_width"], entry["output_height"]),
                         (record["output"]["width"], record["output"]["height"]))


class PerCard(unittest.TestCase):
    """cards[] in and cards[] out: same length, same order, and nothing is reordered."""

    CARDS = [card(H200, vram_total_gb=141.0), card(A40, vram_total_gb=44.7), card(B200),
             card(MIG, vram_total_gb=44.5)]

    def test_same_length_same_order(self):
        entry = one(request(tiers=[tier(host_ram_gb=377.0,
                                        cards=[dict(c) for c in self.CARDS])]))
        self.assertEqual([c["gpu_name"] for c in entry["cards"]],
                         [c["gpu_name"] for c in self.CARDS])
        # **Each answer is planned against ITS OWN card** — the echoed name alone is built in the
        # same loop as the answer, so it cannot show a misalignment (review F5).
        self.assertEqual([c["hardware_used"]["gpu_name"] for c in entry["cards"]],
                         [H200, A40, B200, A40])
        self.assertEqual([c["vram_source"] for c in entry["cards"]],
                         ["table", "table", "table", "nearest_memory"])
        self.assertEqual([c["hardware_used"]["vram_total_gb"] for c in entry["cards"]],
                         [TABLE[H200]["vram_total_gb"], TABLE[A40]["vram_total_gb"],
                          TABLE[B200]["vram_total_gb"], TABLE[A40]["vram_total_gb"]])
        self.assertEqual(entry["cards"][3]["resolved_from"], {"card": MIG, "measured": A40})

    def test_labels_echoed(self):
        cards = [card(A40, label="idle"), card(H200, label="catalog"), card(B200)]
        entry = one(request(tiers=[tier(host_ram_gb=377.0, cards=cards)]))
        self.assertEqual([c["label"] for c in entry["cards"]], ["idle", "catalog", None])

    def test_fits_any_and_all(self):
        # A 4320 target: the A40 cannot hold it, the B200 can.
        entry = one(request(job(target_short_edge_px=4320),
                            [tier(host_ram_gb=377.0, cards=[card(A40), card(B200)])]))
        self.assertEqual([c["fits"] for c in entry["cards"]], [False, True])
        self.assertIs(entry["fits_any"], True)
        self.assertIs(entry["fits_all"], False)

        allfit = one(request(tiers=[tier(cards=[card(A40), card(B200)])]))
        self.assertEqual([c["fits"] for c in allfit["cards"]], [True, True])
        self.assertIs(allfit["fits_any"], True)
        self.assertIs(allfit["fits_all"], True)

        none = one(request(job(target_short_edge_px=4320), [tier(cards=[card(A40)])]))
        self.assertIs(none["fits_any"], False)
        self.assertIs(none["fits_all"], False)

    def test_a_refused_card_still_answers(self):
        entry = one(request(job(target_short_edge_px=4320),
                            [tier(host_ram_gb=377.0, cards=[card(A40), card(B200)])]))
        self.assertEqual(len(entry["cards"]), 2)
        self.assertFalse(entry["cards"][0]["fits"])
        self.assertIsNotNone(entry["cards"][0]["reason"])
        self.assertIsNone(entry["cards"][0]["predicted_seconds"])
        self.assertIsNotNone(entry["cards"][0]["quality"]["ideal_window"])

    def test_tier_fields_named_once(self):
        entry = one(request(tiers=[tier(worker_commit=_same_tree_commit())]), commit=SHA_A)
        for field in ("output_width", "output_height", "registry_version", "commit",
                      "handler_tree", "worker_handler_tree", "handler_match"):
            self.assertIn(field, entry)
            self.assertNotIn(field, entry["cards"][0])

    def test_several_tiers_answer_in_order(self):
        entries = ps.estimate_core(request(tiers=[tier(), tier(tier="t2", host_ram_gb=377.0,
                                                              cards=[card(B200)])]), commit=None)
        self.assertEqual([e["tier"] for e in entries], ["t1", "t2"])
        self.assertEqual([c["gpu_name"] for e in entries for c in e["cards"]], [A40, B200])


class RateFrom(unittest.TestCase):
    """Whose measurement the time is (§4a-ii): null where it is this card's own."""

    def test_one_number_on_three_cards_says_where_it_came_from(self):
        cards = [card(A40), card(H200), card(B200)]
        entry = one(request(tiers=[tier(host_ram_gb=377.0, cards=cards)]))
        seconds = [c["predicted_seconds"] for c in entry["cards"]]
        self.assertEqual(len(set(seconds)), 1, "the case exists because the numbers are equal")
        rates = [c["rate_from"] for c in entry["cards"]]
        self.assertIsNone(rates[0], "the A40 owns the rows this rate came from")
        for got, name in zip(entry["cards"][1:], (H200, B200)):
            self.assertIsNotNone(got["rate_from"], name)
            self.assertEqual(got["rate_from"]["running_on"], name)
            self.assertEqual(got["rate_from"]["rows_measured_on"], [A40])
            self.assertEqual(got["prediction_basis"], "borrowed", name)
        self.assertNotEqual(rates, [None, None, None])

    def test_rate_from_is_the_workers_own_keys(self):
        got = answer(request(tiers=[tier(host_ram_gb=377.0, cards=[card(B200)])]))
        self.assertEqual(got["rate_from"], got["rationale"]["timing_from_another_card"])

    def test_tiling_half(self):
        # `high` tiling has no rows at all, on any card: the rate is the default-tiling rows'.
        got = answer(request(job(tile_quality="high"), [tier(cards=[card(A40)])]))
        self.assertEqual(got["rate_from"]["tiling"],
                         got["rationale"]["timing_from_another_tiling"])
        self.assertNotIn("running_on", got["rate_from"])
        self.assertEqual(got["prediction_basis"], "borrowed")

    def test_both_halves_at_once(self):
        got = answer(request(job(tile_quality="high"),
                             [tier(host_ram_gb=377.0, cards=[card(B200)])]))
        self.assertEqual(got["rate_from"]["running_on"], B200)
        self.assertEqual(got["rate_from"]["rows_measured_on"], [A40])
        self.assertEqual(got["rate_from"]["tiling"]["running_at"], "high")

    def test_borrowed_in_either_sense(self):
        # Memory borrowed, time this card's own: the A40's rows priced a MIG resolved to the A40.
        got = answer(request(tiers=[tier(cards=[card(MIG, vram_total_gb=44.5)])]))
        self.assertEqual(got["prediction_basis"], "borrowed")
        self.assertIsNotNone(got["resolved_from"])
        self.assertIsNone(got["rate_from"])

    def test_refused_card_has_no_rate(self):
        got = answer(request(job(target_short_edge_px=4320)))
        self.assertFalse(got["fits"])
        self.assertIsNone(got["rate_from"])


class RefusedQuality(unittest.TestCase):
    """A refused card still carries ideal_window, and nulls for the rest (§3b)."""

    def test_ideal_window_survives_a_refusal(self):
        got = answer(request(job(target_short_edge_px=4320)))
        self.assertFalse(got["fits"])
        self.assertEqual(got["quality"]["ideal_window"], planner.ideal_window(90))
        others = {k: v for k, v in got["quality"].items() if k != "ideal_window"}
        self.assertEqual(set(others.values()), {None}, others)
        self.assertEqual(sorted(got["quality"]), sorted(f for f, _ in QUALITY_FROM_RATIONALE))

    def test_a_still_refusal_reports_its_own_ideal(self):
        got = answer(request(job(frames=1, is_still=True, target_short_edge_px=8000),
                             [tier(cards=[card(A40)])]))
        self.assertFalse(got["fits"])
        self.assertEqual(got["quality"]["ideal_window"], planner.ideal_window(1))


class Quality(unittest.TestCase):
    """§3d: window, tail and tiling — and the chunk and the blocks are not on the wire."""

    def test_quality_equals_the_plan(self):
        got = answer(request())
        for field, key in QUALITY_FROM_RATIONALE:
            self.assertEqual(got["quality"][field], got["rationale"][key], field)
        self.assertEqual(sorted(got["quality"]), sorted(f for f, _ in QUALITY_FROM_RATIONALE))

    def test_chunk_and_blocks_are_not_on_the_wire(self):
        from service import app
        status, payload = app.route("POST", "/estimate",
                                    json.dumps(request()).encode("utf-8"), commit=SHA_A)
        self.assertEqual(status, 200)
        text = json.dumps(payload)
        for absent in ("chunk_size", "blocks_to_swap", "chunks", "tail_chunk", "rationale"):
            self.assertNotIn(absent, text)

    def test_ideal_window_says_whether_a_bigger_card_helps(self):
        entry = one(request(tiers=[tier(host_ram_gb=377.0, cards=[card(A40), card(B200)])]))
        small, large = (c["quality"] for c in entry["cards"])
        self.assertLess(small["best_window"], small["ideal_window"])
        self.assertEqual(large["best_window"], large["ideal_window"])


class VramOrder(unittest.TestCase):
    """given, then table, then nearest_memory, then pool_floor — each stops the next (§4a-i)."""

    def test_given_first(self):
        got = answer(request(tiers=[tier(cards=[card(A40, vram_total_gb=44.43,
                                                     vram_free_gb=44.08)])]))
        self.assertEqual(got["vram_source"], "given")
        self.assertEqual(got["hardware_used"]["vram_total_gb"], 44.43)
        self.assertEqual(got["hardware_used"]["vram_free_gb"], 44.08)
        self.assertIsNone(got["vram_stats"])

    def test_free_without_total_is_not_a_reading(self):
        got = answer(request(tiers=[tier(cards=[card(A40, vram_free_gb=44.08)])]))
        self.assertEqual(got["vram_source"], "table")
        self.assertEqual(got["hardware_used"]["vram_free_gb"], TABLE[A40]["vram_free_gb"])

    def test_table_before_nearest(self):
        # A nominal that points AWAY from the card's own table row: nearest-first would plan the
        # H200's 139.8, so the figures separate the two orderings, not just the label (review F4).
        self.assertEqual(ps.nearest_measured(140.0, TABLE), H200)
        got = answer(request(tiers=[tier(cards=[card(A40, vram_total_gb=140.0)])]))
        self.assertEqual(got["vram_source"], "table")
        self.assertEqual(got["hardware_used"]["gpu_name"], A40)
        self.assertEqual(got["hardware_used"]["vram_total_gb"], TABLE[A40]["vram_total_gb"])
        self.assertIsNone(got["resolved_from"])
        self.assertEqual(got["prediction_basis"], got["rationale"]["prediction_basis"])

    def test_total_without_free_is_a_nominal_never_planned_against(self):
        got = answer(request(tiers=[tier(cards=[card(MIG, vram_total_gb=44.5)])]))
        self.assertEqual(got["vram_source"], "nearest_memory")
        self.assertNotEqual(got["hardware_used"]["vram_total_gb"], 44.5)
        self.assertEqual(got["hardware_used"]["vram_total_gb"], TABLE[A40]["vram_total_gb"])

    def test_unmeasured_without_nominal_is_pool_floor(self):
        got = answer(request(tiers=[tier(cards=[card(MIG)])]))
        self.assertEqual(got["vram_source"], "pool_floor")

    def test_basis_not_rewritten_where_the_card_is_cfs(self):
        for entry_card in (card(A40), card(A40, vram_total_gb=44.43, vram_free_gb=44.08)):
            got = answer(request(tiers=[tier(cards=[entry_card])]))
            self.assertIn(got["vram_source"], ("given", "table"))
            self.assertIsNone(got["resolved_from"])
            self.assertEqual(got["prediction_basis"], got["rationale"]["prediction_basis"])
            self.assertEqual(got["prediction_basis"], "measured")


class PoolFloor(unittest.TestCase):
    """No nominal: the worst measured card in THIS list, else the table's worst (§4a-i)."""

    def test_worst_measured_in_this_list(self):
        cards = [card(MIG), card(H200, vram_total_gb=141.0), card(B200, vram_total_gb=179.0)]
        got = answer(request(tiers=[tier(host_ram_gb=377.0, cards=cards)]))
        self.assertEqual(got["vram_source"], "pool_floor")
        # The H200 is the worst card in THIS list that the table measures — not the A40, which is
        # the table's worst and is not in this list.
        self.assertEqual(got["hardware_used"]["gpu_name"], H200)
        self.assertEqual(got["resolved_from"], {"card": MIG, "measured": H200})
        self.assertEqual(got["prediction_basis"], "borrowed")

    def test_floor_card_is_borrowed_even_when_the_worker_measured_it(self):
        # The floor lands on the A40, which the table measures at this size, so the worker labels
        # the rate "measured". It is not a measurement of the card CF named.
        cards = [card(MIG), card(A40, vram_total_gb=44.7)]
        got = answer(request(tiers=[tier(cards=cards)]))
        self.assertEqual(got["vram_source"], "pool_floor")
        self.assertEqual(got["hardware_used"]["gpu_name"], A40)
        self.assertEqual(got["rationale"]["prediction_basis"], "measured")
        self.assertEqual(got["prediction_basis"], "borrowed")

    def test_tables_worst_where_the_list_measures_none(self):
        got = answer(request(tiers=[tier(cards=[card(MIG), card("NVIDIA MADE UP 9000")])]))
        self.assertEqual(got["vram_source"], "pool_floor")
        worst = min(TABLE, key=lambda name: TABLE[name]["vram_total_gb"])
        self.assertEqual(got["hardware_used"]["gpu_name"], worst)
        self.assertEqual(got["resolved_from"], {"card": MIG, "measured": worst})
        self.assertEqual(got["hardware_used"]["vram_total_gb"], TABLE[worst]["vram_total_gb"])


class Nearest(unittest.TestCase):
    """A nominal resolves to the nearest measured card; resolved_from names both."""

    def test_resolves_and_is_borrowed(self):
        self.assertNotIn(MIG, TABLE)
        got = answer(request(tiers=[tier(cards=[card(MIG, vram_total_gb=44.5)])]))
        self.assertEqual(got["vram_source"], "nearest_memory")
        self.assertEqual(got["resolved_from"], {"card": MIG, "measured": A40})
        self.assertEqual(got["hardware_used"]["gpu_name"], A40)
        self.assertEqual(got["rationale"]["prediction_basis"], "measured")
        self.assertEqual(got["prediction_basis"], "borrowed")

    def test_nominal_near_a_bigger_card(self):
        got = answer(request(tiers=[tier(host_ram_gb=377.0,
                                         cards=[card(MIG, vram_total_gb=96.0)])]))
        self.assertEqual(got["resolved_from"], {"card": MIG, "measured": BLACKWELL})

    def test_tie_takes_lower_memory(self):
        low, high = TABLE[A40]["vram_total_gb"], TABLE[BLACKWELL]["vram_total_gb"]
        table = {A40: TABLE[A40], BLACKWELL: TABLE[BLACKWELL]}
        self.assertEqual(ps.nearest_measured((low + high) / 2.0, table), A40)


class VramStats(unittest.TestCase):
    """The corpus behind a table figure rides back; the plan uses the MINIMUM (§4a)."""

    def test_stats_on_table(self):
        got = answer(request(tiers=[tier(cards=[card(A40)])]))
        self.assertEqual(got["vram_source"], "table")
        self.assertEqual(got["vram_stats"], TABLE[A40]["stats"])
        self.assertEqual(got["vram_stats"]["n_at_total"], TABLE[A40]["readings_at_total"])
        self.assertLess(got["vram_stats"]["n_at_total"], got["vram_stats"]["n"])
        self.assertEqual(got["hardware_used"]["vram_free_gb"], TABLE[A40]["vram_free_gb"])
        self.assertNotEqual(got["hardware_used"]["vram_free_gb"],
                            got["vram_stats"]["free_mean_gb"])
        self.assertLess(got["hardware_used"]["vram_free_gb"], got["vram_stats"]["free_max_gb"])

    def test_stats_on_nearest(self):
        got = answer(request(tiers=[tier(cards=[card(MIG, vram_total_gb=44.5)])]))
        self.assertEqual(got["vram_stats"], TABLE[A40]["stats"])

    def test_stats_are_copies_not_the_tables_own_object(self):
        table = ps.load_vram_table()
        entry = one(request(tiers=[tier(cards=[card(A40), card(A40)])]),)
        first, second = (c["vram_stats"] for c in entry["cards"])
        self.assertEqual(first, TABLE[A40]["stats"])
        self.assertIsNot(first, second)
        self.assertIsNot(first, table["cards"][A40]["stats"])

    def test_no_stats_where_the_figure_is_not_the_tables(self):
        got = answer(request(tiers=[tier(cards=[card(A40, vram_total_gb=44.43,
                                                     vram_free_gb=44.08)])]))
        self.assertIsNone(got["vram_stats"])

    def test_stats_recomputed_from_the_corpus(self):
        # **Recomputed here, not compared with itself**: min over a superset of the readings the
        # figure comes from is below it for ANY implementation, right or wrong (review F3).
        if SERVICE not in sys.path:
            sys.path.insert(0, SERVICE)
        import generate_vram_table as gen
        readings = gen.read_corpus(RUNS)
        by_card = {}
        for rid, (gpu_name, total, free, _utc) in readings.items():
            if rid not in gen.LEAK_EXCLUDED:
                by_card.setdefault(gpu_name, []).append((total, free))
        self.assertEqual(sorted(by_card), sorted(TABLE))
        for name, rows in by_card.items():
            frees = [f for _t, f in rows]
            lowest_total = min(t for t, _ in rows)
            row = TABLE[name]
            self.assertEqual(row["vram_total_gb"], lowest_total, name)
            self.assertEqual(row["vram_free_gb"],
                             min(f for t, f in rows if t == lowest_total), name)
            self.assertEqual(row["stats"], {
                "free_mean_gb": round(sum(frees) / len(frees), 4),
                "free_min_gb": min(frees),
                "free_max_gb": max(frees),
                "n": len(frees),
                "n_at_total": len([t for t, _ in rows if t == lowest_total]),
            }, name)
            # The two answer different questions (§4a): the spread ranges over every reading, the
            # planned figure rests on the readings at the minimum total alone.
            self.assertLessEqual(row["stats"]["n_at_total"], row["stats"]["n"], name)

    def test_the_committed_table_is_the_generators(self):
        out = subprocess.run([sys.executable, os.path.join(SERVICE, "generate_vram_table.py"),
                              "--check", RUNS], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)


class HostRam(unittest.TestCase):
    """A card's own host_ram_gb wins over the tier's; neither refuses THAT card by name."""

    def test_card_before_tier(self):
        cards = [card(A40, host_ram_gb=51.22), card(H200)]
        entry = one(request(tiers=[tier(host_ram_gb=46.57, cards=cards)]))
        self.assertEqual(entry["cards"][0]["hardware_used"]["host_ram_gb"], 51.22)
        self.assertEqual(entry["cards"][1]["hardware_used"]["host_ram_gb"], 46.57)

    def test_neither_refuses_that_card(self):
        body = request(tiers=[tier(host_ram_gb=None,
                                   cards=[card(A40, host_ram_gb=46.57), card(H200)])])
        refused_field(self, body, "tiers[0].cards[1].host_ram_gb")

    def test_tier_host_ram_may_be_absent_when_every_card_has_one(self):
        t = tier(cards=[card(A40, host_ram_gb=46.57)])
        del t["host_ram_gb"]
        got = answer(request(tiers=[t]))
        self.assertEqual(got["hardware_used"]["host_ram_gb"], 46.57)


class Refusals(unittest.TestCase):
    """By name, never defaulted (§4)."""

    def test_empty_cards(self):
        refused_field(self, request(tiers=[tier(cards=[])]), "tiers[0].cards")

    def test_absent_cards(self):
        t = tier()
        del t["cards"]
        refused_field(self, request(tiers=[t]), "tiers[0].cards")

    def test_card_without_gpu_name(self):
        refused_field(self, request(tiers=[tier(cards=[card(A40), {"vram_total_gb": 44.7}])]),
                      "tiers[0].cards[1].gpu_name")

    def test_no_host_ram(self):
        t = tier(cards=[card(A40)])
        del t["host_ram_gb"]
        refused_field(self, request(tiers=[t]), "tiers[0].cards[0].host_ram_gb")

    def test_both_sizing_forms(self):
        refused_field(self, request(job(output_size={"width": 2000, "height": 1000})),
                      "job.output_size")

    def test_neither_sizing_form(self):
        body = request(job())
        del body["job"]["target_short_edge_px"]
        refused_field(self, body, "job.target_short_edge_px")

    def test_short_worker_commit(self):
        refused_field(self, request(tiers=[tier(worker_commit=_same_tree_commit()[:7])]),
                      "tiers[0].worker_commit")

    def test_empty_tiers(self):
        refused_field(self, {"job": job(), "tiers": []}, "tiers")

    def test_still_with_frames(self):
        refused_field(self, request(job(frames=90, is_still=True)), "job.frames")

    def test_an_empty_table_is_not_the_callers_fault(self):
        # A table that measures no card is a DEPLOY fault: it stops the request loudly, it is
        # never a 400 naming a field CF never sent, and it stops before anything is planned.
        from service import app
        with self.assertRaises(ps.TableUnusable):
            ps.estimate_core(request(tiers=[tier(cards=[card(MIG)])]), commit=None,
                             vram_table={"cards": {}, "corpus": {}})
        real = ps.load_vram_table
        ps.load_vram_table = lambda *a, **k: {"cards": {}, "corpus": {}}
        try:
            status, payload = app.route("POST", "/estimate",
                                        json.dumps(request()).encode("utf-8"), commit=None)
        finally:
            ps.load_vram_table = real
        self.assertEqual(status, 503)
        self.assertNotIn("refused", payload)


class Residency(unittest.TestCase):
    """On a refusal, planner.plan refuses too, and the labels follow planner.fits (§3b)."""

    def test_host_refusal_reports_route_up(self):
        got = answer(request(tiers=[tier(host_ram_gb=8.0)]))
        self.assertFalse(got["fits"])
        self.assertIn("host RAM", got["reason"])
        verdict = got["planner_verdict"]
        self.assertNotEqual(verdict["action"], "plan")
        self.assertEqual(got["residency"], "route_up")
        self.assertEqual(got["residency"], verdict["residency"])
        self.assertEqual(estimator._refusal_text(verdict["reason"]), got["reason"])
        self.assertEqual(got["anchored"], estimator._usable_vram(got["hardware_used"])
                         <= planner.ANCHORED_MAX_USABLE)

    def test_diverging_verdict_is_an_error(self):
        # Only the service's own call diverges: estimator.plan's internal planner calls are real.
        real_plan, real_estimate = planner.plan, estimator.plan
        state = {"estimator_done": False}

        def estimate_then_flag(*args, **kwargs):
            try:
                return real_estimate(*args, **kwargs)
            finally:
                state["estimator_done"] = True

        def other_reason(*args, **kwargs):
            reply = real_plan(*args, **kwargs)
            return dict(reply, reason="a different constraint") if state["estimator_done"] \
                else reply

        planner.plan, estimator.plan = other_reason, estimate_then_flag
        try:
            with self.assertRaises(RuntimeError):
                answer(request(tiers=[tier(host_ram_gb=8.0)]))
        finally:
            planner.plan, estimator.plan = real_plan, real_estimate

    def test_vram_refusal_labels_follow_planner_fits(self):
        got = answer(request(job(target_short_edge_px=4320)))
        self.assertFalse(got["fits"])
        verdict = got["planner_verdict"]
        self.assertNotIn("residency", verdict)
        self.assertEqual(estimator._refusal_text(verdict["reason"]), got["reason"])
        hw = got["hardware_used"]
        fits = planner.fits((1920, 1080), 90, 4320, estimator._usable_vram(hw),
                            host_ram_gb=hw["host_ram_gb"], tile_quality="default",
                            gpu_name=hw["gpu_name"])
        self.assertFalse(fits["fits"])
        self.assertEqual(got["residency"], "route_up")
        self.assertEqual(got["residency"], fits["residency"])
        self.assertIsNotNone(got["anchored"])
        self.assertEqual(got["anchored"], fits["anchored"])

    def test_refusal_anchored_at_both_sides_of_the_span(self):
        for name, expected in ((H200, True), (B200, False)):
            got = answer(request(tiers=[tier(host_ram_gb=8.0, cards=[card(name)])]))
            self.assertFalse(got["fits"], name)
            self.assertEqual(got["vram_source"], "table", name)
            hw = got["hardware_used"]
            fits = planner.fits((1920, 1080), 90, 1480, estimator._usable_vram(hw),
                                host_ram_gb=hw["host_ram_gb"], tile_quality="default",
                                gpu_name=hw["gpu_name"])
            self.assertFalse(fits["fits"], name)
            self.assertIs(got["anchored"], expected, name)
            self.assertEqual(got["anchored"], fits["anchored"], name)

    def test_refusal_anchored_on_the_boundary(self):
        # A banked H200 reading (free 139.07) leaves usable exactly at ANCHORED_MAX_USABLE.
        got = answer(request(tiers=[tier(host_ram_gb=8.0,
                                         cards=[card(H200, vram_total_gb=139.8,
                                                     vram_free_gb=139.07)])]))
        self.assertFalse(got["fits"])
        self.assertEqual(estimator._usable_vram(got["hardware_used"]),
                         planner.ANCHORED_MAX_USABLE)
        self.assertIs(got["anchored"], True)

    def test_fit_residency_from_the_plan(self):
        got = answer(request())
        self.assertTrue(got["fits"])
        self.assertEqual(got["residency"], got["rationale"]["residency"])


class NoTorch(unittest.TestCase):
    """The service's environment has no torch, and every module it imports loads."""

    def test_imports_without_torch(self):
        probe = (
            "import sys, importlib.util as u\n"
            "sys.path.insert(0, {root!r})\n"
            "import service.app, service.planner_service, service.worker_path\n"
            "heavy = sorted(m for m in ('torch', 'numpy', 'cv2', 'boto3') if m in sys.modules)\n"
            "print('loaded-heavy', heavy)\n"
            "print('torch-installed', u.find_spec('torch') is not None)\n"
        ).format(root=REPO_ROOT)
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("loaded-heavy []", out.stdout)
        self.assertIn("torch-installed False", out.stdout)

    def test_worker_is_the_resolved_handler(self):
        from service import worker_path
        for module in (estimator, planner):
            self.assertEqual(os.path.dirname(os.path.abspath(module.__file__)),
                             worker_path.HANDLER, module.__name__)

    def test_resolved_handler_wins_over_an_earlier_path_entry(self):
        decoy = tempfile.mkdtemp(prefix="decoy_handler_")
        probe = (
            "import sys; sys.path.insert(0, {decoy!r}); sys.path.insert(1, {handler!r})\n"
            "sys.path.insert(0, {root!r})\n"
            "import service.planner_service, estimator\n"
            "print(estimator.__file__)\n"
        ).format(decoy=decoy, handler=os.path.join(REPO_ROOT, "handler"), root=REPO_ROOT)
        try:
            with open(os.path.join(decoy, "estimator.py"), "w") as handle:
                handle.write("DECOY = True\n")
            out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        finally:
            shutil.rmtree(decoy)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(os.path.dirname(out.stdout.strip()), os.path.join(REPO_ROOT, "handler"))

    def test_dependency_list_names_no_handler_requirements(self):
        with open(os.path.join(SERVICE, "requirements.txt"), encoding="utf-8") as handle:
            text = handle.read().lower()
        self.assertNotIn("torch", text)
        self.assertNotIn("handler/requirements", text)
        with open(os.path.join(SERVICE, "Dockerfile"), encoding="utf-8") as handle:
            docker = handle.read()
        self.assertNotIn("handler/requirements.txt", docker)


class OutputSize(unittest.TestCase):
    """Planned at short_edge_covering; output_width/height equal the canvas exactly."""

    def test_canvas(self):
        body = request(job(output_size={"width": 2001, "height": 1003}))
        del body["job"]["target_short_edge_px"]
        entry = one(body)
        covering = estimator.short_edge_covering(1920, 1080, 2001, 1003)
        self.assertEqual(entry["planned_short_edge_px"], covering)
        self.assertEqual(entry["cards"][0]["rationale"]["output_width"],
                         estimator.output_dimensions(1920, 1080, covering)[0])
        self.assertEqual((entry["output_width"], entry["output_height"]), (2001, 1003))


class Still(unittest.TestCase):
    """An image source plans at frames 1 and delivers output_dimensions."""

    def test_still(self):
        entry = one(request(job(frames=1, is_still=True, source_width=749, source_height=500,
                                target_short_edge_px=1920)))
        got = entry["cards"][0]
        self.assertTrue(got["fits"])
        self.assertEqual(got["quality"]["best_window"], 1)
        self.assertEqual((entry["output_width"], entry["output_height"]),
                         estimator.output_dimensions(749, 500, 1920))


class Match(unittest.TestCase):
    """handler_match on handler/'s tree, not the commit (§5)."""

    def test_same_tree_other_commit_matches(self):
        self.assertNotEqual(_same_tree_commit(), _git("rev-parse", "HEAD").strip())
        entry = one(request(tiers=[tier(worker_commit=_same_tree_commit())]), commit=SHA_A)
        self.assertEqual(entry["handler_tree"], _git("rev-parse", "HEAD:handler").strip())
        self.assertEqual(entry["worker_handler_tree"],
                         _git("rev-parse", _same_tree_commit() + ":handler").strip())
        self.assertIs(entry["handler_match"], True)
        self.assertEqual(entry["commit"], SHA_A)
        self.assertNotIn("commit_match", entry)

    def test_other_tree_mismatches(self):
        entry = one(request(tiers=[tier(worker_commit=OTHER_TREE_COMMIT)]))
        self.assertEqual(entry["worker_handler_tree"],
                         _git("rev-parse", OTHER_TREE_COMMIT + ":handler").strip())
        self.assertNotEqual(entry["worker_handler_tree"], entry["handler_tree"])
        self.assertIs(entry["handler_match"], False)

    def test_commit_not_in_table_is_null(self):
        entry = one(request(tiers=[tier(worker_commit=SHA_B)]))
        self.assertIsNone(entry["worker_handler_tree"])
        self.assertIsNone(entry["handler_match"])
        self.assertIsNotNone(entry["handler_tree"])

    def test_none_sent_is_null(self):
        entry = one(request())
        self.assertIsNone(entry["worker_handler_tree"])
        self.assertIsNone(entry["handler_match"])

    def test_service_commit_is_information(self):
        with self.assertRaises(ValueError):
            ps.service_commit({"CF_PLANNER_COMMIT": "abc1234"})
        self.assertEqual(ps.service_commit({"CF_PLANNER_COMMIT": SHA_A}), SHA_A)
        self.assertIsNone(ps.service_commit({}))
        entry = one(request(tiers=[tier(worker_commit=_same_tree_commit())]), commit=None)
        self.assertIsNone(entry["commit"])
        self.assertIs(entry["handler_match"], True)


class TablesAfterTheDoor(unittest.TestCase):
    """A malformed request is refused by name whatever state the committed tables are in."""

    def test_refusal_named_with_unreadable_history(self):
        real = ps.load_handler_history

        def broken(*_args, **_kwargs):
            raise OSError("no table")

        ps.load_handler_history = broken
        try:
            refused_field(self, request(tiers=[tier(cards=[])]), "tiers[0].cards")
        finally:
            ps.load_handler_history = real


class HistoryTable(unittest.TestCase):
    """Generated from git, current at HEAD; a hand edit or a missed handler change fails (§5)."""

    def test_table_is_generated(self):
        out = subprocess.run([sys.executable, os.path.join(SERVICE, "generate_handler_history.py"),
                              "--check"], capture_output=True, text=True, cwd=REPO_ROOT)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)

    def _check(self, path, cwd=None):
        script = os.path.join(cwd or REPO_ROOT, "service", "generate_handler_history.py")
        return subprocess.run([sys.executable, script, "--check", path], capture_output=True,
                              text=True)

    def test_check_fails_on_a_hand_edit(self):
        table = ps.load_handler_history()
        edits = {
            "tree changed": lambda t: t["commits"].__setitem__(list(t["commits"])[3], "0" * 40),
            "commit removed": lambda t: t["commits"].pop(list(t["commits"])[3]),
            "ref changed": lambda t: t.__setitem__("ref", "other"),
            "newest off main": lambda t: t.__setitem__("newest", SHA_B),
        }
        for label, edit in edits.items():
            edited = json.loads(json.dumps(table))
            edit(edited)
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
                handle.write(json.dumps(edited, indent=2) + "\n")
            try:
                self.assertEqual(self._check(handle.name).returncode, 1, label)
            finally:
                os.remove(handle.name)

    def test_shallow_clone_stops(self):
        clone = tempfile.mkdtemp(prefix="shallow_")
        try:
            subprocess.run(["git", "clone", "-q", "--depth", "3", "file://" + REPO_ROOT, clone],
                           check=True, capture_output=True)
            # The generator under test is THIS tree's, run against the shallow clone's history.
            os.makedirs(os.path.join(clone, "service"), exist_ok=True)
            script = os.path.join(clone, "service", "generate_handler_history.py")
            shutil.copy(os.path.join(SERVICE, "generate_handler_history.py"), script)
            for args in ([], ["--check"]):
                out = subprocess.run([sys.executable, script] + args, capture_output=True,
                                     text=True)
                self.assertEqual(out.returncode, 2, (args, out.stdout, out.stderr))
                self.assertIn("shallow", out.stderr)
        finally:
            shutil.rmtree(clone)

    def test_newest_is_current_at_head(self):
        table = ps.load_handler_history()
        newest = table["newest"]
        self.assertEqual(subprocess.run(["git", "-C", REPO_ROOT, "merge-base", "--is-ancestor",
                                         newest, "HEAD"]).returncode, 0)
        # EVERY commit after newest, not the net diff: a change and its revert net to nothing and
        # still ran on some tier (review F1 on P1d).
        self.assertEqual(_git("log", "--format=%H", "{}..HEAD".format(newest), "--", "handler/"),
                         "")
        self.assertEqual(table["commits"][newest], _git("rev-parse", "HEAD:handler").strip())
        self.assertEqual(ps.service_handler_tree(table), table["commits"][newest])


class Projection(unittest.TestCase):
    """For the same inputs, every §3b field on the wire equals the core output."""

    TIER_FIELDS = ("tier", "fits_any", "fits_all", "output_width", "output_height",
                   "registry_version", "commit", "handler_tree", "worker_handler_tree",
                   "handler_match")
    CARD_FIELDS = ("gpu_name", "label", "fits", "predicted_seconds", "prediction_basis",
                   "rate_from", "reason", "residency", "anchored", "binding_phase", "quality",
                   "hardware_used", "vram_source", "vram_stats", "resolved_from")

    def _through_http(self, body):
        from service import app
        status, payload = app.route("POST", "/estimate", json.dumps(body).encode("utf-8"),
                                    commit=SHA_A)
        self.assertEqual(status, 200, payload)
        return payload

    def test_every_field_on_the_wire(self):
        cases = [
            request(),
            request(job(target_short_edge_px=4320), [tier(host_ram_gb=16.0)]),
            request(tiers=[tier(cards=[card(MIG, vram_total_gb=44.5), card(A40, label="idle")],
                                worker_commit=_same_tree_commit())]),
            request(tiers=[tier(worker_commit=OTHER_TREE_COMMIT),
                           tier(tier="t2", cards=[card(B200)], host_ram_gb=377.0)]),
            request(job(target_short_edge_px=4320), [tier(cards=[card(A40)])]),
            request(tiers=[tier(cards=[card(MIG), card(A40, vram_total_gb=44.7),
                                       card(A40, vram_total_gb=44.43, vram_free_gb=44.08)])]),
        ]
        matches, sources = set(), set()
        for body in cases:
            # HTTP first, on the pristine body: a core that mutated the caller's request would
            # otherwise be invisible to the one case positioned to see it (review F9).
            wire = self._through_http(copy.deepcopy(body))["tiers"]
            cores = ps.estimate_core(body, commit=SHA_A)
            self.assertEqual(len(wire), len(cores))
            for core, entry in zip(cores, wire):
                self.assertEqual(sorted(entry), sorted(self.TIER_FIELDS + ("cards",)))
                matches.add(entry["handler_match"])
                for field in self.TIER_FIELDS:
                    self.assertIs(type(entry[field]), type(core[field]), field)
                    self.assertEqual(entry[field], core[field], field)
                self.assertEqual(len(entry["cards"]), len(core["cards"]))
                for got, answered in zip(entry["cards"], core["cards"]):
                    self.assertEqual(sorted(got), sorted(self.CARD_FIELDS))
                    self.assertNotIn("rationale", got)
                    self.assertNotIn("planner_verdict", got)
                    sources.add(got["vram_source"])
                    for field in self.CARD_FIELDS:
                        self.assertIs(type(got[field]), type(answered[field]), field)
                        self.assertEqual(got[field], answered[field], field)
        self.assertEqual(matches, {None, True, False})
        self.assertEqual(sources, {"given", "table", "nearest_memory", "pool_floor"})

    def test_refusal_on_the_wire_names_the_field(self):
        from service import app
        body = request(tiers=[tier(cards=[])])
        status, payload = app.route("POST", "/estimate", json.dumps(body).encode("utf-8"),
                                    commit=SHA_A)
        self.assertEqual(status, 400)
        self.assertEqual(payload["refused"]["field"], "tiers[0].cards")

    def test_version(self):
        from service import app
        status, payload = app.route("GET", "/version", b"", commit=SHA_A)
        self.assertEqual(status, 200)
        self.assertEqual(payload["commit"], SHA_A)
        self.assertEqual(payload["registry_version"], planner.REGISTRY_VERSION)
        self.assertEqual(payload["calibration_rows"], len(estimator.load_calibration()))
        with open(os.path.join(SERVICE, "vram_table.json"), encoding="utf-8") as handle:
            self.assertEqual(payload["vram_table_corpus"], json.load(handle)["corpus"])
        table = ps.load_handler_history()
        self.assertEqual(payload["handler_tree"], table["commits"][table["newest"]])
        self.assertEqual(payload["handler_tree"], _git("rev-parse", "HEAD:handler").strip())
        self.assertEqual(payload["handler_history_newest"], table["newest"])


class Isolation(unittest.TestCase):
    """A change confined to service/ leaves the image's build context byte-identical."""

    def test_service_commits_touch_only_service(self):
        commits = _git("log", "--format=%H", "--", "service/").split()
        for sha in commits:
            paths = _git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root",
                         sha).split()
            outside = [p for p in paths if not p.startswith("service/")]
            self.assertEqual(outside, [], "{} touches outside service/".format(sha))
            parent = _git("rev-list", "--parents", "-n", "1", sha).split()[1:]
            if parent:
                self.assertEqual(_git("rev-parse", "{}:handler".format(sha)),
                                 _git("rev-parse", "{}:handler".format(parent[0])), sha)

    def test_working_tree_handler_untouched(self):
        self.assertEqual(_git("status", "--porcelain", "--", "handler/"), "")

    def test_image_context_is_handler(self):
        with open(os.path.join(REPO_ROOT, ".github", "workflows", "docker-publish.yml"),
                  encoding="utf-8") as handle:
            workflow = handle.read()
        contexts = set(re.findall(r"^\s*context:\s*(\S+)", workflow, re.M))
        self.assertEqual(contexts, {"./handler"})


class Agnostic(unittest.TestCase):
    """No module under service/ imports or names a provider client, URL or credential."""

    FORBIDDEN = re.compile(
        r"runpod|replicate|vast\.ai|lambdalabs|boto3|botocore|requests|httpx|urllib\.request|"
        r"https?://|api[_-]?key|secret|token|password|credential",
        re.I)

    def test_no_provider_names(self):
        hits = []
        for directory, _dirs, files in os.walk(SERVICE):
            if "tests" in os.path.relpath(directory, SERVICE).split(os.sep):
                continue
            for name in files:
                if not name.endswith((".py", ".txt", ".json", ".toml", "Dockerfile")):
                    continue
                path = os.path.join(directory, name)
                with open(path, encoding="utf-8") as handle:
                    for number, line in enumerate(handle, 1):
                        if self.FORBIDDEN.search(line):
                            hits.append("{}:{}: {}".format(
                                os.path.relpath(path, REPO_ROOT), number, line.strip()))
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
