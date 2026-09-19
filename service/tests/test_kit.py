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

#: The memory half of the in-image card table (W6), as the service resolves it.
TABLE = ps.memory_cards(estimator.load_card_table())

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
    body = {"tier": "t1", "host_ram_gb": 46.57, "cards": [card()]}
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


def answer_with(card_table, body, index=0):
    """One card's answer against `card_table` handed to the core (W6)."""
    return ps.estimate_core(body, commit=None, card_table=card_table)[0]["cards"][index]


def refused_field(test, body, field):
    with test.assertRaises(ps.Refusal) as caught:
        ps.estimate_core(body, commit=None)
    test.assertEqual(caught.exception.field, field, caught.exception.message)


def with_card_table(document, call):
    """`call()` with the worker's in-image card table replaced by `document` (W6)."""
    saved = estimator.load_card_table
    estimator.load_card_table = lambda *a, **k: json.loads(json.dumps(document))
    try:
        return call()
    finally:
        estimator.load_card_table = saved


def shipped_rates():
    return estimator.table_rates(estimator.load_card_table())


def _git(*args):
    return subprocess.run(["git", "-C", REPO_ROOT] + list(args), capture_output=True,
                          text=True, check=True).stdout


class Parity(unittest.TestCase):
    """cd622500: the job's request, its hardware block as its one card, reproduces its rationale."""

    RECORD = "2026-08-31_01REAL20260831T15562300.json"
    #: The two keys the handler adds after `estimator.plan` returns (§7, ruled on C3).
    HANDLER_ADDED = ("deadline", "effective_temporal_window")
    #: **Keys W5 P3 re-derives** (ruled 2026-09-18): the record was written by the lookup, and the
    #: simple rate replaced it. Checked against the worker's own derivation instead, so parity
    #: still means the service answers exactly what the worker would — today.
    P3_REDERIVED = ("predicted_seconds", "seconds_per_frame")

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
        compared = sorted(k for k in recorded
                          if k not in self.HANDLER_ADDED and k not in self.P3_REDERIVED)
        mismatched = {k: (recorded[k], fresh.get(k, "<absent>"))
                      for k in compared if fresh.get(k, object()) != recorded[k]}
        self.assertEqual(mismatched, {}, "recorded vs fresh")
        only_fresh = sorted(set(fresh) - set(recorded))
        print("\n  parity: {} recorded keys equal; excluded {}; only in fresh output: {}".format(
            len(compared), list(self.HANDLER_ADDED), only_fresh))
        # **The record's number is the lookup's; the service's is the rate's** (W5 P3). The rate
        # is the A40's batched one, read from the card table (W6).
        self.assertEqual(recorded["predicted_seconds"], 897.1)
        rate = shipped_rates()[(A40, estimator.BATCHED)]
        pixels = fresh["output_width"] * fresh["output_height"]
        self.assertEqual(fresh["seconds_per_frame"], round(pixels / 1e6 / rate["mpx_per_s"], 4))
        # **Tied to the frame count, not to itself** (review, P3): the service's number is the
        # rate over the delivered plane times the source's frames, so a wrong frame count or a
        # term added on the way out fails here.
        self.assertEqual(got["predicted_seconds"],
                         round(pixels / 1e6 / rate["mpx_per_s"] * source["estimated_frames"], 1))
        # Every field the wire carries is the worker's own, key for key (review F2, F3 on P1).
        for field, key in PROJECTED_FROM_RATIONALE:
            expected = fresh[key] if key in self.P3_REDERIVED else recorded[key]
            self.assertEqual(got[field], expected, field)
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
        entry = one(request(), commit=SHA_A)
        for field in ("output_width", "output_height", "registry_version", "commit"):
            self.assertIn(field, entry)
            self.assertNotIn(field, entry["cards"][0])

    def test_several_tiers_answer_in_order(self):
        entries = ps.estimate_core(request(tiers=[tier(), tier(tier="t2", host_ram_gb=377.0,
                                                              cards=[card(B200)])]), commit=None)
        self.assertEqual([e["tier"] for e in entries], ["t1", "t2"])
        self.assertEqual([c["gpu_name"] for e in entries for c in e["cards"]], [A40, B200])


class RateFrom(unittest.TestCase):
    """Whose measurement the time is (§4a-ii): null where it is this card's own."""

    def test_one_number_on_two_cards_says_where_it_came_from(self):
        # **W5 P3**: a rate is per card and regime, so the lookup's "one number on three cards"
        # (the pixel band held only A40 rows) is gone. What survives is a card with no rows in
        # the job's REGIME: the B200 has no window-1 rate, so a still on it borrows the slowest
        # measured card's — the A40's — and the two answer the same number.
        cards = [card(A40), card(B200)]
        entry = one(request(job(frames=1, is_still=True, source_width=749, source_height=500, target_short_edge_px=1920), [tier(host_ram_gb=377.0, cards=cards)]))
        seconds = [c["predicted_seconds"] for c in entry["cards"]]
        self.assertEqual(len(set(seconds)), 1, "the case exists because the numbers are equal")
        rates = [c["rate_from"] for c in entry["cards"]]
        self.assertIsNone(rates[0], "the A40 owns the rows this rate came from")
        got = entry["cards"][1]
        self.assertEqual(got["rate_from"]["running_on"], B200)
        self.assertEqual(got["rate_from"]["rows_measured_on"], [A40])
        self.assertEqual(got["prediction_basis"], "borrowed")

    def test_rate_from_is_the_workers_own_keys(self):
        got = answer(request(job(frames=1, is_still=True, source_width=749, source_height=500, target_short_edge_px=1920), [tier(host_ram_gb=377.0, cards=[card(B200)])]))
        self.assertEqual(got["rate_from"], got["rationale"]["timing_from_another_card"])

    def test_tiling_half(self):
        # **The RTX PRO 6000 has no `high` row** (W5 P3 banked `high` rows for the A40 and the
        # H200): its rate is the default-tiling rows', and the label says so.
        got = answer(request(job(tile_quality="high"),
                             [tier(host_ram_gb=377.0, cards=[card(BLACKWELL)])]))
        self.assertEqual(got["rate_from"]["tiling"],
                         got["rationale"]["timing_from_another_tiling"])
        self.assertNotIn("running_on", got["rate_from"])
        self.assertEqual(got["prediction_basis"], "borrowed")

    def test_both_halves_at_once(self):
        # A `high` still on the B200: no window-1 rate of its own, and no `high` row among the
        # A40 window-1 rows it borrows.
        got = answer(request(job(frames=1, is_still=True, source_width=749, source_height=500,
                                 target_short_edge_px=1920, tile_quality="high"),
                             [tier(host_ram_gb=377.0, cards=[card(B200)])]))
        self.assertEqual(got["rate_from"]["running_on"], B200)
        self.assertEqual(got["rate_from"]["rows_measured_on"], [A40])
        self.assertEqual(got["rate_from"]["tiling"]["running_at"], "high")

    def test_borrowed_in_either_sense(self):
        # Memory borrowed; time NOT borrowed any more (J5, f05b28d): the A40's rows used to price
        # a MIG resolved to the A40. Memory says nothing about speed, so the time is withheld.
        got = answer(request(tiers=[tier(cards=[card(MIG, vram_total_gb=44.5)])]))
        self.assertIsNone(got["prediction_basis"])
        self.assertIsNotNone(got["resolved_from"])
        self.assertIsNone(got["rate_from"])
        self.assertEqual(got["timing_unavailable"]["running_on"], MIG)

    def test_refused_card_has_no_rate(self):
        got = answer(request(job(target_short_edge_px=4320)))
        self.assertFalse(got["fits"])
        self.assertIsNone(got["rate_from"])


class AbsenceNamesTheSentCard(unittest.TestCase):
    """W5 F4 (ruled 2026-09-19): an absence names the card CF SENT, with the worker's own reason
    and regime. **Since W6 the worker times the card under that name itself** (§4a-ii), so there
    is nothing left to re-point.

    Reached with a patched card table: the L40S is PRICED (batched only) and carries no memory,
    so a nominal resolves its memory to the A40; and no card holds a window-1 rate, so a
    still reaches the no-rate-in-regime absence.
    """

    L40S = "NVIDIA L40S"
    # The L40S is priced and carries no memory, so its memory resolves elsewhere (W6).
    CARDS = {"generated_utc": "test", "cards": {
        A40: {"vram_total_gb": 44.34, "vram_free_gb": 43.72,
              "mpx_per_s": {"batched": 0.6, "unbatched": None}},
        L40S: {"vram_total_gb": None, "vram_free_gb": None,
               "mpx_per_s": {"batched": 0.6, "unbatched": None}}}}

    def test_the_absence_names_the_card_cf_sent(self):
        self.assertNotIn(self.L40S, TABLE)
        got = with_card_table(self.CARDS, lambda: answer(request(
            job(frames=1, is_still=True, source_width=749, source_height=500,
                target_short_edge_px=1920),
            [tier(cards=[card(self.L40S, vram_total_gb=44.5)])])))
        self.assertEqual(got["resolved_from"], {"card": self.L40S, "measured": A40})
        self.assertIsNone(got["predicted_seconds"])
        absent = got["timing_unavailable"] or {}
        self.assertEqual(absent.get("running_on"), self.L40S,
                         "CF asked about the L40S and was told about the A40")
        # The worker's reason stands: this is F4's absence, not J5's.
        self.assertEqual(absent.get("regime"), estimator.UNBATCHED)
        self.assertIn("regime", absent.get("why", ""))


class OwnTimeSubstitutedMemory(unittest.TestCase):
    """**A memory substitution is not a time substitution** (§4a-ii; W6, the recurrence the gate
    filed). A card the card table PRICES but gives no memory resolves its memory to another
    card and is timed at ITS OWN rate — it used to be planned, and so priced, under the
    substitute's name.

    **`borrowed` WITH `rate_from` NULL IS THE DESIGNED READING, NOT A CONTRADICTION** (§4a-i,
    C14): `borrowed` covers both senses, `rate_from` null says the time is this card's own, and
    `resolved_from` says the memory was not. Do not "fix" it.
    """

    L40S = "NVIDIA L40S"
    L40S_RATE = 0.9

    def _cards(self):
        document = json.loads(json.dumps(estimator.load_card_table()))
        # Priced, and no memory: the case two tables made and one table still can (W6).
        document["cards"][self.L40S] = {"vram_total_gb": None, "vram_free_gb": None,
                                        "mpx_per_s": {"batched": self.L40S_RATE,
                                                      "unbatched": None}}
        return document

    def test_own_rate_under_substituted_memory(self):
        self.assertNotIn(self.L40S, TABLE)
        got = with_card_table(self._cards(), lambda: answer(request(
            tiers=[tier(cards=[card(self.L40S, vram_total_gb=44.5)])])))
        self.assertEqual(got["resolved_from"], {"card": self.L40S, "measured": A40})
        self.assertEqual(got["hardware_used"]["gpu_name"], A40)
        width, height = estimator.output_dimensions(1920, 1080, 1480)
        self.assertEqual(got["predicted_seconds"],
                         round(width * height / 1e6 / self.L40S_RATE * 90, 1),
                         "the L40S's own rate, not the A40's")
        self.assertIsNone(got["rate_from"])
        self.assertIsNone(got["timing_unavailable"])
        self.assertEqual(got["prediction_basis"], "borrowed")


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


class MaxTarget(unittest.TestCase):
    """§3e: on a refusal, the largest target that plans on this card — the worker's own walk,
    reported and never taken; null when nothing smaller plans."""

    def _workers_walk(self, got, body_job):
        """What the worker itself reports for the same card and shape: estimator.plan's own
        refusal, read from its `reduce_target_resolution` option."""
        worker_job = {"target_short_edge_px": ps.read_job({"job": body_job})["target"],
                      "source_width": body_job["source_width"],
                      "source_height": body_job["source_height"],
                      "estimated_frames": body_job["frames"], "still": body_job["is_still"],
                      "tile_quality": body_job["tile_quality"],
                      "schedule": body_job["schedule"]}
        with self.assertRaises(estimator.WorkerError) as caught:
            estimator.plan(worker_job, got["hardware_used"])
        return (caught.exception.shortfall or {}).get("options") or []

    def test_equals_the_workers_walk(self):
        body_job = job(target_short_edge_px=4320)
        got = answer(request(body_job))
        self.assertFalse(got["fits"])
        options = self._workers_walk(got, body_job)
        self.assertEqual(len(options), 1, options)
        edge = int(options[0]["how"].rsplit("=", 1)[1])
        target = got["max_target"]
        self.assertEqual(target["target_short_edge_px"], edge)
        self.assertEqual((target["output_width"], target["output_height"]),
                         estimator.output_dimensions(1920, 1080, edge))
        self.assertIn("delivers {}x{} ".format(target["output_width"], target["output_height"]),
                      options[0]["cost"])
        self.assertIn("at a window of {} frames".format(target["best_window"]),
                      options[0]["cost"])
        # **Reported, never taken**: the card still refuses at what the caller asked for.
        self.assertLess(edge, 4320)

    def test_nothing_smaller_plans_is_null(self):
        body_job = job(target_short_edge_px=4320)
        got = answer(request(body_job, [tier(cards=[card(A40, vram_total_gb=1.5,
                                                         vram_free_gb=1.0)])]))
        self.assertFalse(got["fits"])
        self.assertEqual(self._workers_walk(got, body_job), [])
        self.assertIsNone(got["max_target"])

    def test_a_fitting_card_is_null(self):
        got = answer(request())
        self.assertTrue(got["fits"])
        self.assertIsNone(got["max_target"])

    def test_an_output_size_request_gets_a_short_edge(self):
        body_job = job(target_short_edge_px=None, output_size={"width": 7680, "height": 4320})
        del body_job["target_short_edge_px"]
        got = answer(request(body_job))
        self.assertFalse(got["fits"])
        self.assertEqual(sorted(got["max_target"]),
                         ["best_window", "output_height", "output_width",
                          "target_short_edge_px"])
        # **The worker's answer for the covering short edge, not a canvas** (review, P2).
        options = self._workers_walk(got, body_job)
        edge = int(options[0]["how"].rsplit("=", 1)[1])
        self.assertEqual(got["max_target"]["target_short_edge_px"], edge)
        self.assertLess(edge, ps.read_job({"job": body_job})["target"])
        self.assertEqual((got["max_target"]["output_width"], got["max_target"]["output_height"]),
                         estimator.output_dimensions(1920, 1080, edge))

    def test_on_the_wire_for_every_card(self):
        entry = one(request(job(target_short_edge_px=4320),
                            [tier(cards=[card(A40), card(A40, vram_total_gb=1.5,
                                                         vram_free_gb=1.0)])]))
        wire = ps.project(entry)
        for got in wire["cards"]:
            self.assertIn("max_target", got)

    def test_a_changed_worker_option_is_loud(self):
        """The service reads the worker's own option; if its shape moves, the answer must
        fail rather than guess."""
        saved = estimator._terminal_options
        # The real sentence for 1024 on this card, so each case below breaks ONE half of it.
        width, height = estimator.output_dimensions(1920, 1080, 1024)
        window = planner.plan((1920, 1080), 90, 1024, usable_gb=estimator._usable_vram(
            answer(request())["hardware_used"]), host_ram_gb=46.57, gpu_name=A40)["w"]
        true_cost = "delivers {}x{} instead of 2x2, at a window of {} frames".format(
            width, height, window)
        cases = (
            [{"option": "reduce_target_resolution", "how": "resend smaller", "cost": "?"}],
            # Right dimensions, wrong window — and right window, wrong dimensions.
            [{"option": "reduce_target_resolution", "how": "resend with target_short_edge_px=1024",
              "cost": true_cost.replace("window of {}".format(window), "window of 999")}],
            [{"option": "reduce_target_resolution", "how": "resend with target_short_edge_px=1024",
              "cost": true_cost.replace("{}x{} ".format(width, height), "1x1 ")}],
            # A target that does not plan on this card, with the sentence its refused plan WOULD
            # render — so only the action check can object.
            [{"option": "reduce_target_resolution", "how": "resend with target_short_edge_px=8192",
              "cost": "delivers {}x{} instead of 2x2, at a window of {} frames".format(
                  *estimator.output_dimensions(1920, 1080, 8192),
                  planner.plan((1920, 1080), 90, 8192, usable_gb=estimator._usable_vram(
                      answer(request())["hardware_used"]), host_ram_gb=46.57,
                      gpu_name=A40).get("w"))}],
            # A renamed option, where the list is not empty.
            [{"option": "shrink", "how": "resend with target_short_edge_px=1024",
              "cost": true_cost}],
        )
        for listed in cases:
            estimator._terminal_options = lambda *a, **k: [dict(o) for o in listed]
            try:
                # **Its own raise, not any RuntimeError** (review, P2): the verdict check and
                # TableUnusable are RuntimeErrors too.
                with self.assertRaisesRegex(
                        RuntimeError, "reduce_target_resolution|re-derive|changed shape",
                        msg=repr(listed)):
                    answer(request(job(target_short_edge_px=4320)))
            finally:
                estimator._terminal_options = saved

    def test_a_refusal_without_its_options_is_loud(self):
        """The worker's refusal with its options list removed must fail, not read as "nothing
        smaller plans"."""
        real = estimator.plan

        def stripped(*a, **k):
            try:
                return real(*a, **k)
            except estimator.WorkerError as refusal:
                (refusal.shortfall or {}).pop("options", None)
                raise
        estimator.plan = stripped
        try:
            with self.assertRaisesRegex(RuntimeError, "carries no options list"):
                answer(request(job(target_short_edge_px=4320)))
        finally:
            estimator.plan = real

    def test_step_for_step_even_off_the_default_tiling(self):
        """The walk prices at the default tiling whatever the request says; the service reports
        that walk step for step. (Swept 2026-09-18: tile_quality moved neither the action nor
        the window at any target, so the two agree today — this pins that they keep agreeing.)"""
        body_job = job(target_short_edge_px=4320, tile_quality="high")
        got = answer(request(body_job))
        self.assertFalse(got["fits"])
        options = self._workers_walk(got, body_job)
        self.assertEqual(got["max_target"]["target_short_edge_px"],
                         int(options[0]["how"].rsplit("=", 1)[1]))
        self.assertIn("at a window of {} frames".format(got["max_target"]["best_window"]),
                      options[0]["cost"])


class TimingUnavailable(unittest.TestCase):
    """J5: a card no row was measured on gets a NAMED absence of time — and still fits, still
    has quality. A null time is not a refusal, and a refusal is still a refusal."""

    def test_an_unseen_card_fits_with_no_time_and_says_why(self):
        got = answer(request(tiers=[tier(cards=[card(MIG, vram_total_gb=44.5,
                                                     vram_free_gb=44.0)])]))
        self.assertTrue(got["fits"])
        self.assertIsNotNone(got["quality"]["best_window"])
        self.assertIsNone(got["predicted_seconds"])
        self.assertIsNone(got["prediction_basis"])
        self.assertEqual(got["timing_unavailable"]["running_on"], MIG)
        self.assertIn(A40, got["timing_unavailable"]["cards_with_rows"])
        self.assertNotIn(MIG, got["timing_unavailable"]["cards_with_rows"])

    def test_an_unseen_card_that_cannot_hold_it_still_refuses(self):
        got = answer(request(job(target_short_edge_px=4320),
                             [tier(cards=[card(MIG, vram_total_gb=44.5, vram_free_gb=44.0)])]))
        self.assertFalse(got["fits"])
        self.assertIsNone(got["timing_unavailable"])

    def test_memory_substituted_and_time_withheld_are_both_visible(self):
        """With no VRAM reading, memory resolves to a measured card (§4a-i) — and time is still
        judged on the name CF SENT: withheld, and said so, beside the memory substitution."""
        got = answer(request(tiers=[tier(cards=[card(MIG, vram_total_gb=44.5)])]))
        self.assertTrue(got["fits"])
        self.assertIsNotNone(got["resolved_from"])
        self.assertNotEqual(got["hardware_used"]["gpu_name"], MIG)
        self.assertIsNone(got["predicted_seconds"])
        self.assertIsNone(got["prediction_basis"])
        self.assertIsNone(got["rate_from"])
        self.assertEqual(got["timing_unavailable"]["running_on"], MIG)
        self.assertIsNotNone(got["quality"]["best_window"])

    def test_the_workers_own_reason_is_kept_when_nothing_was_substituted(self):
        """A card sent with a full reading plans under its own name, so the worker's own
        timing_unavailable is the answer — the service's "memory was resolved" would be false."""
        got = answer(request(tiers=[tier(cards=[card(MIG, vram_total_gb=44.5,
                                                     vram_free_gb=44.0)])]))
        self.assertEqual(got["vram_source"], "given")
        self.assertNotIn("memory was resolved", got["timing_unavailable"]["why"])

    def test_a_measured_card_is_unchanged(self):
        got = answer(request())
        self.assertTrue(got["fits"])
        self.assertIsNotNone(got["predicted_seconds"])
        self.assertIsNone(got["timing_unavailable"])


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
        # Withheld time must carry no one else's rate beside it (review, J5) — asserted below on a
        # resolved card that borrows (W5 P3 gave the H200 its own batched rate).
        self.assertIsNone(got["rate_from"])
        # **Since J5's service ruling (f05b28d) the MIG's TIME is withheld**, judged on the name
        # CF sent; the memory resolution above is unchanged.
        self.assertIsNone(got["prediction_basis"])
        self.assertEqual(got["timing_unavailable"]["running_on"], MIG)

    def test_withheld_time_carries_no_borrowed_rate(self):
        # **The half above, on a card that borrows** (W5 P3): a still resolved onto the B200,
        # which has no window-1 rate, so the worker's rationale names the A40's rows — and the
        # MIG's withheld time must not carry them.
        got = answer(request(job(frames=1, is_still=True, source_width=749, source_height=500, target_short_edge_px=1920),
                             [tier(host_ram_gb=377.0, cards=[card(MIG), card(B200, vram_total_gb=179.0)])]))
        self.assertEqual(got["resolved_from"], {"card": MIG, "measured": B200})
        # Since W6 the worker times under the MIG's name, so no borrowed rate exists at all.
        self.assertNotIn("timing_from_another_card", got["rationale"])
        self.assertIsNone(got["rate_from"])
        self.assertIsNone(got["prediction_basis"])
        self.assertEqual(got["timing_unavailable"]["running_on"], MIG)

    def test_floor_card_time_is_judged_on_the_name_sent(self):
        # The floor lands on the A40, which the card table prices. The time is still the MIG's
        # to answer, and the worker itself withholds it under the MIG's name (W6).
        cards = [card(MIG), card(A40, vram_total_gb=44.7)]
        got = answer(request(tiers=[tier(cards=cards)]))
        self.assertEqual(got["vram_source"], "pool_floor")
        self.assertEqual(got["hardware_used"]["gpu_name"], A40)
        self.assertEqual(got["rationale"]["timing_unavailable"]["running_on"], MIG)
        # **Since J5's service ruling (f05b28d) the MIG's TIME is withheld**, judged on the name
        # CF sent; the memory resolution above is unchanged.
        self.assertIsNone(got["prediction_basis"])
        self.assertEqual(got["timing_unavailable"]["running_on"], MIG)

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
        # **Since J5's service ruling (f05b28d) the MIG's TIME is withheld**, judged on the name
        # CF sent — by the worker itself since W6; the memory resolution above is unchanged.
        self.assertIsNone(got["prediction_basis"])
        self.assertEqual(got["timing_unavailable"]["running_on"], MIG)

    def test_nominal_near_a_bigger_card(self):
        got = answer(request(tiers=[tier(host_ram_gb=377.0,
                                         cards=[card(MIG, vram_total_gb=96.0)])]))
        self.assertEqual(got["resolved_from"], {"card": MIG, "measured": BLACKWELL})

    def test_tie_takes_lower_memory(self):
        low, high = TABLE[A40]["vram_total_gb"], TABLE[BLACKWELL]["vram_total_gb"]
        table = {A40: TABLE[A40], BLACKWELL: TABLE[BLACKWELL]}
        self.assertEqual(ps.nearest_measured((low + high) / 2.0, table), A40)


class NoVramStats(unittest.TestCase):
    """W6 Q4 (ruled by CF, 2026-09-19): `vram_stats` is dropped outright — nothing read it — and
    the card table carries total and free only. Memory comes from that table's figures."""

    def test_no_vram_stats_anywhere(self):
        from service import app
        body = request(tiers=[tier(cards=[card(A40), card(MIG, vram_total_gb=44.5), card(MIG)])])
        status, payload = app.route("POST", "/estimate", json.dumps(body).encode("utf-8"),
                                    commit=None)
        self.assertEqual(status, 200, payload)
        for got in list(one(body)["cards"]) + list(payload["tiers"][0]["cards"]):
            self.assertNotIn("vram_stats", got)
        self.assertNotIn("vram_stats", ps.CARD_WIRE_FIELDS)

    def test_memory_is_the_card_tables(self):
        shipped = estimator.load_card_table()["cards"]
        got = answer(request(tiers=[tier(cards=[card(A40)])]))
        self.assertEqual(got["vram_source"], "table")
        self.assertEqual((got["hardware_used"]["vram_total_gb"],
                          got["hardware_used"]["vram_free_gb"]),
                         (shipped[A40]["vram_total_gb"], shipped[A40]["vram_free_gb"]))

    def test_a_card_with_a_rate_and_no_memory_is_not_measured_for_memory(self):
        document = json.loads(json.dumps(estimator.load_card_table()))
        document["cards"]["NVIDIA L40S"] = {"vram_total_gb": None, "vram_free_gb": None,
                                            "mpx_per_s": {"batched": 0.9, "unbatched": None}}
        self.assertNotIn("NVIDIA L40S", ps.memory_cards(document))
        got = answer_with(document, request(tiers=[tier(cards=[card("NVIDIA L40S")])]))
        self.assertEqual(got["vram_source"], "pool_floor")
        # Its time is still its own (§4a-ii).
        self.assertIsNone(got["rate_from"])
        self.assertIsNone(got["timing_unavailable"])

    def test_the_old_tables_are_gone(self):
        for name in ("vram_table.json", "generate_vram_table.py"):
            self.assertFalse(os.path.exists(os.path.join(SERVICE, name)), name)
        self.assertFalse(os.path.exists(os.path.join(REPO_ROOT, "handler", "calibration.json")))


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
                             card_table={"cards": {A40: {"mpx_per_s": {"batched": 0.6}}}})
        # **An in-image table that fails its check, or is absent, is the same deploy fault.**
        for broken in (None, {"cards": {}}):
            status, payload = with_card_table(broken, lambda: app.route(
                "POST", "/estimate", json.dumps(request()).encode("utf-8"), commit=None))
            self.assertEqual(status, 503, broken)
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

    def test_still_timing_is_pinned(self):
        """**W1 J10 must not move the service's answer for a still** (gate, 2026-09-18). The
        service already sends frames 1 for a still (planner_service.py:307), so the worker-side
        fix changes nothing here — and a silent move is the regression CF would see first.
        Pinned at cf-upscale-image bb39a9c, before J10."""
        got = one(request(job(frames=1, is_still=True, source_width=749, source_height=500,
                              target_short_edge_px=1920)))["cards"][0]
        # **Moved by W5 P3, by ruling** — 27.4 was the lookup's. Pinned now to the mechanism, not
        # to a number: the A40's window-1 rate over the delivered plane, one frame. A new table
        # moves it without failing this; a service that stops answering the worker's rate fails.
        rate = shipped_rates()[(A40, estimator.UNBATCHED)]
        width, height = estimator.output_dimensions(749, 500, 1920)
        self.assertEqual((got["predicted_seconds"], got["prediction_basis"]),
                         (round(width * height / 1e6 / rate["mpx_per_s"], 1), "measured"))


class CommitNotVerified(unittest.TestCase):
    """W6 item 3 (ruled by CF, 2026-09-19): the service does not compare commits.

    `worker_commit` is not read — a caller still sending it has it ignored, whatever it holds —
    and no tier answer carries `worker_handler_tree` or `handler_match`.
    """

    SENT = (None, "a" * 40, "abc1234", 7, "not a sha")

    def test_worker_commit_is_ignored(self):
        bare = one(request())
        for sent in self.SENT:
            with self.subTest(sent=sent):
                self.assertEqual(one(request(tiers=[tier(worker_commit=sent)])), bare)

    def test_absent_worker_commit_is_accepted(self):
        t = tier()
        t.pop("worker_commit", None)
        self.assertEqual(one(request(tiers=[t])), one(request()))

    def test_no_match_fields(self):
        from service import app
        body = request(tiers=[tier(worker_commit="a" * 40)])
        status, payload = app.route("POST", "/estimate", json.dumps(body).encode("utf-8"),
                                    commit=SHA_A)
        self.assertEqual(status, 200, payload)
        for entry in (one(body, commit=SHA_A), payload["tiers"][0]):
            for field in ("worker_handler_tree", "handler_match", "handler_tree"):
                self.assertNotIn(field, entry)
            self.assertEqual(entry["commit"], SHA_A)

    def test_version_carries_the_commit_and_no_tree(self):
        # Option (a), ruled by the gate: the tree's only source was the deleted history table.
        from service import app
        status, payload = app.route("GET", "/version", b"", commit=SHA_A)
        self.assertEqual(status, 200)
        self.assertEqual(payload["commit"], SHA_A)
        for field in ("handler_tree", "handler_history_newest"):
            self.assertNotIn(field, payload)

    def test_history_machinery_is_gone(self):
        for name in ("handler_history.json", "generate_handler_history.py"):
            self.assertFalse(os.path.exists(os.path.join(SERVICE, name)), name)
        for attr in ("load_handler_history", "service_handler_tree", "HANDLER_HISTORY_PATH"):
            self.assertFalse(hasattr(ps, attr), attr)


class ServiceCommit(unittest.TestCase):
    """The service's own commit is information, never a verdict (§5)."""

    def test_service_commit_is_information(self):
        with self.assertRaises(ValueError):
            ps.service_commit({"CF_PLANNER_COMMIT": "abc1234"})
        self.assertEqual(ps.service_commit({"CF_PLANNER_COMMIT": SHA_A}), SHA_A)
        self.assertIsNone(ps.service_commit({}))
        self.assertIsNone(one(request(), commit=None)["commit"])


class TablesAfterTheDoor(unittest.TestCase):
    """A malformed request is refused by name whatever state the committed tables are in."""

    def test_refusal_named_with_unreadable_table(self):
        with_card_table(None, lambda: refused_field(self, request(tiers=[tier(cards=[])]),
                                                    "tiers[0].cards"))


class Projection(unittest.TestCase):
    """For the same inputs, every §3b field on the wire equals the core output."""

    TIER_FIELDS = ("tier", "fits_any", "fits_all", "output_width", "output_height",
                   "registry_version", "commit")
    CARD_FIELDS = ("gpu_name", "label", "fits", "max_target", "predicted_seconds", "prediction_basis",
                   "rate_from", "timing_unavailable", "reason", "residency", "anchored", "binding_phase", "quality",
                   "hardware_used", "vram_source", "resolved_from")

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
            request(tiers=[tier(cards=[card(MIG, vram_total_gb=44.5), card(A40, label="idle")])]),
            request(tiers=[tier(), tier(tier="t2", cards=[card(B200)], host_ram_gb=377.0)]),
            request(job(target_short_edge_px=4320), [tier(cards=[card(A40)])]),
            request(tiers=[tier(cards=[card(MIG), card(A40, vram_total_gb=44.7),
                                       card(A40, vram_total_gb=44.43, vram_free_gb=44.08)])]),
        ]
        sources = set()
        for body in cases:
            # HTTP first, on the pristine body: a core that mutated the caller's request would
            # otherwise be invisible to the one case positioned to see it (review F9).
            wire = self._through_http(copy.deepcopy(body))["tiers"]
            cores = ps.estimate_core(body, commit=SHA_A)
            self.assertEqual(len(wire), len(cores))
            for core, entry in zip(cores, wire):
                self.assertEqual(sorted(entry), sorted(self.TIER_FIELDS + ("cards",)))
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
        shipped = estimator.load_card_table()
        self.assertEqual(payload["card_table_generated_utc"], shipped["generated_utc"])
        self.assertEqual(payload["card_table_cards_priced"],
                         len({name for name, _regime in shipped_rates()}))
        self.assertNotIn("calibration_rows", payload)
        self.assertNotIn("vram_table_corpus", payload)


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
