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
import subprocess
import sys
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

with open(os.path.join(SERVICE, "vram_table.json"), encoding="utf-8") as _handle:
    TABLE = json.load(_handle)["cards"]


#: Wire field -> the rationale key it is the worker's output of.
PROJECTED_FROM_RATIONALE = (
    ("predicted_seconds", "predicted_seconds"), ("residency", "residency"),
    ("best_window", "window"), ("ideal_window", "ideal_window"),
    ("binding_phase", "binding_phase"), ("anchored", "anchored"),
    ("prediction_basis", "prediction_basis"),
)


def job(**overrides):
    body = {"source_width": 1920, "source_height": 1080, "frames": 90, "is_still": False,
            "target_short_edge_px": 1480, "tile_quality": "default", "schedule": "max_window"}
    body.update(overrides)
    return body


def tier(**overrides):
    body = {"tier": "t1", "host_ram_gb": 46.57,
            "cards": [{"gpu_name": "NVIDIA A40", "vram_total_gb": 44.7}],
            "worker_commit": None, "idle": None, "last_run": None}
    body.update(overrides)
    return body


def request(job_body=None, tiers=None):
    return {"job": job_body or job(), "tiers": tiers or [tier()]}


def one(body, commit=None):
    return ps.estimate_core(body, commit=commit)[0]


def refused_field(test, body, field):
    with test.assertRaises(ps.Refusal) as caught:
        ps.estimate_core(body, commit=None)
    test.assertEqual(caught.exception.field, field, caught.exception.message)


class Parity(unittest.TestCase):
    """cd622500: the job's request with its own hardware block as idle reproduces its rationale."""

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
                  cards=[{"gpu_name": hw["gpu_name"], "vram_total_gb": hw["vram_total_gb"]}],
                  idle={k: hw[k] for k in ("gpu_name", "vram_total_gb", "vram_free_gb",
                                           "host_ram_gb")})])
        core = one(body)
        fresh, recorded = core["rationale"], record["rationale"]

        self.assertEqual(core["card_source"], "idle")
        self.assertEqual(core["vram_source"], "idle")
        compared = sorted(k for k in recorded if k not in self.HANDLER_ADDED)
        mismatched = {k: (recorded[k], fresh.get(k, "<absent>"))
                      for k in compared if fresh.get(k, object()) != recorded[k]}
        self.assertEqual(mismatched, {}, "recorded vs fresh")
        only_fresh = sorted(set(fresh) - set(recorded))
        print("\n  parity: {} recorded keys equal; excluded {}; only in fresh output: {}".format(
            len(compared), list(self.HANDLER_ADDED), only_fresh))
        self.assertEqual(core["predicted_seconds"], 897.1)
        self.assertEqual(recorded["predicted_seconds"], 897.1)
        # Every planning field the wire carries is the worker's own, key for key (review F2, F3).
        for field, key in PROJECTED_FROM_RATIONALE:
            self.assertEqual(core[field], recorded[key], field)
        self.assertEqual(core["prediction_basis"], "measured")


class Residency(unittest.TestCase):
    """On a refusal, planner.plan refuses too, and residency comes from that verdict."""

    def test_host_refusal_reports_route_up(self):
        # 8 GiB of host RAM: the host slice cannot hold the window, so a bigger machine is the answer.
        core = one(request(tiers=[tier(host_ram_gb=8.0)]))
        self.assertFalse(core["fits"])
        self.assertIn("host RAM", core["reason"])
        verdict = core["planner_verdict"]
        self.assertNotEqual(verdict["action"], "plan")
        self.assertEqual(core["residency"], "route_up")
        self.assertEqual(core["residency"], verdict["residency"])
        self.assertEqual(estimator._refusal_text(verdict["reason"]), core["reason"])
        self.assertEqual(core["anchored"], estimator._usable_vram(core["hardware_used"])
                         <= planner.ANCHORED_MAX_USABLE)

    def test_diverging_verdict_is_an_error(self):
        # If planner.plan refuses for a different reason than estimator.plan did, the residency
        # read is not from the same verdict, and the service must not answer (review F4).
        # Only the service's own call diverges: estimator.plan's internal planner calls are real.
        real_plan, real_estimate = planner.plan, estimator.plan
        state = {"estimator_done": False}

        def estimate_then_flag(*args, **kwargs):
            try:
                return real_estimate(*args, **kwargs)
            finally:
                state["estimator_done"] = True

        def other_reason(*args, **kwargs):
            answer = real_plan(*args, **kwargs)
            return dict(answer, reason="a different constraint") if state["estimator_done"] \
                else answer

        planner.plan, estimator.plan = other_reason, estimate_then_flag
        try:
            with self.assertRaises(RuntimeError):
                one(request(tiers=[tier(host_ram_gb=8.0)]))
        finally:
            planner.plan, estimator.plan = real_plan, real_estimate

    def test_vram_refusal_labels_follow_planner_fits(self):
        # An 8K target on the A40's table figures: the VRAM floor refuses, and that terminal
        # answer carries no residency. P1a (cf-planner.md §3b @5a8652f): residency and anchored
        # are what planner.fits answers for the same inputs.
        core = one(request(job(target_short_edge_px=4320)))
        self.assertFalse(core["fits"])
        verdict = core["planner_verdict"]
        self.assertNotEqual(verdict["action"], "plan")
        self.assertNotIn("residency", verdict)
        self.assertEqual(estimator._refusal_text(verdict["reason"]), core["reason"])
        hw = core["hardware_used"]
        fits = planner.fits((1920, 1080), 90, 4320, estimator._usable_vram(hw),
                            host_ram_gb=hw["host_ram_gb"], tile_quality="default",
                            gpu_name=hw["gpu_name"])
        self.assertFalse(fits["fits"])
        self.assertEqual(core["residency"], "route_up")
        self.assertEqual(core["residency"], fits["residency"])
        self.assertIsNotNone(core["anchored"])
        self.assertEqual(core["anchored"], fits["anchored"])

    def test_refusal_anchored_at_both_sides_of_the_span(self):
        # Review W1 on P1a: the A40 is anchored either way. The H200's table figures sit on the
        # boundary (total above ANCHORED_MAX_USABLE, usable at or below it), the B200 beyond it.
        for card, expected in (("NVIDIA H200", True), ("NVIDIA B200", False)):
            nominal = TABLE[card]["vram_total_gb"]
            core = one(request(tiers=[tier(host_ram_gb=8.0,
                                           cards=[{"gpu_name": card, "vram_total_gb": nominal}])]))
            self.assertFalse(core["fits"], card)
            self.assertEqual(core["vram_source"], "table", card)
            hw = core["hardware_used"]
            fits = planner.fits((1920, 1080), 90, 1480, estimator._usable_vram(hw),
                                host_ram_gb=hw["host_ram_gb"], tile_quality="default",
                                gpu_name=hw["gpu_name"])
            self.assertFalse(fits["fits"], card)
            self.assertIs(core["anchored"], expected, card)
            self.assertEqual(core["anchored"], fits["anchored"], card)
            self.assertEqual(core["residency"], fits["residency"], card)

    def test_refusal_anchored_on_the_boundary(self):
        # A banked H200 reading (free 139.07) leaves usable exactly at ANCHORED_MAX_USABLE, which
        # planner.fits counts as anchored (<=).
        idle = {"gpu_name": "NVIDIA H200", "vram_total_gb": 139.8, "vram_free_gb": 139.07}
        core = one(request(tiers=[tier(host_ram_gb=8.0, idle=idle,
                                       cards=[{"gpu_name": "NVIDIA H200", "vram_total_gb": 139.8}])]))
        self.assertFalse(core["fits"])
        self.assertEqual(estimator._usable_vram(core["hardware_used"]), planner.ANCHORED_MAX_USABLE)
        self.assertIs(core["anchored"], True)

    def test_fit_residency_from_the_plan(self):
        core = one(request())
        self.assertTrue(core["fits"])
        self.assertEqual(core["residency"], core["rationale"]["residency"])


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
        import shutil
        import tempfile
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
        body = request(job(target_short_edge_px=None, output_size={"width": 2001, "height": 1003}))
        del body["job"]["target_short_edge_px"]
        core = one(body)
        covering = estimator.short_edge_covering(1920, 1080, 2001, 1003)
        self.assertEqual(core["planned_short_edge_px"], covering)
        self.assertEqual(core["rationale"]["output_width"],
                         estimator.output_dimensions(1920, 1080, covering)[0])
        self.assertEqual((core["output_width"], core["output_height"]), (2001, 1003))


class Still(unittest.TestCase):
    """An image source plans at frames 1 and delivers output_dimensions."""

    def test_still(self):
        core = one(request(job(frames=1, is_still=True, source_width=749, source_height=500,
                               target_short_edge_px=1920)))
        self.assertTrue(core["fits"])
        self.assertEqual(core["rationale"]["window"], 1)
        self.assertEqual((core["output_width"], core["output_height"]),
                         estimator.output_dimensions(749, 500, 1920))

    def test_still_with_frames_refused(self):
        refused_field(self, request(job(frames=90, is_still=True)), "job.frames")


class WorstCard(unittest.TestCase):
    """No idle worker: the card with the least vram_total_gb in cards[] is planned."""

    def test_least_memory(self):
        cards = [{"gpu_name": "NVIDIA H200", "vram_total_gb": 141.0},
                 {"gpu_name": "NVIDIA A40", "vram_total_gb": 44.7},
                 {"gpu_name": "NVIDIA B200", "vram_total_gb": 179.0}]
        core = one(request(tiers=[tier(cards=cards)]))
        self.assertEqual(core["card_source"], "worst_card")
        self.assertEqual(core["hardware_used"]["gpu_name"], "NVIDIA A40")


class VramOrder(unittest.TestCase):
    """idle, then last_run of the SAME card, then table; each stops the next being consulted."""

    IDLE = {"gpu_name": "NVIDIA A40", "vram_total_gb": 44.43, "vram_free_gb": 44.08}
    LAST = {"gpu_name": "NVIDIA A40", "vram_total_gb": 47.40, "vram_free_gb": 47.05}

    def test_idle_first(self):
        core = one(request(tiers=[tier(idle=dict(self.IDLE), last_run=dict(self.LAST))]))
        self.assertEqual(core["vram_source"], "idle")
        self.assertEqual(core["hardware_used"]["vram_free_gb"], 44.08)

    def test_idle_with_one_figure_falls_through_to_last_run(self):
        idle = {"gpu_name": "NVIDIA A40", "vram_total_gb": 44.43}
        core = one(request(tiers=[tier(idle=idle, last_run=dict(self.LAST))]))
        self.assertEqual(core["card_source"], "idle")
        self.assertEqual(core["vram_source"], "last_run")
        self.assertEqual(core["hardware_used"]["vram_free_gb"], 47.05)

    def test_basis_not_rewritten_off_nearest(self):
        for t in (tier(), tier(idle=dict(self.IDLE)), tier(last_run=dict(self.LAST))):
            core = one(request(tiers=[t]))
            self.assertNotEqual(core["vram_source"], "nearest_memory")
            self.assertEqual(core["prediction_basis"], core["rationale"]["prediction_basis"])
            self.assertEqual(core["prediction_basis"], "measured")

    def test_last_run_before_table(self):
        core = one(request(tiers=[tier(last_run=dict(self.LAST))]))
        self.assertEqual(core["vram_source"], "last_run")
        self.assertEqual(core["hardware_used"]["vram_total_gb"], 47.40)

    def test_last_run_of_another_card_not_used(self):
        other = {"gpu_name": "NVIDIA H200", "vram_total_gb": 139.8, "vram_free_gb": 139.06}
        core = one(request(tiers=[tier(last_run=other)]))
        self.assertEqual(core["vram_source"], "table")
        self.assertEqual(core["hardware_used"]["vram_total_gb"],
                         TABLE["NVIDIA A40"]["vram_total_gb"])
        self.assertEqual(core["hardware_used"]["vram_free_gb"],
                         TABLE["NVIDIA A40"]["vram_free_gb"])

    def test_last_run_with_one_figure_falls_through_to_table(self):
        last = {"gpu_name": "NVIDIA A40", "vram_free_gb": 47.05}
        core = one(request(tiers=[tier(last_run=last)]))
        self.assertEqual(core["vram_source"], "table")

    def test_host_ram_idle_before_tier(self):
        idle = dict(self.IDLE, host_ram_gb=51.22)
        core = one(request(tiers=[tier(idle=idle)]))
        self.assertEqual(core["hardware_used"]["host_ram_gb"], 51.22)


class TableGenerated(unittest.TestCase):
    """vram_table.json is the generator's output over the corpus, never hand-edited (§4a)."""

    def test_committed_table_matches_corpus(self):
        out = subprocess.run([sys.executable, os.path.join(SERVICE, "generate_vram_table.py"),
                              "--check", RUNS], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)


class Nearest(unittest.TestCase):
    """An unmeasured gpu_name resolves to the nearest measured card by memory."""

    MIG = "NVIDIA RTX PRO 6000 Blackwell MIG 2g.48gb"

    def test_resolves_to_nearest_and_is_not_measured(self):
        self.assertNotIn(self.MIG, TABLE)
        core = one(request(tiers=[tier(cards=[{"gpu_name": self.MIG, "vram_total_gb": 44.5}])]))
        self.assertEqual(core["vram_source"], "nearest_memory")
        self.assertEqual(core["resolved_from"], {"card": self.MIG, "measured": "NVIDIA A40"})
        self.assertEqual(core["hardware_used"]["gpu_name"], "NVIDIA A40")
        self.assertEqual(core["hardware_used"]["vram_free_gb"], TABLE["NVIDIA A40"]["vram_free_gb"])
        self.assertEqual(core["rationale"]["prediction_basis"], "measured")
        self.assertEqual(core["prediction_basis"], "borrowed")

    def test_tie_takes_lower_memory(self):
        low, high = TABLE["NVIDIA A40"]["vram_total_gb"], TABLE[
            "NVIDIA RTX PRO 6000 Blackwell Server Edition"]["vram_total_gb"]
        midpoint = (low + high) / 2.0
        table = {"NVIDIA A40": TABLE["NVIDIA A40"],
                 "NVIDIA RTX PRO 6000 Blackwell Server Edition":
                     TABLE["NVIDIA RTX PRO 6000 Blackwell Server Edition"]}
        chosen = ps.nearest_measured(midpoint, table)
        self.assertEqual(chosen, "NVIDIA A40")

    def test_idle_card_with_own_total_resolves(self):
        idle = {"gpu_name": self.MIG, "vram_total_gb": 94.0}
        core = one(request(tiers=[tier(idle=idle)]))
        self.assertEqual(core["vram_source"], "nearest_memory")
        self.assertEqual(core["resolved_from"]["measured"],
                         "NVIDIA RTX PRO 6000 Blackwell Server Edition")

    def test_duplicate_name_nominal_is_the_worst_entry(self):
        cards = [{"gpu_name": self.MIG, "vram_total_gb": 94.0},
                 {"gpu_name": self.MIG, "vram_total_gb": 44.5}]
        core = one(request(tiers=[tier(cards=cards)]))
        self.assertEqual(core["resolved_from"]["measured"], "NVIDIA A40")

    def test_idle_card_without_nominal_refused(self):
        idle = {"gpu_name": self.MIG}
        refused_field(self, request(tiers=[tier(idle=idle)]), "tiers[0].idle.gpu_name")


class Refusal(unittest.TestCase):
    """Each unusable input refused by name."""

    def test_missing_host_ram(self):
        t = tier()
        del t["host_ram_gb"]
        refused_field(self, request(tiers=[t]), "tiers[0].host_ram_gb")

    def test_empty_cards(self):
        refused_field(self, request(tiers=[tier(cards=[])]), "tiers[0].cards")

    def test_both_sizing_forms(self):
        body = request(job(output_size={"width": 2000, "height": 1000}))
        refused_field(self, body, "job.output_size")

    def test_neither_sizing_form(self):
        body = request(job())
        del body["job"]["target_short_edge_px"]
        refused_field(self, body, "job.target_short_edge_px")


#: e499dd5 is the image tiers ran before service/ existed: same handler/ tree as HEAD, other commit.
SAME_TREE_COMMIT = "e499dd51929b3fe25a0b5fdce7aee840d2027789"
#: 26294cc is e499dd5's parent, and e499dd5 changed handler/estimator.py.
OTHER_TREE_COMMIT = "26294cc5cb1f90fdaabe84f59223c1215a6f3fc1"


class Match(unittest.TestCase):
    """handler_match on handler/'s tree, not the commit (§5)."""

    def test_same_tree_other_commit_matches(self):
        self.assertNotEqual(SAME_TREE_COMMIT, _git("rev-parse", "HEAD").strip())
        core = one(request(tiers=[tier(worker_commit=SAME_TREE_COMMIT)]), commit=SHA_A)
        self.assertEqual(core["handler_tree"], _git("rev-parse", "HEAD:handler").strip())
        self.assertEqual(core["worker_handler_tree"],
                         _git("rev-parse", SAME_TREE_COMMIT + ":handler").strip())
        self.assertIs(core["handler_match"], True)
        self.assertEqual(core["commit"], SHA_A)
        self.assertNotIn("commit_match", core)

    def test_other_tree_mismatches(self):
        core = one(request(tiers=[tier(worker_commit=OTHER_TREE_COMMIT)]))
        self.assertEqual(core["worker_handler_tree"],
                         _git("rev-parse", OTHER_TREE_COMMIT + ":handler").strip())
        self.assertNotEqual(core["worker_handler_tree"], core["handler_tree"])
        self.assertIs(core["handler_match"], False)

    def test_commit_not_in_table_is_null(self):
        core = one(request(tiers=[tier(worker_commit=SHA_B)]))
        self.assertIsNone(core["worker_handler_tree"])
        self.assertIsNone(core["handler_match"])
        self.assertIsNotNone(core["handler_tree"])

    def test_none_sent_is_null(self):
        core = one(request())
        self.assertIsNone(core["worker_handler_tree"])
        self.assertIsNone(core["handler_match"])

    def test_short_sha_refused(self):
        refused_field(self, request(tiers=[tier(worker_commit=SAME_TREE_COMMIT[:7])]),
                      "tiers[0].worker_commit")

    def test_service_commit_is_information(self):
        with self.assertRaises(ValueError):
            ps.service_commit({"CF_PLANNER_COMMIT": "abc1234"})
        self.assertEqual(ps.service_commit({"CF_PLANNER_COMMIT": SHA_A}), SHA_A)
        self.assertIsNone(ps.service_commit({}))
        core = one(request(tiers=[tier(worker_commit=SAME_TREE_COMMIT)]), commit=None)
        self.assertIsNone(core["commit"])
        self.assertIs(core["handler_match"], True)


class HistoryTable(unittest.TestCase):
    """Generated from git, current at HEAD; a hand edit or a missed handler change fails (§5)."""

    def test_table_is_generated(self):
        out = subprocess.run([sys.executable, os.path.join(SERVICE, "generate_handler_history.py"),
                              "--check"], capture_output=True, text=True, cwd=REPO_ROOT)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)

    def test_newest_is_current_at_head(self):
        table = ps.load_handler_history()
        newest = table["newest"]
        self.assertEqual(subprocess.run(["git", "-C", REPO_ROOT, "merge-base", "--is-ancestor",
                                         newest, "HEAD"]).returncode, 0)
        self.assertEqual(_git("diff", "--name-only", newest, "HEAD", "--", "handler/"), "")
        self.assertEqual(table["commits"][newest], _git("rev-parse", "HEAD:handler").strip())
        self.assertEqual(ps.service_handler_tree(table), table["commits"][newest])


class Projection(unittest.TestCase):
    """For the same inputs, every §3b field on the wire equals the core output."""

    WIRE_FIELDS = ("fits", "predicted_seconds", "reason", "residency", "output_width",
                   "output_height", "best_window", "ideal_window", "binding_phase", "anchored",
                   "prediction_basis", "hardware_used", "card_source", "vram_source",
                   "resolved_from", "registry_version", "commit", "handler_tree",
                   "worker_handler_tree", "handler_match")

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
            request(tiers=[tier(cards=[{"gpu_name": Nearest.MIG, "vram_total_gb": 44.5}],
                                worker_commit=SAME_TREE_COMMIT)]),
        ]
        for body in cases:
            cores = ps.estimate_core(copy.deepcopy(body), commit=SHA_A)
            wire = self._through_http(body)["tiers"]
            self.assertEqual(len(wire), len(cores))
            for core, entry in zip(cores, wire):
                self.assertEqual(sorted(entry), sorted(("tier",) + self.WIRE_FIELDS))
                self.assertNotIn("rationale", entry)
                self.assertEqual(entry["tier"], core["tier"])
                for field in self.WIRE_FIELDS:
                    self.assertEqual(entry[field], core[field], field)

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
        self.assertEqual(payload["handler_history_newest"], table["newest"])


def _git(*args):
    return subprocess.run(["git", "-C", REPO_ROOT] + list(args), capture_output=True,
                          text=True, check=True).stdout


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
