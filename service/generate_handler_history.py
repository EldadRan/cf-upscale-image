"""Generate `handler_history.json`: every commit on `main` → the git tree id of its `handler/`.

`cf-planner.md` §5 (cf-upscale-project 5b0c2aa). **Never hand-edit the output.** The service runs
no git — its image and its deploy build carry no `.git` — so a `worker_commit` is resolved from this
table, and the service's own `handler_tree` is the table's newest entry.

**A table cannot hold the commit that adds it**, so `newest` is `main` at generation time and HEAD
moves past it. The kit holds that gap honest: nothing after `newest` may touch `handler/`.

Usage (from a FULL, non-shallow checkout of cf-upscale-image):
    python3 service/generate_handler_history.py                 write service/handler_history.json
    python3 service/generate_handler_history.py --check [PATH]  exit 1 unless the table is exactly
                                                                what `newest`'s history generates,
                                                                for `ref` main, on main

**A shallow clone stops both modes.** `rev-list` ends at the shallow boundary, so a table written
there is silently cut short and a correct one checked there looks wrong.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
HISTORY_PATH = os.path.join(HERE, "handler_history.json")
REF = "main"


def _git(*args, stdin=None):
    return subprocess.run(["git", "-C", REPO_ROOT] + list(args), input=stdin,
                          capture_output=True, text=True, check=True).stdout


class Stop(Exception):
    """The checkout cannot answer the question honestly."""


def _refuse_shallow():
    if _git("rev-parse", "--is-shallow-repository").strip() == "true":
        raise Stop("shallow clone: the history table needs every commit reachable from main")


def _is_ancestor(older, newer):
    return subprocess.run(["git", "-C", REPO_ROOT, "merge-base", "--is-ancestor", older, newer],
                          capture_output=True).returncode == 0


def build(newest):
    """The table for every commit reachable from `newest`, newest first."""
    commits = _git("rev-list", newest).split()
    lookups = "".join("{}:handler\n".format(sha) for sha in commits)
    answers = _git("cat-file", "--batch-check", stdin=lookups).splitlines()
    if len(answers) != len(commits):
        raise RuntimeError("cat-file answered {} of {} lookups".format(len(answers), len(commits)))
    table = {}
    for sha, answer in zip(commits, answers):
        parts = answer.split()
        # A commit with no handler/ maps to null: there is no tree to match, and none is invented.
        table[sha] = parts[0] if len(parts) == 3 and parts[1] == "tree" else None
    return {
        "generated_by": "service/generate_handler_history.py — never hand-edited",
        "ref": REF,
        "newest": newest,
        "count": len(table),
        "commits": table,
    }


def render(history):
    return json.dumps(history, indent=2) + "\n"


def check(path):
    """Why `path` is not the generated table, or None when it is."""
    try:
        with open(path, encoding="utf-8") as handle:
            committed = handle.read()
        history = json.loads(committed)
        newest = history["newest"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return "unreadable: {}".format(error)
    if history.get("ref") != REF:
        return "ref is {!r}, not {!r}".format(history.get("ref"), REF)
    if not _is_ancestor(newest, REF):
        return "newest {} is not on {}".format(newest, REF)
    if render(build(newest)) != committed:
        return "differs from what {}'s history generates".format(newest)
    return None


def write():
    newest = _git("rev-parse", REF).strip()
    # **A stale local main is refused** where the last fetch knows better: the table would lack
    # commits a tier can already be running.
    upstream = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "--verify", "--quiet",
                               "refs/remotes/origin/" + REF], capture_output=True, text=True)
    if upstream.returncode == 0 and not _is_ancestor(upstream.stdout.strip(), newest):
        raise Stop("local {} is behind origin/{}; update it first".format(REF, REF))
    with open(HISTORY_PATH, "w", encoding="utf-8") as handle:
        handle.write(render(build(newest)))
    return newest


def main(argv):
    try:
        _refuse_shallow()
        if argv and argv[0] == "--check":
            path = argv[1] if len(argv) > 1 else HISTORY_PATH
            problem = check(path)
            if problem:
                print("{}: {}".format(path, problem), file=sys.stderr)
                return 1
            print("{} matches the history of its newest entry".format(path))
            return 0
        newest = write()
    except Stop as stop:
        print("STOP: {}".format(stop), file=sys.stderr)
        return 2
    print("wrote {} ({} at {})".format(HISTORY_PATH, REF, newest))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
