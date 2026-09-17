"""Generate `handler_history.json`: every commit on `main` → the git tree id of its `handler/`.

`cf-planner.md` §5 (cf-upscale-project 5b0c2aa). **Never hand-edit the output.** The service runs
no git — its image and its deploy build carry no `.git` — so a `worker_commit` is resolved from this
table, and the service's own `handler_tree` is the table's newest entry.

**A table cannot hold the commit that adds it**, so `newest` is `main` at generation time and HEAD
moves past it. The kit holds that gap honest: nothing after `newest` may touch `handler/`.

Usage (from a checkout of cf-upscale-image):
    python3 service/generate_handler_history.py           write service/handler_history.json
    python3 service/generate_handler_history.py --check   exit 1 unless the committed table is
                                                          exactly what `newest`'s history generates
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


def main(argv):
    if "--check" in argv:
        try:
            with open(HISTORY_PATH, encoding="utf-8") as handle:
                committed = handle.read()
            newest = json.loads(committed)["newest"]
        except (OSError, ValueError, KeyError) as error:
            print("handler_history.json unreadable: {}".format(error), file=sys.stderr)
            return 1
        if render(build(newest)) != committed:
            print("handler_history.json differs from what {}'s history generates".format(newest),
                  file=sys.stderr)
            return 1
        print("handler_history.json matches the history of {}".format(newest))
        return 0
    newest = _git("rev-parse", REF).strip()
    with open(HISTORY_PATH, "w", encoding="utf-8") as handle:
        handle.write(render(build(newest)))
    print("wrote {} ({} at {})".format(HISTORY_PATH, REF, newest))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
