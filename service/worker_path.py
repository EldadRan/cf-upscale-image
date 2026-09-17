"""Put the worker's `handler/` on the import path. **One place, and it searches for nothing.**

`CF_PLANNER_HANDLER` when set, otherwise `handler/` beside this directory — the same layout in a
checkout and in the service's container. Several near-identical `handler/` trees exist on the
machines this runs on, and a resolver that guessed would plan with a worker nobody ships.

The service imports `handler/` and never writes to it (`cf-planner.md` §2).
"""

import os
import sys

HANDLER = os.path.abspath(os.environ.get("CF_PLANNER_HANDLER") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "handler"))

if not os.path.isfile(os.path.join(HANDLER, "estimator.py")):
    raise ImportError("no worker handler at {} (estimator.py missing)".format(HANDLER))

if HANDLER not in sys.path:
    sys.path.insert(0, HANDLER)
