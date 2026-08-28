# `.githooks/` — the hooks git actually runs

**These are not copies. `core.hooksPath` points git here, so the tracked file IS the hook.**

```
git config core.hooksPath .githooks
```

One command, once per clone. Until it is run, **this repository has no hooks at all** — a fresh
clone can push a stale graph freely.

## Why a tracked directory rather than a copy into `.git/hooks/`

Both hooks lived in `.git/hooks/` until 2026-08-28, untracked by deliberate design. That made
`roles.md`'s statement — *"a pre-push hook in BOTH repositories refuses a push whose graph is
stale"* — **a false sentence about the project**: the hooks were in two directories on one Mac, and
every other clone could push stale graphs while the shared law told its reader they could not.

The obvious repair is a tracked copy plus an install step. **It was rejected because it creates two
artefacts and makes the tracked one a claim about the other.** A committed hook nobody re-installs
reads as the guard while the guard is whatever was copied months ago, and a witness comparing the
two can only run where a hook is already installed — blind in exactly the case it exists for.

`core.hooksPath` has one file. There is nothing to keep in step, so nothing can drift.

## The cost, written here because here is where it is paid

**`core.hooksPath` replaces the hooks directory wholesale.** Anything dropped into `.git/hooks/`
after this is silently ignored — no error, no warning, and the hook simply never fires.

Today that costs nothing: `pre-push` was the only hook in either repository when this was set up.
**The person who pays is whoever adds a `pre-commit` in a year, drops it in the obvious place, and
watches it do nothing.** If that is you: put it here instead, and it will work.

## What the hook does and does not do

It refuses a push of this image whose graph IN `cf-upscale-project` is stale — `handler/` is
graphed there, not here. It reads the sibling repository and blocks; it never writes to it. **It asserts and
never repairs** — a hook that regenerated and committed would make a repository write on its own
behalf during a push.

**It runs only when the pushed RANGE touches `handler/`**, minus `handler/vendor/` — the
generator excludes the vendored tree and says so in every diagram it writes, so a vendor bump
cannot make a diagram lie. A documents-only push cannot make a diagram
lie, and refusing it was blocking the other role over files neither push touched.

**When it does run, the check reads the WORKING TREE** on both sides — this `handler/` and the
sibling's `architecture/`. An uncommitted edit in either still refuses. That limit is named in
the hook itself and is not fixed by the narrowing.

**One arm is untested and says so in the hook.** No vendor-only commit exists in this history,
so the `handler/vendor/` exclusion has never been exercised by a real range. A commit invented
to turn it green would test the invention, so the note stays instead.

**Nothing here enters the image.** `docker-publish.yml` builds with `context: ./handler`, so a
top-level directory is outside the build context by construction.

Bypass for a genuine emergency is git's own: `git push --no-verify`.
