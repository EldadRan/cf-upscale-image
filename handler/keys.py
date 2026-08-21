"""The names this worker writes under CF's prefix.

**This is contract, not implementation.** CF hands over a prefix and records the prefix, not the
keys — so when the job record expires (async results are retained 30 minutes; the job is deleted
at its `ttl`), recovery starts with a `ListObjectsV2` and finds *some set of files whose names CF
did not choose*. A listing CF has to guess at is not a recovery path. So the names are:

  **deterministic** — no timestamp, no uuid, no attempt counter, nothing that varies run to run;
  **derivable** — CF can predict every one of them from the prefix alone, without asking;
  **stable on a re-run** — CF re-sends the same `request_id` and the same prefix, and a second
  run must overwrite the first rather than produce a parallel set.

There is no submission idempotency key and deliberately will not be one: recovery is by lookup,
and a lookup only works against names it can predict.

**The prefix already carries the request identity** (`req/01JQ8Z…/`), so these names do not
repeat it. Putting `request_id` in the filename as well would make the key longer without making
it more predictable, and would invite the reading that the name is unique rather than the prefix.

One role writes one key. That is what makes a repeated role in `derive[]` a refusal rather than a
silent overwrite returning a manifest claiming two files where one exists.
"""

#: The upscaled video. The expensive artefact, and the one written before any derive is
#: attempted — a failed derive is recoverable from the master, while a master that was never
#: written costs hours of GPU to reproduce.
MASTER = "master.mp4"

#: A still's master. PNG, for the reason in `encoder.still_master_extension`: it is the artefact a
#: person opens, and it opens everywhere. **`master.webp` stays enumerated** even though nothing
#: writes it now — a listing-based recovery has to recognise objects written by earlier builds,
#: and jobs run before this change left `master.webp` in R2.
STILL_MASTERS = ("master.png", "master.webp")


#: The default stem, and the one every name above is built from. Named so the fallback path and
#: the caller path are visibly the same construction with a different stem.
DEFAULT_STEM = "master"

#: How long a caller-supplied stem may be. An S3 key is capped at 1 024 bytes *including the
#: prefix*, and CF's prefixes (`req/01JQ8Z.../`) already spend some of that — so this is generous
#: for any identifier a front-end would hold, and nowhere near the ceiling.
MAX_STEM = 100

#: Stripped from a caller's stem, per the contract amendment of 2026-08-19. **Path separators,
#: because the prefix is the only place the caller chooses *where*** — a stem carrying `/` would
#: write a child of the prefix that CF's listing-based recovery does not expect, and one carrying
#: `..` would try to leave it. Control characters, because a name is read by people out of logs
#: and listings, and a key nobody can print is a key nobody can ask about.
_STRIPPED = "/\\"

#: Extensions this worker itself writes. A caller that sends `job-42.mp4` meant `job-42` and put
#: the format in out of habit; appending ours on top would deliver `job-42.mp4.mp4`, and on a
#: still, `job-42.mp4.png` — honest, since the last extension is the true one, but confusing
#: enough that a person would read it as a mistake. Exactly one trailing occurrence is dropped,
#: and only from this closed set, so an ordinary stem like `clip.v2` is left alone.
_OUR_EXTENSIONS = (".mp4", ".png", ".webp")


def sanitize_stem(name):
    """A caller's stem, made safe to use as the last segment of a key. None if nothing survives.

    **The rule is subtraction, not translation.** Nothing here maps a rejected character onto a
    replacement: a stem that comes back shorter is still recognisably the caller's, while one
    whose characters were swapped for underscores is a name the caller never chose and cannot
    predict — and predictability is the entire reason `keys.py` exists.

    Refusing outright was the alternative and is worse: a request already carrying real GPU work
    should not be turned away over a filename, and the field is optional precisely so that a
    caller who gets it wrong still gets a master. A stem that sanitizes to nothing falls back to
    the default, which is the same behaviour as not sending the field at all.
    """
    if not isinstance(name, str):
        return None
    cleaned = "".join(c for c in name if c not in _STRIPPED and (c >= " " and c != "\x7f"))
    # Leading dots would produce a hidden file, and a bare `.` or `..` a path element; trailing
    # dots and whitespace are silently eaten by some filesystems, which makes a name unpredictable
    # in exactly the way this module refuses to be.
    cleaned = cleaned.strip().strip(".").strip()
    for extension in _OUR_EXTENSIONS:
        if cleaned.lower().endswith(extension):
            cleaned = cleaned[:-len(extension)].strip().strip(".").strip()
            break
    cleaned = cleaned[:MAX_STEM].strip().strip(".").strip()
    return cleaned or None


def master_name(is_still, width=None, height=None, name=None):
    """The master's key. `master.mp4` for anything with a time axis; a lossless still otherwise.

    **A one-frame job is not merely a short video, at the point of delivery.** An MP4 master of a
    still is lossy, cannot carry alpha at all (`yuv420p` has no fourth channel), and makes every
    image derive taken from it lossy a second time. So the medium decides the container, and this
    is the only place that decision is spelled.

    **`name` is the caller's stem** (F-2026-08-19-38): CF's front-end holds an internal identifier
    per job and wants the artefact called by it. The stem is theirs; **the extension stays ours**,
    because which one is right is a fact only this worker knows at delivery time — the stills rule
    above decides it — and a caller-chosen extension could state a falsehood about the bytes.

    An absent or unusable `name` falls back to the names above byte-for-byte, so every request
    written before this existed is untouched and nothing certified moves.

    CF records the prefix rather than the keys and reads the manifest for names, so this stays
    within the worker's remit — but it is a visible change to what a job writes, and
    `docs/questions-for-cf.md` says so rather than letting CF find out from an object listing.
    """
    stem = sanitize_stem(name) or DEFAULT_STEM
    if not is_still:
        return stem + ".mp4"
    from encoder import still_master_extension  # local: keeps `keys` importable without PIL
    return stem + still_master_extension(width, height)

#: What the response envelope would have carried, in durable storage. Written on every
#: successful job, because the response does not outlive the job record and this does.
MANIFEST = "manifest.json"

#: Derive roles → their single deterministic key. `crop` is the exception: it writes `count`
#: files, so its name is a function of the ordinal rather than a constant. Still deterministic —
#: `crop_0.webp` is always the highest-energy region of a given request.
DERIVE = {
    "poster": "poster.webp",
    "proxy": "proxy.mp4",
}

CROP_PREFIX = "crop_"
CROP_SUFFIX = ".webp"


def crop_name(ordinal):
    return "{}{}{}".format(CROP_PREFIX, ordinal, CROP_SUFFIX)


#: **Keyed on the extension, not on the literal names** (F-2026-08-19-38). The master's stem is
#: the caller's now, so a table of whole filenames would have had exactly one entry that could
#: never be looked up — and the fallback for a missing content type is `application/octet-stream`
#: at the browser, which turns a view link into a download of something unrecognised. The
#: extension is the only part of a key that says what the bytes are, which is why it stayed ours.
CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".webp": "image/webp",
    ".json": "application/json",
}


def for_role(role):
    """The key a derive role writes. Raises KeyError on an unknown role, which validation
    refuses long before this is reached."""
    return DERIVE[role]


def content_type(name):
    """The MIME type for a key's last segment. **Raises KeyError on an extension this worker does
    not write**, which is the strictness the whole-filename table used to provide: a name that
    reaches storage with no content type is a file the browser will not open, and finding that out
    at upload time beats finding it out from a customer."""
    _, dot, extension = name.rpartition(".")
    if not dot:
        raise KeyError(name)
    return CONTENT_TYPES["." + extension.lower()]


#: Every name this worker writes **when the request does not name the master**, so a test can
#: assert the set rather than a sample, and so a recovery tool has one place to read them from
#: rather than a second list that goes stale. The crop names are enumerated to the maximum because
#: a listing-based recovery has to recognise them without knowing what the request asked for.
#:
#: **A request carrying `output.name` puts one name here beyond prediction, by design** — and only
#: one. The caller chose that name, so the caller can predict it; everything else in the prefix
#: stays derivable, and `manifest.json` (itself always named) carries the master's key for anyone
#: who arrives without the request. That is why the field names the master and nothing else.
ALL = (MASTER,) + STILL_MASTERS + (MANIFEST,) + tuple(sorted(DERIVE.values())) \
    + tuple(crop_name(i) for i in range(8))
