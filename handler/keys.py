"""The names this worker writes under CF's prefix.

**This is contract, not implementation.** CF hands over a prefix and records the prefix, not the
keys — so when the job record expires (async results are retained 30 minutes; the job is deleted
at its `ttl`), recovery starts with a `ListObjectsV2` and finds *some set of files whose names CF
did not choose*. A listing CF has to guess at is not a recovery path. So the names are:

  **deterministic** — no timestamp, no uuid, no attempt counter, nothing that varies run to run;
  **derivable** — CF can predict every one of them from `request_id` alone, without asking;
  **stable on a re-run** — CF re-sends the same `request_id` and the same prefix, and a second
  run must overwrite the first rather than produce a parallel set.

There is no submission idempotency key and deliberately will not be one: recovery is by lookup,
and a lookup only works against names it can predict.

## `<request_id>_<role>.<ext>` — AND THIS REVERSES WHAT THIS FILE USED TO SAY

**The old rule was that the id belongs in the prefix and nowhere else**, and it was right for the
old layout: *"The prefix already carries the request identity (`req/01JQ8Z…/`), so these names do
not repeat it. Putting `request_id` in the filename as well would make the key longer without
making it more predictable."* Two things overturn it and only the second is new.

**A file that leaves its directory is anonymous.** A poster in a CMS, a master downloaded to a
desktop, a proxy attached to a message — every one of them was `master.mp4`, with nothing on it to
say which job made it. The prefix identifies the file only while the file is still in it.

**And the prefix stops carrying the identity.** `req/<id>/` had the id in it; a date-partitioned
prefix is about when rather than about which. *(storage.md §3, CF 2026-08-28. Written here
because the old rule was correct for the old layout and somebody will otherwise re-derive it in
six months.)*

**FILED, NOT SETTLED, and this file does not rest on it.** `storage.md` §3 argues the change from
"two jobs on the same day would both write `manifest.json`" — which holds only if the prefix is
`YYYY/MM/DD/`. Its own §1 gives the layout as `YYYY/MM/DD/<request_id>/`, under which two jobs on
one day are in different directories and could not collide. Both cannot be true. The first
argument above — a file that leaves its directory is anonymous — holds under either reading and
is the one this module is built on; the collision argument is the gate's to resolve.

**THE EXTENSION STAYS OURS**, and that is not a leftover from the old scheme. Which extension is
right is a fact only this worker knows at delivery time — a one-frame job is not a short video at
the point of delivery, an MP4 master of a still is lossy, cannot carry alpha, and makes every
image derive taken from it lossy a second time. A caller-chosen extension could state a falsehood
about the bytes. The role segment is ours for the same reason: `master`, `manifest`, `poster`,
`proxy`, `crop_<n>` are this worker's vocabulary.

**`output.name` is RETIRED** (storage.md §4, CF 2026-08-28). It let a caller choose the master's
stem, for a need `request_id` now serves better and serves for every artefact rather than one.

One role writes one key. That property is unchanged by this rewrite and it never rested on the
key collision anyway — `validation.py` refuses a repeated `derive` role by name against its own
`seen` set, long before a key is built.
"""

#: The role vocabulary, in the order a job produces them. **Data rather than a set of module
#: constants**, because every name is now a function of `request_id` and a constant could only
#: ever be half a key — which is exactly the shape that would tempt a caller to concatenate.
#:
#: `master` and `crop` are absent: the master's extension is decided at delivery by the medium,
#: and `crop` writes `count` files rather than one. Both have their own function below.
ROLE_EXTENSIONS = {
    "manifest": ".json",
    "poster": ".webp",
    "proxy": ".mp4",
}

#: Roles a caller may request in `derive[]`. `manifest` is ours and is not one of them.
DERIVE_ROLES = ("poster", "proxy", "crop")

CROP_ROLE = "crop"
CROP_EXTENSION = ".webp"

#: The separator between the id and the role. One character, and it is not `-` or `.` on purpose:
#: `request_id` may legally contain both (§3a accepts `& $ @ = ; : + , ?` and space, and UUIDv7
#: ids are hyphenated), so a separator drawn from the id's own alphabet would make the boundary
#: ambiguous to anything parsing a key back apart. `_` is refused nowhere and appears in no id
#: shape CF mints.
SEPARATOR = "_"


def _key(request_id, role, extension):
    """`<request_id>_<role><extension>`. The one place the shape is spelled."""
    return "{}{}{}{}".format(request_id, SEPARATOR, role, extension)


def master_name(request_id, is_still, width=None, height=None):
    """The master's key. `.mp4` for anything with a time axis; a lossless still otherwise.

    **A one-frame job is not merely a short video, at the point of delivery.** An MP4 master of a
    still is lossy, cannot carry alpha at all (`yuv420p` has no fourth channel), and makes every
    image derive taken from it lossy a second time. So the medium decides the container, and this
    is the only place that decision is spelled.
    """
    if not is_still:
        return _key(request_id, "master", ".mp4")
    from encoder import still_master_extension  # local: keeps `keys` importable without PIL
    return _key(request_id, "master", still_master_extension(width, height))


def manifest_name(request_id):
    """What the response envelope would have carried, in durable storage. Written on every
    successful job, because the response does not outlive the job record and this does."""
    return _key(request_id, "manifest", ROLE_EXTENSIONS["manifest"])


def for_role(request_id, role):
    """The key a derive role writes. **Raises KeyError on an unknown role**, which validation
    refuses long before this is reached — kept strict so that a role added to one list and not
    the other fails here rather than writing a key nobody can predict."""
    return _key(request_id, role, ROLE_EXTENSIONS[role])


def crop_name(request_id, ordinal):
    """`crop` writes `count` files, so its name is a function of the ordinal rather than a
    constant. Still deterministic — `<id>_crop_0.webp` is always the highest-energy region."""
    return _key(request_id, "{}_{}".format(CROP_ROLE, ordinal), CROP_EXTENSION)


#: **Keyed on the extension, not on whole filenames.** The extension is the only part of a key
#: that says what the bytes are, which is why it stayed ours; and under `<request_id>_<role>` a
#: table of whole filenames could not be written down at all.
CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".webp": "image/webp",
    ".json": "application/json",
}


def content_type(name):
    """The MIME type for a key's last segment. **Raises KeyError on an extension this worker does
    not write**: a name that reaches storage with no content type is a file the browser will not
    open, and finding that out at upload time beats finding it out from a customer."""
    _, dot, extension = name.rpartition(".")
    if not dot:
        raise KeyError(name)
    return CONTENT_TYPES["." + extension.lower()]


#: The maximum number of crops a request may ask for, and therefore the number a listing-based
#: recovery must be able to recognise without knowing what was asked.
MAX_CROPS = 8


def all_names(request_id, still_extensions=(".png", ".webp")):
    """Every name this worker can write for one request, so a test can assert the set rather
    than a sample and a recovery tool has one place to read them from.

    **A function now, not a constant**, which is the whole shape of this change: there is no
    longer a name that can be known without knowing whose job it is.

    Both still extensions are enumerated because which one a still master takes is decided by
    `encoder.still_master_extension` at delivery, from the pixels — so a recovery arriving with
    only the id cannot narrow it and must be able to recognise either.
    """
    names = [master_name(request_id, is_still=False)]
    names += [_key(request_id, "master", ext) for ext in still_extensions]
    names.append(manifest_name(request_id))
    names += [for_role(request_id, role) for role in sorted(ROLE_EXTENSIONS) if role != "manifest"]
    names += [crop_name(request_id, i) for i in range(MAX_CROPS)]
    return tuple(names)
