"""The 2^31 index floor in colour correction, removed without changing a pixel.

**What actually fails.** Every colour-correction method except `adain` and `hsv` routes through
`wavelet_reconstruction`, and that walks a five-level pyramid whose blur is a dilated grouped
convolution:

    radius = 2 ** i                     # i in 0..4, so r_max = 16
    image  = safe_pad_operation(image, (radius,) * 4, mode='replicate')
    output = F.conv2d(image, kernel, groups=num_channels, dilation=radius)

`F.conv2d` dispatches to a kernel that indexes with 32-bit arithmetic when
`canUse32BitIndexMath` says it may, and the tensor it is asked about is the *padded* one. So the
limit is not on the frame, and not on the clip, but on

    w * C * (H + 2r) * (W + 2r) < 2**31

which at 4K puts the last safe window at 81 and at 8K at 21 -- registry
`floors_and_budgets.index_floor`. Measured either side of it: run 4 at w81 (0.96x the bound)
delivered a master, run 3 at w85 (1.008x) crashed 3.2 s into phase 4 with all three earlier
phases already paid for.

**Why chunking is free here.** The pyramid is spatial and grouped -- `wavelet_blur`,
`wavelet_decomposition` and the `add_`/`clamp_` reconstruction all act frame by frame, and no
step mixes one frame's values into another's. Splitting dim 0 and concatenating therefore
returns bit-identical values, not merely similar ones.

**And why only this function.** `lab_color_transfer`'s other half is
`_histogram_matching_channel`, which flattens `[B, H, W]` and sorts it whole: its CDF is a
property of the batch, so chunking *that* would change the output. It is left alone, and it is
never the thing that trips the floor -- one channel of one batch is a third of the padded conv's
element count and stays under the bound wherever the conv does not.

**Nothing splits until it must.** The threshold is the true bound rather than a safety fraction
of it, so a window that fits today takes exactly the path it took when it was measured. That is
what lets the release's re-key runs reproduce their recorded peaks instead of merely passing a
bound.
"""

#: `canUse32BitIndexMath` admits a tensor whose element count fits a signed 32-bit integer, so
#: the last admissible size is `INT_MAX`, not `2**31`.
INDEX_LIMIT = 2 ** 31 - 1

#: The deepest pyramid level's dilation, and so the padding the widest blur adds to each edge.
#: Read off `wavelet_decomposition`'s `radius = 2 ** i` over `levels=5`, not assumed.
MAX_RADIUS = 16


def _radius_for(height, width):
    """The dilation the widest blur will actually use on this plane.

    `wavelet_blur` clamps the radius to an eighth of the shorter side, so a small plane pads by
    less than `MAX_RADIUS` and may hold more frames per pass than the constant alone implies.
    """
    return min(MAX_RADIUS, max(1, min(height, width) // 8))


def frames_per_pass(shape):
    """How many frames of `[B, C, H, W]` the padded convolution can index at once.

    Returns `None` when the whole batch already fits, which is the common case and the one where
    this module must leave no trace.
    """
    frames, channels, height, width = shape
    radius = _radius_for(height, width)
    per_frame = channels * (height + 2 * radius) * (width + 2 * radius)
    if frames * per_frame <= INDEX_LIMIT:
        return None
    # At least one frame: a single 8K frame is 0.1 x the bound, so this floor is a guard against
    # an absurd plane rather than a case that arises.
    return max(1, INDEX_LIMIT // per_frame)


def chunked(reconstruction, torch, log_target=None, debug=None):
    """`wavelet_reconstruction`, split along dim 0 only when the index floor demands it.

    **The banner reads `log_target`, not the forwarded `debug`** — F-2026-08-18-8, and the
    distinction is the whole fix. `lab_color_transfer` calls this function as
    `wavelet_reconstruction(content_feat, style_feat, debug=None)` (vendored
    `src/utils/color_fix.py:280`): an *explicit* keyword, which overrides any default this
    wrapper carries. So on `lab` — the default colour mode, and the one R3 ran — the guard was
    false by construction and the banner could never print, while the split below it ran
    correctly the whole time. R3 delivered its master; only its evidence line was missing.

    `debug=None` as the wrapper's default now mirrors the vendored signature exactly, so what
    reaches the real reconstruction is whatever the caller passed and nothing else. The log
    target is resolved separately, at install time, from the CLI the worker owns.
    """

    def wrapper(content_feat, style_feat, debug=None, **kwargs):
        step = frames_per_pass(content_feat.shape)
        if step is None:
            return reconstruction(content_feat, style_feat, debug=debug, **kwargs)

        frames = content_feat.shape[0]
        # Truthiness, not `is not None`: `cli.debug` is a `Debug` singleton in the image and a
        # plain `False` in the offline double, and `False.log(...)` would turn a split into a
        # crash on exactly the runs this exists to rescue.
        if log_target:
            log_target.log(
                "colour correction split into {} passes of <={} frames: {} frames of "
                "{}x{} would index {:.2f}x the 32-bit convolution limit".format(
                    -(-frames // step), step, frames,
                    content_feat.shape[-1], content_feat.shape[-2],
                    frames * content_feat.shape[1]
                    * (content_feat.shape[-2] + 2 * _radius_for(*content_feat.shape[-2:]))
                    * (content_feat.shape[-1] + 2 * _radius_for(*content_feat.shape[-2:]))
                    / float(INDEX_LIMIT)),
                category="video", force=True, indent_level=1)

        parts = []
        for start in range(0, frames, step):
            stop = min(start + step, frames)
            # **Style is sliced with content.** The two are frame-aligned by construction --
            # `postprocess_all_batches` trims `input_video` to `sample`'s length before calling
            # -- and a whole-batch style against a sliced content would transfer the wrong
            # frames' colour.
            parts.append(reconstruction(
                content_feat[start:stop], style_feat[start:stop], debug=debug, **kwargs))
        return torch.cat(parts, dim=0)

    return wrapper


def install(cli, color_fix, generation_phases, debug=None):
    """Swap the chunked reconstruction into every namespace that resolves the name.

    Two, because Python binds the global at call time in whichever module the caller lives in:
    `lab_color_transfer` and `wavelet_color_fix` reach it through `color_fix`'s own globals,
    while the `wavelet` branch of `postprocess_all_batches` reaches the copy `generation_phases`
    imported. Patching one and not the other fixes half the methods.

    Returns the undo pairs, so a caller that installs for the duration of a stream can put the
    vendored source back exactly as it found it.

    **Loud if the name is gone.** A vendored bump that renamed or inlined `wavelet_reconstruction`
    would otherwise leave this installing nothing, and the first evidence would be a phase-4
    crash on an 8K job after three phases of GPU had been paid for. The build asserts the same
    thing on the image, so in practice this is the second line of defence rather than the first.
    """
    original = getattr(color_fix, "wavelet_reconstruction", None)
    if original is None:
        raise AttributeError(
            "the vendored colour-correction module has no `wavelet_reconstruction`; the 32-bit "
            "index floor cannot be held and any window above 84 frames at 4K (21 at 8K) would "
            "die in phase 4 with the expensive phases already spent")
    # **The log target is the worker's own debug, resolved here rather than forwarded.** The
    # vendored callers decide what `debug` the reconstruction gets — one of them passes `None` —
    # and that is their business; whether *this* worker announces a split is this worker's.
    replacement = chunked(original, cli.torch,
                          log_target=debug if debug is not None else getattr(cli, "debug", None))
    restore = []
    for module in (color_fix, generation_phases):
        if getattr(module, "wavelet_reconstruction", None) is original:
            module.wavelet_reconstruction = replacement
            restore.append((module, original))
    return restore


def uninstall(restore):
    for module, original in restore or ():
        module.wavelet_reconstruction = original
