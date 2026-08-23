"""Practical-RIFE, adapted to the one call the shim makes.

`interpolate.Interpolator` asks its model for `model(a, b, timestep)` and for `to()`/`eval()`.
Practical-RIFE's `train_log.RIFE_HDv3.Model` offers `inference(img0, img1, timestep, scale)`,
`eval()` and a no-argument `device()`. This is the adapter between the two, and nothing else: it
loads no files the bake did not put there, decides no policy, and holds no per-call state.

**Imported lazily and never at module scope.** Constructing the model pulls torch, torchvision
and the vendored `model` package, which is a GPU-box import — the same reason `pipeline.load_cli`
is lazy. Route C reaches it once per job.

**The weights and the model definition come from one pinned archive**, which is unusual and worth
knowing: `RIFE_HDv3.py` ships inside the same zip as `flownet.pkl`, so the class constructed here
is pinned by the hash of the weights it loads. `bake_weights.py` carries both pins and asserts
both. The `model` package beside `train_log` is the second pin — it is not in that archive, and
`RIFE_HDv3` imports it at module scope.
"""
import os
import sys

#: Where `bake_weights.py` put `train_log/` and `model/`. Set by the Dockerfile; a local run
#: points it at a checkout.
MODEL_DIR_ENV = "RIFE_MODEL_DIR"

#: **`scale` is RIFE's flow-pyramid resolution, not our upscale factor.** `inference` builds its
#: scale list as `[16/scale, 8/scale, ...]`, so a smaller number means a coarser pyramid and less
#: memory. 1.0 is the reference script's default and is what §6's fit predicate will be measured
#: against; anything else is a different line in the registry, not a free knob.
DEFAULT_SCALE = 1.0


def model_dir():
    """The baked directory, or a clear refusal naming the variable that was not set."""
    path = os.environ.get(MODEL_DIR_ENV)
    if not path:
        raise RuntimeError(
            "{} is not set. Every image built since RIFE was baked sets it; a local run has to "
            "point it at a directory holding train_log/ and model/.".format(MODEL_DIR_ENV))
    return path


class Rife:
    """One loaded interpolator. Constructed once per job, called once per synthesis.

    **The cast is explicit and nothing process-global is touched** (contract §9b). The reference
    script's `--fp16` path calls `set_default_tensor_type(torch.cuda.HalfTensor)`, which would
    make every tensor created afterwards half-on-CUDA — the encoder's, the host guard's, and on a
    full image SeedVR2's. This casts the module and lets `Interpolator._cast` cast the inputs.
    """

    def __init__(self, model, device, dtype, scale=DEFAULT_SCALE):
        self._model = model
        self._device = device
        self._dtype = dtype
        #: **Carried here so the model call and the shim's padding cannot disagree.** The shim
        #: pads to a multiple of `max(128, 128/scale)`; `inference` builds its pyramid as
        #: `[16/scale, 8/scale, …]`. One value set in two places is two places to be wrong.
        self._scale = scale

    @classmethod
    def load(cls, directory=None, device="cuda", dtype=None, scale=DEFAULT_SCALE):
        """Construct the vendored model and load the baked weights.

        `sys.path` gains the baked directory rather than the file's own, because the vendored
        code imports itself by package name — `train_log.IFNet_HDv3` and `model.warplayer` — so
        the directory that must be importable is the one holding both.
        """
        directory = directory or model_dir()
        train_log = os.path.join(directory, "train_log")
        if not os.path.isfile(os.path.join(train_log, "flownet.pkl")):
            raise RuntimeError(
                "no flownet.pkl under {} — this image was built without the interpolator, or "
                "{} points somewhere else".format(train_log, MODEL_DIR_ENV))

        # **APPENDED, not inserted first, and that is the opposite of `worker_path`'s rule for a
        # reason.** This directory exports two importable names and one of them is `model` — as
        # generic a top-level name as Python has. Putting it first would make every later
        # `import model...` anywhere in the process resolve to Practical-RIFE's package, for the
        # life of the process, with no reset. Nothing in the vendored SeedVR2 tree or the
        # installed packages provides either name today, so appending resolves identically while
        # shadowing nothing — and if something ever does provide `model`, appending loses to it
        # rather than silently winning.
        if directory not in sys.path:
            sys.path.append(directory)

        from train_log.RIFE_HDv3 import Model  # noqa: PLC0415 — deliberate lazy heavy import

        model = Model()
        # `-1` is the rank that strips a `module.` prefix from the checkpoint's keys, which is
        # what a single-process load needs and what the reference script passes.
        model.load_model(train_log, -1)
        model.eval()
        return cls(model, device, dtype, scale).to(device=device, dtype=dtype)

    # ---- the interface `interpolate.Interpolator` uses ----------------------------------------

    def to(self, device=None, dtype=None):
        """Move and cast the network. Returns self, so `prepare()`'s rebind is a no-op."""
        if device is not None:
            self._device = device
        if dtype is not None:
            self._dtype = dtype
        net = self._model.flownet
        if self._dtype is not None:
            self._model.flownet = net.to(device=self._device, dtype=self._dtype)
        else:
            self._model.flownet = net.to(device=self._device)
        return self

    def eval(self):
        self._model.eval()
        return self

    def __call__(self, frame_a, frame_b, timestep, scale=None):
        """One synthesis. `frame_a`/`frame_b` are padded, cast and on the device already.

        The shim owns padding, cropping, dtype and the timestep; this owns the model call and
        nothing else. Keeping the split there is what lets the plan be tested with no GPU.
        """
        return self._model.inference(frame_a, frame_b, timestep,
                                     self._scale if scale is None else scale)
