import warnings
from dataclasses import dataclass

@dataclass(frozen=True)
class ContactNetConfig:
    """Every hyperparameter, in one immutable static object — never a pytree leaf, never traced.

    Passed to `network.init`, `rollout.make_segment_loss` and `train.train` as
    plain Python scalars, which is what keeps `ContactNetParams` arrays-only: a
    non-Array field here would become a leaf and `jax.grad`/`optax` would try to
    update it. `frozen=True` also makes it hashable, so it can be a
    `static_argnum` if a call site ever needs that.
    """
    F: int
    """Per-contact feature count (channels × subchain joints), fixed by `features.py`; ``d_in = H * F``."""

    sigma_0: float
    """Initial per-axis contact STD — the constant `network.init` makes the head emit.

    **1e-4 is a MEASUREMENT-socket number and does not transfer.** Its
    justification was: the port has no contact measurement noise (`N = J Sigma_q
    J^T` only), so the shipped constant for that socket is zero, which
    `softplus(.) + eps > 0` cannot represent; pick it small enough that `Sigma_C
    << J Sigma_q J^T` instead. Measured on the test fixture that term is 1.26e-5
    m^2 (3.5e-3 m std), so 1e-4 sat three orders below.

    Since 2026-07-29 the network drives the PROCESS socket, where that argument
    does not exist. The heuristic's range there is 1e-4 (stance) to 1e1 (swing)
    **as Cholesky factors**, i.e. variance 1e-8 to 1e2. `network.init` zeroes the
    head, so iteration 0 emits a CONSTANT `sigma_0 * I` at every gait phase -- and
    at the stance end that is `freeze_contact_chol`, measured 10.2x worse in
    body-frame velocity than not using contacts at all, which is the run-1
    configuration.

    **Choose the initialization deliberately before the next run** -- see TODO.md,
    "Initialization on the process socket". Nothing here enforces it. Still worth
    re-measuring on the real model (PORT_NOTES.md).
    """

    # architecture
    H: int = 50 # history SAMPLES per evaluation (not span -- see `window_span_s`)

    window_span_s: float = 0.392
    """Seconds the history window reaches BACK over; `stride` is DERIVED from it and `dt`, never the reverse.

    The span, not the sample count, is what has to be argued about. Measured on
    the 2026-07-17 Alex log (1 kHz): joint position f99 = 1.10 Hz, joint torque
    f99 = 4.25 Hz, gyro f99 = 53.9 Hz, accel f99 = 155.6 Hz, and the walking
    stride fundamental is 0.183 Hz (5.47 s). At those bandwidths `H` consecutive
    ticks spans 20 ms and carries one value plus a slope; the fix was to spread
    the SAME `H` samples over a meaningful fraction of a gait stride
    (PORT_NOTES.md, "H is not 20"). 0.392 s is the benchmarked choice.

    A hardcoded tick count silently means a different duration at a different
    rate: at the 200 Hz `run_policy.DT` that shipped until 2026-07-27, the old
    literal `stride = 8` meant a 1.96 s window and a 12.5 Hz Nyquist -- a
    completely different network input, with nothing raising. Three tests in
    `tests/sim/test_sensors.py` broke in exactly this way (see its `_ticks`).
    """

    dt: float = 1.0e-3
    """Sim/filter tick period [s]. MUST match the loop this config is windowed on
    (`run_policy.DT`, and the estimator's own `dt`) -- it is the only thing
    converting `window_span_s` into ticks."""

    widths: tuple[int, ...] = (256, 256) # trunk
    eps: float = 1.0e-6 # softplus floor on diag(L)

    # BPTT / data
    L: int = 128 # ticks the gradient traverses
    B: int = 32 # segments per batch

    # objective
    objective: str = "l2_velocity" # or Beta-NLL when that's ready to be used.
    beta: float = 0.5

    # optimizer
    peak_lr: float = 1.0e-4
    warmup_steps: int = 100
    total_steps: int = 10_000
    max_norm: float = 1.0
    weight_decay: float = 0.0

    # filter coupling
    n_contacts: int = 2
    contact_chol_const: float = 1.0e-4
    """Stance-anchor process factor used ONLY when `freeze_contact_chol` is set;
    kept so the run-1 configuration stays reproducible."""

    freeze_contact_chol: bool = False
    """Freeze the stance-anchor PROCESS socket at `contact_chol_const` (run-1 behaviour).

    `False` is the default because freezing it was measured to be the primary
    cause of run 1's collapse. `sim/sensors.py` drives `contact_chol` from
    `ContactTrust` -- the Schmitt-trigger port of
    `FootSwitchContactProbabilityProvider`, off normal force `f_n/(0.5 m g)` --
    switching 1e-4 (stance) <-> 1e1 (swing). Freezing it at the STANCE value
    pins swing feet as world-static, so the contact update fights a process model
    that is wrong for half the gait.

    Measured (`experiments/measure_tstar.py`, PORT_NOTES): at the 128 ms segment
    horizon the frozen config is **10.2x worse** in body-frame velocity error
    than not using contacts at all, and stays worse out to 4 s. With the recorded
    value it is 0.92x at 128 ms and 0.16x at 4 s. The network's only defence
    against the frozen version is `Sigma_C -> infinity`, which is exactly what
    run 1 learned.

    The original argument for freezing (`dataset.py` docstring (b)) was that the
    recorded value would hand the network a free ground-truth contact flag. It
    does not: **the network never sees `contact_chol`.** Its input is the 24
    feature channels; `contact_chol` enters the *filter's process model* only, so
    feeding the real value changes the filter ContactNet is differentiated
    through, not the information ContactNet receives. It is also not ground
    truth -- `ContactTrust` is a sensor-derived estimate, the same one the
    deployed filter uses.

    Freezing also fights `inEKF/filter.py`'s DECISION (theory doc S3.2): contact
    condition belongs in the process noise, and the FK measurement is not wrong
    during swing. The freeze removes the correct lever and asks the measurement
    socket to compensate.

    This is the canonical statement; `dataset.py`, `rollout.make_warm_in` and
    `online.make_provider` cross-reference it.
    """

    # chained segments (see `dataset.ChainedBatcher`)
    warm_in_s: float = 1.0
    """Seconds a freshly seeded chain runs before its segments are trained on.

    A chain is seeded from ground truth, so it starts at **zero** estimation
    error -- a state the deployed filter is never in. Until the error grows to
    its natural level the contact update has nothing to correct and the gradient
    w.r.t. `Sigma_C` is meaningless (that is the run-1 failure).

    1.0 s, from the 30 s arm-C curve in `experiments/measure_tstar.py`: the
    filter's body-frame velocity error saturates at ~8.5e-2 m/s and is already
    there by 1 s (8.20e-2 at 1 s, 8.45e-2 at 2 s, 8.71e-2 at 4 s, 8.31e-2 at
    29 s). Everything past ~1 s samples the same distribution, so a longer
    warm-in is pure cost: one warm-in scan is **1118 ms**, and at the original
    2.0 s / 20 s settings warm-ins were 310 ms of run 2's 598 ms per step -- 52%
    of the run.
    """

    episode_s: float = 43.0
    """Seconds a chain runs before being re-seeded from ground truth.

    Bounds how far the filter may drift. CoCo-InEKF (arXiv 2605.15122) uses
    T = 100 s (dancing) / 6 s (ground motions).

    43 s is the **ceiling this dataset allows**, not a free choice: a 62 s
    rollout minus the 16 s joint-KF warm-up leaves 45.5 s of legal segment
    starts, so a chain can run at most ``45.5 - warm_in_s``. Setting 100 s here
    would simply never fire -- the rollout-end re-seed would always trip first.
    A true 100 s episode needs rollouts collected at ``--seconds 120``+.

    Raising it costs nothing in signal (the error distribution is flat from 1 s
    to 30 s, see `warm_in_s`) and buys wall time: re-seeds drop from 0.277 to
    ~0.10 per step, and each one is a 1118 ms warm-in scan.
    """

    remat: bool = True

    @property
    def d_in(self) -> int:
        return self.F * self.H

    @property
    def stride(self) -> int:
        """Ticks between history samples, DERIVED: ``round(window_span_s / ((H-1)*dt))``.

        A property and not a field on purpose: `window_span_s` and `dt` are the
        two independently-meaningful numbers, and `stride` is what falls out of
        them. Storing it too would allow the three to disagree.

        `H` samples fence off ``H - 1`` gaps, so the reachable spans are the
        integer multiples of ``(H-1)*dt`` -- 49 ms at the defaults. A requested
        span between two multiples is rounded to the nearest, so the ACHIEVED
        span (`window_span_seconds`) can differ from the request; that is real,
        not a wart to work around, and it is why the achieved value is reported
        rather than the requested one. Ties go to even (Python `round`).

        Clamped at 1, both because `features.window_indices` rejects 0 and
        because ``stride = 1`` is the honest floor: full-rate sampling. `H == 1`
        has no gaps at all, so no spacing is defined and it degenerates to 1.
        """
        if self.H <= 1:
            return 1
        return max(1, round(self.window_span_s / ((self.H - 1) * self.dt)))

    @property
    def window_span_ticks(self) -> int:
        """Ticks the history window reaches back over: ``(H-1)*stride + 1``.

        This, not `H`, is the number to compare against the signal bandwidth
        (see `window_span_s` for the measured f99s and the stride fundamental).
        """
        return (self.H - 1) * self.stride + 1

    @property
    def window_span_seconds(self) -> float:
        """The ACHIEVED span, ``(H-1)*stride*dt`` -- print this, never the request.

        Differs from `window_span_s` whenever `stride` had to round (see there).
        """
        return (self.H - 1) * self.stride * self.dt

    @property
    def effective_rate_hz(self) -> float:
        """Sample rate the window actually observes, ``1/(stride*dt)``.

        `features.boxcar` is the anti-alias filter for it, and its first null
        sits exactly here.
        """
        return 1.0 / (self.stride * self.dt)

    @property
    def nyquist_hz(self) -> float:
        """``0.5/(stride*dt)`` -- the number the bandwidth argument is about.

        Every channel's f99 (see `window_span_s`) must sit below this or it
        folds. At the defaults this is 62.5 Hz, so `q`, `tau` and gyro are clear
        and only the accelerometer is deliberately decimated -- its >Nyquist
        impact ringing is removed by the boxcar rather than scattered into the
        low band (PORT_NOTES.md).
        """
        return 0.5 / (self.stride * self.dt)

    @property
    def dof(self) -> int:
        """Contact measurement dimension. NIS ~ chi^2(dof), so a calibrated
        filter has `nis_over_dof == 1`."""
        return 3 * self.n_contacts

    def __post_init__(self):
        # Config is built once, on the host -- validate "loudly" rather
        # than a bad value coming in later by accident.
        if self.F <= 0 or self.H <= 0:
            raise ValueError(f"H and F must be positive, got H={self.H} and F={self.F}")
        if not self.dt > 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if not self.window_span_s > 0.0:
            raise ValueError(
                f"window_span_s must be positive, got {self.window_span_s}"
            )
        if self.nyquist_hz < 10.0:
            # A warning and NOT an error: a coarse window is a legitimate thing
            # to sweep, and this object is also built by tooling that only wants
            # to read `d_in`. But it must be loud -- the torque channel's
            # f99 = 4.25 Hz starts folding here, and folded energy is
            # sampling-phase dependent, so it mimics signal instead of failing.
            warnings.warn(
                f"history window Nyquist is {self.nyquist_hz:.2f} Hz "
                f"(stride={self.stride} at dt={self.dt}s, span "
                f"{self.window_span_seconds:.3f}s): below ~10 Hz the torque "
                f"channel (f99 = 4.25 Hz) aliases. Shorten window_span_s or "
                f"raise H.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not self.widths or any(w <= 0 for w in self.widths):
            raise ValueError(f"widths must be non-empty and positive, got {self.widths}")
        if not self.sigma_0 > self.eps:
            # Mirrors `network.init`'s guard, one layer earlier: softplus_inv of
            # a non-positive argument is NaN or -inf
            raise ValueError(
                f"need sigma_0 > eps, got sigma_0={self.sigma_0}, eps={self.eps}"
            )

        if self.L <= 0 or self.B <= 0:
            raise ValueError(f"L and B must be positive, got L={self.L} and B={self.B}")
        if self.objective not in ("beta_nll", "l2_velocity"):
            raise ValueError(f"Unknown objective: {self.objective!r}")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"Beta must be in [0,1], got {self.beta}")
        if self.warmup_steps >= self.total_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < total_steps"
                f"({self.total_steps}); the schedule will never reach peak otherwise."
            )
        if self.n_contacts <= 0:
            raise ValueError(f"n_contacts must be positive, got {self.n_contacts}")
        if self.warm_in_s < 0.0:
            raise ValueError(f"warm_in_s must be >= 0, got {self.warm_in_s}")
        if self.episode_s <= self.warm_in_s + self.L * self.dt:
            # An episode that ends inside its own warm-in scores nothing, and the
            # chain would re-seed forever without ever contributing a gradient.
            raise ValueError(
                f"episode_s ({self.episode_s}) must exceed warm_in_s "
                f"({self.warm_in_s}) plus one segment ({self.L * self.dt}); "
                f"otherwise no chain ever produces a trainable segment."
            )

    @property
    def warm_in_ticks(self) -> int:
        return int(round(self.warm_in_s / self.dt))

    @property
    def episode_ticks(self) -> int:
        return int(round(self.episode_s / self.dt))
