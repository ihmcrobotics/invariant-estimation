import warnings
from dataclasses import dataclass

@dataclass(frozen=True)
class ContactNetConfig:
    """
    Every hyperparameter, in one immutable and static object.

    Never a pytree leaf and never is traced, this is passed to `network.init`,
    `rollout.make_segment_loss`, and `train.train` as plain Python scalars, which
    is what keeps `ContactNetParams` arrays-only (if it was non-Array,
    it would become a leaf and `jax.grad`/`optax` would try and update it).

    `frozen=True` also makes it hashable, so it can be a `static_argnum` if a call site
    needs it to be in the future.

    The two fields without defaults are the open numbers that cannot be guessed,
    as they are positional and cannot be omitted.
    """
    F: int
    """
    Per contact feature count (channels x subchain joints). Fixed by
    `features.py` once its channel ordering is frozen, with `d_in = H * F`.
    """

    sigma_0: float
    """
    Initial per axis contact STD [m].

    NOT read from the Java filter, as the port has no contact *measurement* noise
    (`N = J Sigma_q J^T` only), so the shipped constant for this socket is zero,
    which `softplus(.) + eps > 0` cannot represent. Pick it small enough such that
    `Sigma_C << J Sigma_q J^T` instead. Meausred on the test fixture, that term is
    1.26e-5 m^2 (3.5e-3 m std), so 1e-4 sits three orders of magnitude below the limit.

    NOTE: this needs to be re-measured on the real-model, so this becomes more accurate,
    as described in PORT_NOTES.md
    """
    # architecture
    H: int = 50 # history SAMPLES per evaluation (not span -- see `window_span_s`)

    window_span_s: float = 0.392
    """
    Seconds the history window reaches BACK over. The window is specified as a
    DURATION, and `stride` is derived from it and `dt` -- never the reverse.

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
    """
    Sim/filter tick period [s]. MUST match the loop this config is windowed on
    (`run_policy.DT`, and the estimator's own `dt`) -- it is the only thing
    converting `window_span_s` into ticks.
    """

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
    """
    Stance anchor proccess factor, held CONST during training run.

    The sim switches this 1e-4 <-> 1e1 from a contact detector (`sim/sensors.py`),
    freezing it removes that ground truth so the network cannot lean on the GT stance.
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

        This, not `H`, is the number to compare against the signal bandwidth.
        Measured on the 2026-07-17 log, joint position has f99 = 1.10 Hz and
        torque f99 = 4.25 Hz, so the span must cover a meaningful fraction of a
        stride (5.47 s there) rather than a fraction of a millisecond.
        """
        return (self.H - 1) * self.stride + 1

    @property
    def window_span_seconds(self) -> float:
        """The ACHIEVED span, ``(H-1)*stride*dt`` -- compare against `window_span_s`.

        Differs from the request whenever `stride` had to round (see there).
        Print this, never the request, when reporting what the network sees.
        """
        return (self.H - 1) * self.stride * self.dt

    @property
    def effective_rate_hz(self) -> float:
        """Sample rate the window actually observes, ``1/(stride*dt)``.

        The strided gather resamples at this rate; `features.boxcar` is the
        anti-alias filter for it, and its first null sits exactly here.
        """
        return 1.0 / (self.stride * self.dt)

    @property
    def nyquist_hz(self) -> float:
        """``0.5/(stride*dt)`` -- the number the bandwidth argument is about.

        Every channel's f99 must sit below this or it folds: torque 4.25 Hz,
        gyro 53.9 Hz, accel 155.6 Hz (2026-07-17 log). At the defaults this is
        62.5 Hz, so `q`, `tau` and gyro are clear and only the accelerometer is
        deliberately decimated -- its >Nyquist impact ringing is removed by the
        boxcar rather than scattered into the low band (PORT_NOTES.md).
        """
        return 0.5 / (self.stride * self.dt)

    @property
    def dof(self) -> int:
        """
        Contact measurement dimension. NIS ~ chi^2(dof), so a calibrated
        filter has `nis_over_dof == 1`.
        """
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



