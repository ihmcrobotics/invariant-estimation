import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class ContactNetConfig:
    """Every hyperparameter, in one immutable static object -- never a pytree leaf.

    Plain Python scalars only, so `ContactNetParams` stays arrays-only under
    `jax.grad`/`optax`. `frozen=True` keeps it hashable for `static_argnum`.
    """

    # F: per-contact feature count (channels x subchain joints), fixed by
    # features.py: [w(3), a(3), q(J_sub), qd(J_sub), tau(J_sub), p(3), v(3)]
    # = 12 + 3*J_sub = 30 at J_sub=6. d_in = H * F.
    F: int = 30

    # sigma_0: initial per-axis contact STD the head emits at init. Process
    # socket since 2026-07-29 (see config docstring history / TODO.md).
    sigma_0: float = 1.0e-4

    # architecture
    H: int = 20                    # history SAMPLES per evaluation (CoCo Table VI point)

    window_span_s: float = 0.019   # ~= (H-1)*dt so the derived `stride` rounds to 1
    """Seconds the history window reaches BACK over; `stride` is DERIVED from it
    and `dt`, never the reverse. At (H-1)*dt = 19 ms this yields stride==1 --
    consecutive ticks, full sensor bandwidth (CoCo Table V/VI reference)."""

    dt: float = 1.0e-3             # sim/filter tick period [s]; the only span->ticks conversion

    widths: tuple[int, ...] = (256, 256)  # trunk
    eps: float = 1.0e-6            # softplus floor on diag(L)

    # BPTT / data
    L: int = 128                   # ticks the gradient traverses
    B: int = 32                    # segments per batch

    # objective. "beta_nll" is accepted vocabulary but NOT implemented --
    # `rollout.make_segment_loss` raises NotImplementedError on it (the loss fn is
    # missing from losses.py, and the InEKF diagnostics publish no `logdet_S`).
    # `beta` is its plumbed-but-unused hyperparameter.
    objective: str = "l2_velocity"
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
    """Freeze the stance-anchor PROCESS socket at `contact_chol_const` (run-1).
    Default False: freezing pins swing feet as world-static and was measured the
    primary cause of run-1's collapse. The network never sees `contact_chol`."""

    # chained segments (see `dataset.ChainedBatcher`)
    warm_in_s: float = 1.0
    """Seconds a freshly seeded chain runs before its segments are trained on.
    A chain is seeded from ground truth (zero error); the gradient w.r.t.
    Sigma_C is meaningless until the error grows to its natural level (~1 s)."""

    episode_s: float = 43.0
    """Seconds a chain runs before re-seeding from ground truth. Bounds drift.
    43 s is this dataset's ceiling (62 s rollout - 16 s joint-KF warm-up leaves
    45.5 s of legal starts, minus warm_in_s)."""

    remat: bool = True

    @property
    def d_in(self) -> int:
        return self.F * self.H

    @property
    def stride(self) -> int:
        """Ticks between history samples: round(window_span_s / ((H-1)*dt)).

        A property, not a field: window_span_s and dt are the independent
        numbers, stride falls out of them. H samples fence off H-1 gaps, so
        reachable spans are integer multiples of (H-1)*dt. Clamped at 1 (the
        honest full-rate floor; features.window_indices rejects 0)."""
        if self.H <= 1:
            return 1
        return max(1, round(self.window_span_s / ((self.H - 1) * self.dt)))

    @property
    def window_span_ticks(self) -> int:
        """Ticks the window reaches back over: (H-1)*stride + 1."""
        return (self.H - 1) * self.stride + 1

    @property
    def window_span_seconds(self) -> float:
        """The ACHIEVED span, (H-1)*stride*dt -- print this, never the request."""
        return (self.H - 1) * self.stride * self.dt

    @property
    def effective_rate_hz(self) -> float:
        """Sample rate the window observes, 1/(stride*dt)."""
        return 1.0 / (self.stride * self.dt)

    @property
    def nyquist_hz(self) -> float:
        """0.5/(stride*dt). Every channel's f99 must sit below this or it folds."""
        return 0.5 / (self.stride * self.dt)

    @property
    def dof(self) -> int:
        """Contact measurement dimension. NIS ~ chi^2(dof)."""
        return 3 * self.n_contacts

    def __post_init__(self):
        # Built once, on the host -- validate loudly.
        if self.F <= 0 or self.H <= 0:
            raise ValueError(f"H and F must be positive, got H={self.H} and F={self.F}")
        if not self.dt > 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if not self.window_span_s > 0.0:
            raise ValueError(f"window_span_s must be positive, got {self.window_span_s}")
        if self.nyquist_hz < 10.0:
            # Warning not error: a coarse window is a legitimate sweep, and this
            # object is also built by tooling that only reads d_in. Loud because
            # the torque f99=4.25 Hz starts folding here.
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
            # Mirrors network.init's guard: softplus_inv of a non-positive arg is NaN/-inf.
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
            # An episode ending inside its own warm-in scores nothing and re-seeds forever.
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
