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
    H: int = 20
    """History ticks per evaluation (CoCo Table VI point). CONSECUTIVE ticks at
    the full sensor rate: the window reaches back (H-1)*dt = 19 ms and there is
    no smoothing or decimation between them. The strided/boxcar window geometry
    (`window_span_s`, a derived `stride`, and the Nyquist guard it needed) was
    removed 2026-08-03 -- it had been pinned at stride==1 since the coherent
    1 kHz regime landed, so the boxcar was an identity and the span was just
    (H-1)*dt. See PORT_NOTES, "Dropping the boxcar and the strided window"."""

    dt: float = 1.0e-3             # sim/filter tick period [s]

    widths: tuple[int, ...] = (256, 256)  # trunk
    eps: float = 1.0e-6            # softplus floor on diag(L)

    # BPTT / data
    L: int = 128                   # ticks the gradient traverses
    B: int = 32                    # segments per batch

    # objective. Both are implemented: `l2_velocity` is the CoCo-faithful run-1
    # objective and the trusted baseline; `beta_nll` is the Seitzer beta-weighted
    # innovation NLL. The comment here used to say beta_nll was NOT implemented
    # while the default had already been flipped TO it -- so a run that meant to
    # be the L2 baseline silently trained beta_nll. Pass --objective explicitly.
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

    # training domain randomization on command (not env yet)
    cmd_vx_range: tuple = (0.30, 0.90)
    cmd_vy_range: tuple = (0.25, 0.50)
    cmd_yaw_range: tuple = (0.30, 1.50)
    cmd_resample_s: float = 3.0

    # seeds keep training deterministic and reproducible, but still subject to the stochasticity of the environment and the dataset
    init_seed: int = 0
    batcher_seed: int = 0

    # environment variables for domain randomization
    env_dr: bool = False
    friction_range: tuple = (0.6, 1.2)
    friction_low_tail_prob: float = 0.25
    # MEASURED, 2026-08-05: the original (0.15, 0.45) tail put mu as low as 0.15 --
    # effectively ice -- and the flat-trained policy fell on it. An axis ablation
    # through the real collect path (friction-only vs pushes-only) attributed the
    # falls to FRICTION, not to the pushes: with pushes at their configured
    # 30-120 N the robot stayed up. 7 of the first 8 DR rollouts were lost this
    # way, INCLUDING one on flat ground, which is what ruled the terrain out.
    # 0.45-0.70 against a ~1.0 nominal is still a real slip regime; a tail the
    # policy cannot survive yields no data at all, which trains nothing.
    friction_low_tail: tuple = (0.45, 0.70)
    disturb_rate_hz: float = 0.4
    disturb_mag_N: tuple = (30.0, 120.0)
    disturb_dur_s: float = 0.1
    terrain_mix: tuple = (("flat", 0.25), ("waves", 0.25), ("stepping_stones",0.25), ("hard_stepping",0.25))

    @property
    def d_in(self) -> int:
        return self.F * self.H

    @property
    def window_span_ticks(self) -> int:
        """Ticks the window reaches back over: H consecutive ticks."""
        return self.H

    @property
    def window_span_seconds(self) -> float:
        """Seconds the window reaches back over, (H-1)*dt."""
        return (self.H - 1) * self.dt

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
        # No Nyquist guard: the window samples consecutive ticks, so it observes
        # the full 1/dt rate and nothing can alias into it.
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
