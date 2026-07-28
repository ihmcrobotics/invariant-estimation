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
    H: int = 50 # history SAMPLES per evaluation (not span -- see `stride`)
    stride: int = 8 # ticks between history samples; window spans (H-1)*stride+1
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
    def window_span_ticks(self) -> int:
        """Ticks the history window reaches back over: ``(H-1)*stride + 1``.

        This, not `H`, is the number to compare against the signal bandwidth.
        Measured on the 2026-07-17 log, joint position has f99 = 1.10 Hz and
        torque f99 = 4.25 Hz, so the span must cover a meaningful fraction of a
        stride (5.47 s there) rather than a fraction of a millisecond.
        """
        return (self.H - 1) * self.stride + 1

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
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
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



