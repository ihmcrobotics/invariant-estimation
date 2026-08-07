r"""experiments/z_budget.py — an EXACT additive decomposition of the InEKF height error.

Motivation
----------
The closed-loop InEKF sinks at a near-constant −0.013…−0.018 m/s while walking
(`results.md`, `~/Documents/filter-debugging/sink-derivation.pdf`).  Every previous
attribution has been a *difference of runs* ("turn off X, see what moves"), which
mixes the effect of X with the feedback the change induces.  This script instead
builds a **budget that closes to machine precision**: every metre of height error
is assigned to exactly one of six named terms, and their sum is checked against the
observed error at 1e-12.  A term can then be read as "this is what that mechanism
put in", with no counterfactual involved.

The decomposition
-----------------
Write ``e_z = p̂_z − p_z^true``.  One estimator tick runs three sub-steps, so the
mean's height moves in three places (`inEKF/filter.py::make_step`):

    p0 --propagate--> p1 --contact FK update--> p2 --gravity leveling--> p3

Each update is applied as ``X̂⁺ = exp(−ξ)X̂`` (I5), i.e. a LEFT multiplication, so
its effect on ``p`` is an affine map ``p ↦ ΔR·p + t`` — a rotation about the WORLD
ORIGIN plus a translation.  Those two are separated because they are different
physics: the ``ΔR·p`` part is a lever arm that grows with how far the robot has
walked, the ``t`` part is not.  With ``ΔR_C = R2 R1ᵀ`` and ``ΔR_G = R3 R2ᵀ``,

    Δe_z  =  PROP_CARRY   (v̂_z − v_z^true)·dt          velocity error, integrated
           + PROP_2ND     the O(dt²) remainder of propagation
           + CONT_ROT     (ΔR_C p1 − p1)_z             contact update, lever arm
           + CONT_TRANS   (p2 − ΔR_C p1)_z             contact update, translation
           + GRAV_ROT     (ΔR_G p2 − p2)_z             gravity update, lever arm
           + GRAV_TRANS   (p3 − ΔR_G p2)_z             gravity update, translation

exactly (`_position_budget`).  Since PROP_CARRY normally dominates, the velocity
error gets its own closing budget (`_velocity_budget`).  Propagation writes
``v̂⁺ = v̂ + R̂ u dt + g dt`` with ``u = Γ₁(ω dt) ā``; recovering ``u`` from the
recorded stages (``u = R̂ᵀ(Δv̂_prop − g dt)/dt``, exact) splits it against truth:

    Δe_v,z =  PV_ATT     ((R̂ − R^true) u)_z dt          attitude error x specific force
            + PV_SENS    (R^true u)_z dt + g_z dt − Δv_z^true
                                                        everything sensor/model:
                                                        IMU lever arm, R_mount,
                                                        gravity constant, Γ₁ / dt
            + CV_CONT    contact update's pull on v̂     (rot + trans)
            + CV_GRAV    gravity update's pull on v̂

Finally the **common-mode deposit**, the quantity the null-space derivation says
can never be removed: the contact update moves the base by ``Δp`` and each anchor
by ``Δd_i``; the orthogonal projection of that onto
``C = {ξ_p = ξ_d_i = δ}`` is ``δ = (Δp + Σ Δd_i)/(N+1)``.  Its z component,
accumulated, is the height that went where ``H`` cannot see it.

Fidelity
--------
The tracing InEKF step is a copy of `inEKF/filter.make_step` with extra outputs, so
it could silently drift from the shipped filter.  `--verify` re-runs the SHIPPED
step over the recorded `inekf_inputs` from the same seed and asserts the position
trajectories agree to 1e-12.  Run it at least once after touching either file.

Usage
-----
    uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --verify
    uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --yaw 0.3 --out results/zb_c2.npz
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import NamedTuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import mujoco                                                     # noqa: E402

from jax import Array                                             # noqa: E402

import run_estimator as rest                                      # noqa: E402
import run_policy as rp                                           # noqa: E402

from invariant_estimation import inEKF as inf                     # noqa: E402
from invariant_estimation.inEKF.contact import digest             # noqa: E402
from invariant_estimation.inEKF.correct import (                  # noqa: E402
    UpdateDiagnostics, innovation, linear_update, measurement_noise,
)
from invariant_estimation.inEKF.gravity_update import (           # noqa: E402
    assemble_gravity_leveling, is_quasi_static, update_gravity_reference,
)
from invariant_estimation.inEKF.propagate import propagate        # noqa: E402
from invariant_estimation.inEKF.state import InEKFState           # noqa: E402
from invariant_estimation.pipeline import main_estimator as me    # noqa: E402
from invariant_estimation.sim.estimator_loop import EstimatorRuntime  # noqa: E402
from invariant_estimation.sim.sensors import IMUNoise                # noqa: E402


# ---------------------------------------------------------------------------
# The tracing InEKF step — `inEKF/filter.make_step` plus per-stage means
# ---------------------------------------------------------------------------

class ZOutputs(NamedTuple):
    """`inf.InEKFOutputs` widened with the per-stage means the budget needs.

    Stage index: 0 = tick entry, 1 = after propagate, 2 = after the contact FK
    update, 3 = after gravity leveling (== next tick's stage 0).
    """
    state: InEKFState
    contact_innovation: Array
    contact_diagnostics: UpdateDiagnostics
    gravity_diagnostics: UpdateDiagnostics
    tilt_angle: Array
    quasi_static: Array
    p_stage: Array          # (4, 3)   base position
    v_stage: Array          # (4, 3)   base velocity
    R_stage: Array          # (4, 3, 3) base attitude
    d_stage: Array          # (4, 3)   MEAN contact anchor (over the N slots)
    d_slots: Array          # (4, N, 3) per-slot anchors
    sigma_c: Array          # (N, 3, 3) digested contact process covariance
    P_diag: Array           # (3N+9,)  prior covariance diagonal (post-propagate)
    P_vp: Array             # (3, 3)   velocity-position cross block
    P_pd0: Array            # (3, 3)   position-anchor-0 cross block
    Np_body: Array          # (N, 3, 3) contact measurement noise AS FED to the update
    frame_gap: Array        # scalar   ‖R̂ Np R̂ᵀ − Np‖_F / ‖Np‖_F, slot 0


def _row_mask(dim: int, block: str) -> Array:
    r"""Column vector that zeroes chosen tangent rows of ``K`` (rotation-first, I4).

    An EXPERIMENT knob, not a filter feature: the contact update's `H` has no
    rotation and no velocity columns, so everything it writes into ``R̂`` and ``v̂``
    comes through ``P``'s cross-covariance blocks.  Zeroing those rows of ``K``
    severs exactly that path and nothing else — the position and anchor corrections
    are untouched.  Joseph form still yields a valid (now conservative) ``P`` for
    the gain actually applied.
    """
    m = np.ones(dim)
    if block in ("rot", "rotvel"):
        m[0:3] = 0.0
    if block in ("vel", "rotvel"):
        m[3:6] = 0.0
    return jnp.asarray(m, dtype=jnp.float64)[:, None]


def _masked_linear_update(state, H, residual, R, mask):
    """`correct.linear_update` with `K` row-masked. Kept in lockstep with it by hand."""
    from jax.scipy.linalg import cho_factor, cho_solve

    from invariant_estimation.config import section
    from invariant_estimation.inEKF.correct import apply_correction, joseph_update

    S = H @ state.P @ H.T + R
    S = 0.5 * (S + S.T)
    factor = cho_factor(S)
    diag = jnp.abs(jnp.diag(factor[0]))
    condition_proxy = (jnp.max(diag) / jnp.min(diag)) ** 2
    K = cho_solve(factor, H @ state.P).T
    nis = residual @ cho_solve(factor, residual)
    applied = (condition_proxy < float(section("inekf")["cond_max"])).astype(jnp.float64)
    K = applied * K * mask
    xi = K @ residual
    return (
        apply_correction(state, xi)._replace(P=joseph_update(state.P, K, H, R)),
        UpdateDiagnostics(
            applied=applied, nis=nis, condition_proxy=condition_proxy,
            correction_rotation_norm=jnp.linalg.norm(xi[0:3]),
            logdet_s=2.0 * jnp.sum(jnp.log(diag)),
        ),
    )


def make_tracing_inekf_step(ekf, kinematics, mask_block: str = "none",
                            rotate_R: bool = False):
    r"""Byte-for-byte the shipped `inf.make_step` body, emitting the stage means.

    Kept as a literal copy rather than a wrapper: the point is to observe the
    intermediate states, which the shipped step does not expose.  `--verify`
    guards the copy against drift.

    ``rotate_R`` applies ``R̂ Np R̂ᵀ`` before the contact update — the rotation the
    shipped scan path omits.  ``Np = J Σ_q Jᵀ`` is a **body-frame** covariance
    (``_make_contact_kinematics`` returns ``y = ᴮR_Wᵀ(p_foot − p_B)`` and ``ᴮR_W``
    does not depend on ``q``, so ``J`` is the body-frame Jacobian), while the
    residual ``ν = R̂y − (d̂ − p̂)`` is **world-frame**.  The Java behavioural
    contract is explicit — *"contact/measurement covariance supplied body-frame
    and rotated by R"* (`TEST_SUITE_MAP.md:893`) — and the port's own test says
    *"`correct` consumes world-frame noise; `contact_update` rotates it itself"*
    (`tests/inEKF/test_contact_updater.py:277`).  The scan path does neither.
    """
    gravity_params = ekf.gravity_params
    mask = None if mask_block == "none" else _row_mask(3 * ekf.N + 9, mask_block)

    def step(carry, inputs):
        state, gravity_ref = carry
        s0 = state

        sigma_c = digest(inputs.contact_chol, ekf.params)
        state = propagate(state, inputs.omega, inputs.accel, sigma_c, ekf.params)
        s1 = state

        frames = kinematics(inputs.joint.q, inputs.joint.q_dot)
        Np_raw = inf.contact_position_noise(frames.J, inputs.joint.sigma_q)
        # R̂ Np R̂ᵀ, vectorised over contacts — the omitted rotation (see docstring).
        Np_world = jnp.einsum("ij,njk,lk->nil", state.R, Np_raw, state.R)
        Np = Np_world if rotate_R else Np_raw
        gap = (jnp.linalg.norm(Np_world[0] - Np_raw[0])
               / jnp.maximum(jnp.linalg.norm(Np_raw[0]), 1e-300))

        P_prior = state.P
        nu = innovation(state, frames.y)
        if mask is None:
            state, contact_diagnostics = linear_update(
                state, ekf.params.H, nu, measurement_noise(Np)
            )
        else:
            state, contact_diagnostics = _masked_linear_update(
                state, ekf.params.H, nu, measurement_noise(Np), mask
            )
        s2 = state

        gate = is_quasi_static(
            gravity_ref, inputs.accel, inputs.raw_omega, gravity_params
        ).astype(jnp.float64)
        meas = assemble_gravity_leveling(
            gravity_ref, state, inputs.accel, gravity_params
        )
        state, gravity_diagnostics = linear_update(
            state, meas.H, meas.residual, meas.R,
            cond_max=gravity_params.cond_max, gate=gate,
        )
        s3 = state
        gravity_ref = update_gravity_reference(
            meas.ref, inputs.accel, inputs.raw_omega, ekf.params.dt, gravity_params
        )

        stages = (s0, s1, s2, s3)
        outputs = ZOutputs(
            state=state,
            contact_innovation=nu,
            contact_diagnostics=contact_diagnostics,
            gravity_diagnostics=gravity_diagnostics,
            tilt_angle=meas.tilt_angle,
            quasi_static=gate,
            p_stage=jnp.stack([s.p for s in stages]),
            v_stage=jnp.stack([s.v for s in stages]),
            R_stage=jnp.stack([s.R for s in stages]),
            d_stage=jnp.stack([s.d.mean(axis=0) for s in stages]),
            d_slots=jnp.stack([s.d for s in stages]),
            sigma_c=sigma_c,
            P_diag=jnp.diag(P_prior),
            P_vp=P_prior[3:6, 6:9],
            P_pd0=P_prior[6:9, 9:12],
            Np_body=Np,
            frame_gap=gap,
        )
        return inf.InEKFCarry(state=state, gravity_ref=gravity_ref), outputs

    return step


# ---------------------------------------------------------------------------
# Runtime / loop subclasses that keep the full per-substep record
# ---------------------------------------------------------------------------

class TracingRuntime(EstimatorRuntime):
    """`EstimatorRuntime` that retains every substep of the scanned outputs."""

    def advance(self, batch):
        if self.carry is None:
            raise RuntimeError("call seed() before advance()")
        self.carry, out = self._advance(self.carry, self._stack(batch))
        self.full = out
        self.last = self._view(out, batch[-1])
        return self.last


class TracingSensorReader(rest.SimSensorReader):
    r"""`SimSensorReader` that publishes MuJoCo's own contact set alongside the
    `ContactTrust` decision, and can drive `Σ_C` from truth instead.

    No pool records ground-truth contact (`sim/collect.py` keeps only
    `reader.truth(d)`), so the FP/FN question cannot be asked of existing data.
    Measuring it here instead of re-collecting has the added virtue of being the
    *same* closed-loop run the drift is measured in.

    ``contact_source = "oracle"`` replaces the Schmitt/dwell decision with
    ``normal force > 0`` at zero latency — the causal test for hypothesis (a).
    """
    contact_source = "trust"

    def read(self, d):
        sensors = super().read(d)
        forces = self.contact_forces(d)
        self.last_truth = (forces > 0.0).astype(float)
        self.last_load = self.contact_loads(d)
        if self.contact_source == "oracle":
            trusted = self.last_truth
            chol = np.where(trusted[:, None, None] > 0.0,
                            self.stance_chol, self.swing_chol)
            sensors = sensors._replace(
                contact=trusted,
                contact_chol=chol * np.tile(np.eye(3), (len(trusted), 1, 1)))
        return sensors


class TracingContactNetRuntime(rest.ContactNetRuntime):
    """`ContactNetRuntime` that retains every substep, same as `TracingRuntime`.

    A separate class because `ContactNetRuntime` binds `EstimatorRuntime` at class
    creation, so patching the base name does not reach it.
    """

    def advance(self, batch):
        if self.carry is None:
            raise RuntimeError("call seed() before advance()")
        sensors = self._stack(batch)
        self._ostate, contact_chol = self._provider_scan(self._ostate, sensors)
        self.carry, out = self._advance(
            self.carry, sensors._replace(contact_chol=contact_chol))
        self.full = out
        self.last = self._view(out, batch[-1])
        return self.last


class TracingLoop(rest.EstimatedLoop):
    """`EstimatedLoop` that snapshots truth at every ESTIMATOR tick, not just control ticks."""

    # Out-of-distribution knobs for hypothesis (b). Class attributes so the loop
    # can be constructed by `make_estimated_loop` without threading them through.
    ood_friction: float | None = None
    ood_payload: float = 0.0
    ood_push_z: float = 0.0
    terrain_field = None            # (N, N) heightfield, or None for the plane

    def _apply_ood(self):
        r"""Push the environment outside the ContactNet training distribution.

        The pool randomised friction over [0.45, 1.2], applied to slide-friction
        column 0 of both foot geoms AND the floor (MuJoCo mixes pairwise friction
        by element-wise max, so setting one alone does nothing). Payload and
        VERTICAL forcing were never randomised at all — training pushes are
        horizontal (`collect.py:153`), which is why they are the interesting axes
        for a *vertical* drift question.
        """
        if self.ood_friction is not None:
            gids = list(self.reader.foot_gids) + [
                mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")]
            for g in gids:
                if g >= 0:
                    self.m.geom_friction[g, 0] = float(self.ood_friction)
            print(f"  OOD friction mu = {self.ood_friction}")
        if self.ood_payload:
            bid = self.reader.base_bid
            self.m.body_mass[bid] += float(self.ood_payload)
            print(f"  OOD payload +{self.ood_payload} kg on the pelvis "
                  f"(now {self.m.body_mass[bid]:.2f} kg)")
        if self.ood_push_z:
            print(f"  OOD vertical forcing +-{self.ood_push_z} N at 0.7 Hz "
                  f"(training pushes are horizontal-only)")

    def __init__(self, *a, x0: float = 0.0, **kw):
        super().__init__(*a, **kw)
        if self.terrain_field is not None:
            # RAISE the spawn clear of the relief — `+=`, not `=`. Assigning puts
            # the pelvis at field.max()+0.02 instead of its nominal ~0.9 m, i.e.
            # buried to the chest, which is the bug `collect.py:250` records.
            self.d.qpos[2] += float(self.terrain_field.max()) + 0.02
            mujoco.mj_forward(self.m, self.d)
            self.rt.seed(self.d)
            self.batch = [self.reader.read(self.d) for _ in range(len(self.batch))]
            print(f"  terrain: relief {self.terrain_field.max():.3f} m, "
                  f"spawn raised to z={self.d.qpos[2]:.3f}")
        if self.ood_friction is not None or self.ood_payload or self.ood_push_z:
            self._apply_ood()
            mujoco.mj_forward(self.m, self.d)
        if x0:
            # Translate the whole experiment along +x BEFORE anything steps or is
            # seeded. Flat ground and a translation-invariant policy make this a
            # pure change of where the WORLD ORIGIN sits relative to the robot --
            # which is the only thing the `exp(-ξ)` left-multiplication's rotation
            # lever arm (`ΔR·p`) depends on. Anything that scales with it is a
            # gauge artifact of the world-centric parameterisation, not physics.
            self.d.qpos[0] += float(x0)
            mujoco.mj_forward(self.m, self.d)
            self.rt.seed(self.d)
            self.batch = [self.reader.read(self.d) for _ in range(len(self.batch))]
        self.truth_batch = [self.reader.truth(self.d) for _ in range(len(self.batch))]
        self.rec: dict[str, list] = {k: [] for k in
                                     ("p_stage", "v_stage", "R_stage", "d_stage",
                                      "d_slots", "quasi_static", "contact_innovation",
                                      "sigma_c", "P_diag", "P_vp", "P_pd0",
                                      "Np_body", "frame_gap",
                                      "p_true", "v_true", "R_true",
                                      "w_true", "accel", "omega", "contact")}
        self.rec_inputs: list = []           # the InEKF boundary inputs, for --verify
        self.seed_carry = None               # the InEKF carry before the first tick
        # Populated only when the reader is a `TracingSensorReader`; otherwise these
        # stay empty and the recorded key set is unchanged (matrix cells stay
        # bit-identical to the ones collected before this was added).
        self.contact_truth: list = []
        self.contact_load: list = []
        self.rec.setdefault("contact_true", [])
        self.rec.setdefault("contact_load", [])
        # Prime AFTER the lists exist. Doing it earlier (next to `truth_batch`) is
        # silently undone by the initialisation above, which drops the first
        # control tick's samples and shifts `contact_true` against `contact` by one
        # batch for the rest of the run. The `--contact-source oracle` self-check
        # below exists precisely to catch that: with the oracle, FP and FN must be
        # exactly zero, because the trust vector IS the truth vector.
        if hasattr(self.reader, "last_truth"):
            self.contact_truth = [self.reader.last_truth] * len(self.batch)
            self.contact_load = [self.reader.last_load] * len(self.batch)

    def _harvest(self):
        f = self.rt.full
        for k in ("p_stage", "v_stage", "R_stage", "d_stage", "d_slots", "quasi_static",
                  "contact_innovation", "sigma_c", "P_diag", "P_vp", "P_pd0",
                  "Np_body", "frame_gap"):
            self.rec[k].append(np.asarray(getattr(f.inekf, k)))
        self.rec_inputs.append(f.inekf_inputs)
        self.rec["accel"].append(np.asarray(f.inekf_inputs.accel))
        self.rec["omega"].append(np.asarray(f.inekf_inputs.omega))
        self.rec["p_true"].append(np.stack([t["p"] for t in self.truth_batch]))
        self.rec["v_true"].append(np.stack([t["v"] for t in self.truth_batch]))
        self.rec["R_true"].append(np.stack([t["R"] for t in self.truth_batch]))
        self.rec["w_true"].append(np.stack([t["omega"] for t in self.truth_batch]))
        if self.contact_truth:
            self.rec["contact_true"].append(np.stack(self.contact_truth))
            self.rec["contact_load"].append(np.stack(self.contact_load))
        self.rec["contact"].append(np.stack([np.asarray(s.contact) for s in self.batch]))

    def _tick_sync(self):
        if self.seed_carry is None:
            self.seed_carry = self.rt.carry[1]
        est = self.rt.advance(self.batch)
        self._harvest()
        self.batch, self.truth_batch = [], []
        self.contact_truth, self.contact_load = [], []
        self._act_on(est)
        self._record(est)
        if self.ood_push_z:
            # Deterministic sinusoid rather than a Poisson process: the question is
            # whether the estimator degrades under vertical excitation at all, and a
            # fixed waveform makes the seeds comparable.
            self.d.xfrc_applied[self.reader.base_bid, 2] = self.ood_push_z * np.sin(
                2 * np.pi * 0.7 * self.d.time)
        for k in range(rp.DECIMATION):
            mujoco.mj_step(self.m, self.d)
            if (k + 1) % self.est_every == 0:
                self.batch.append(self.reader.read(self.d))
                self.truth_batch.append(self.reader.truth(self.d))
                if hasattr(self.reader, "last_truth"):
                    self.contact_truth.append(self.reader.last_truth)
                    self.contact_load.append(self.reader.last_load)
        self._ramp_t += rp.DECIMATION * rp.DT

    def stacked(self) -> dict[str, np.ndarray]:
        return {k: np.concatenate(v, axis=0) for k, v in self.rec.items() if v}

    def stacked_inputs(self):
        """The recorded boundary inputs, concatenated along the time axis."""
        return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *self.rec_inputs)


# ---------------------------------------------------------------------------
# The budget itself (pure NumPy, offline)
# ---------------------------------------------------------------------------

class Budget(NamedTuple):
    terms: dict[str, np.ndarray]      # per-tick, metres (or m/s)
    total: np.ndarray                 # per-tick observed delta
    closure: float                    # max |sum(terms) - total|


def _position_budget(rec: dict, dt: float) -> Budget:
    """Exact six-way split of Δ(p̂_z − p_z^true) per tick."""
    p, v, R = rec["p_stage"], rec["v_stage"], rec["R_stage"]
    pt, vt = rec["p_true"], rec["v_true"]

    # Previous tick's truth; tick 0's predecessor is the seed (robot at rest, t=0).
    pt_prev = np.concatenate([pt[:1], pt[:-1]], axis=0)
    vt_prev = np.concatenate([vt[:1], vt[:-1]], axis=0)

    dR_c = np.einsum("tij,tkj->tik", R[:, 2], R[:, 1])       # ΔR_C = R2 R1ᵀ
    dR_g = np.einsum("tij,tkj->tik", R[:, 3], R[:, 2])       # ΔR_G = R3 R2ᵀ
    rot_c = np.einsum("tij,tj->ti", dR_c, p[:, 1]) - p[:, 1]
    rot_g = np.einsum("tij,tj->ti", dR_g, p[:, 2]) - p[:, 2]

    prop = p[:, 1, 2] - p[:, 0, 2]
    d_true = pt[:, 2] - pt_prev[:, 2]
    carry = (v[:, 0, 2] - vt_prev[:, 2]) * dt

    terms = {
        "PROP_CARRY": carry,
        "PROP_2ND": prop - d_true - carry,
        "CONT_ROT": rot_c[:, 2],
        "CONT_TRANS": p[:, 2, 2] - p[:, 1, 2] - rot_c[:, 2],
        "GRAV_ROT": rot_g[:, 2],
        "GRAV_TRANS": p[:, 3, 2] - p[:, 2, 2] - rot_g[:, 2],
    }
    total = (p[:, 3, 2] - p[:, 0, 2]) - d_true
    closure = float(np.max(np.abs(sum(terms.values()) - total)))
    return Budget(terms, total, closure)


def _velocity_budget(rec: dict, dt: float, g: np.ndarray,
                     lever: np.ndarray | None = None) -> Budget:
    r"""Exact split of Δ(v̂_z − v_z^true) per tick.

    ``PV_SENS`` — everything that is neither attitude error nor a filter update —
    is split further when the IMU lever arm ``r`` (body frame, base-body origin →
    accelerometer site) is known:

        PV_GAMMA   (R^true (Γ₁ ā − ā))_z dt        rotation compensation over dt
        PV_LEVER   (R^true (α×r + ω×(ω×r)))_z dt   the accelerometer is not at the
                                                   body origin the state tracks
        PV_INTEG   the remainder — one-sided sampling of ā at 1/dt against a
                   contact-impulsive true acceleration, plus any model mismatch
    """
    v, R = rec["v_stage"], rec["R_stage"]
    vt, Rt = rec["v_true"], rec["R_true"]
    vt_prev = np.concatenate([vt[:1], vt[:-1]], axis=0)
    Rt_prev = np.concatenate([Rt[:1], Rt[:-1]], axis=0)

    # u = Γ₁(ω dt) ā, recovered exactly from the propagation increment:
    #   v1 = v0 + R0 u dt + g dt  ⟹  u = R0ᵀ (v1 − v0 − g dt) / dt
    dv_prop = v[:, 1] - v[:, 0]
    u = np.einsum("tji,tj->ti", R[:, 0], dv_prop - g * dt) / dt

    est_dv = np.einsum("tij,tj->ti", R[:, 0], u) * dt + g * dt
    true_dv = np.einsum("tij,tj->ti", Rt_prev, u) * dt + g * dt
    d_vtrue = vt[:, 2] - vt_prev[:, 2]

    dR_c = np.einsum("tij,tkj->tik", R[:, 2], R[:, 1])
    dR_g = np.einsum("tij,tkj->tik", R[:, 3], R[:, 2])
    cv_rot = (np.einsum("tij,tj->ti", dR_c, v[:, 1]) - v[:, 1])[:, 2]
    gv_rot = (np.einsum("tij,tj->ti", dR_g, v[:, 2]) - v[:, 2])[:, 2]

    terms = {
        "PV_ATT": est_dv[:, 2] - true_dv[:, 2],
        "CV_CONT_ROT": cv_rot,
        "CV_CONT_TRANS": v[:, 2, 2] - v[:, 1, 2] - cv_rot,
        "CV_GRAV_ROT": gv_rot,
        "CV_GRAV_TRANS": v[:, 3, 2] - v[:, 2, 2] - gv_rot,
    }
    sens = true_dv[:, 2] - d_vtrue
    if lever is None:
        terms["PV_SENS"] = sens
    else:
        a_meas = rec["accel"]                                   # ā at the IMU site, body axes
        gamma = np.einsum("tij,tj->ti", Rt_prev, u - a_meas)[:, 2] * dt
        lev = np.einsum("tij,tj->ti", Rt_prev, lever)[:, 2] * dt
        terms["PV_GAMMA"] = gamma
        terms["PV_LEVER"] = lev
        terms["PV_INTEG"] = sens - gamma - lev

    total = (v[:, 3, 2] - v[:, 0, 2]) - d_vtrue
    closure = float(np.max(np.abs(sum(terms.values()) - total)))
    return Budget(terms, total, closure)


def _lever_acceleration(rec: dict, r_body: np.ndarray, dt: float) -> np.ndarray:
    r"""``α×r + ω×(ω×r)`` in body axes — the specific-force offset of an IMU at ``r``.

    ``ω`` is the TRUE base angular velocity (body frame, `mjOBJ_XBODY`); ``α`` is
    its central difference.  The centripetal term ``ω×(ω×r) = ω(ω·r) − r‖ω‖²`` is
    the sign-definite one: it always points toward the axis, so an IMU mounted off
    the body origin reads a persistent offset while the robot rocks.
    """
    w = rec["w_true"]
    alpha = np.gradient(w, dt, axis=0)
    cross = lambda a, b: np.cross(a, b)                                    # noqa: E731
    return cross(alpha, r_body) + cross(w, cross(w, np.broadcast_to(r_body, w.shape)))


def _height_attribution(pos: Budget, vel: Budget, dt: float) -> dict[str, float]:
    r"""Fold the velocity budget into the position one — a single height ledger.

    ``PROP_CARRY_k = e_v(k−1)·dt`` and ``e_v(k−1) = e_v(0) + Σ_{j<k} Δe_v(j)``, so

        Σ_k PROP_CARRY_k = e_v(0)·T + Σ_j Δe_v(j)·(T − j)·dt

    i.e. every velocity-budget term contributes height in proportion to how long it
    had left to be integrated.  Replacing PROP_CARRY by that expansion gives one
    table in metres whose entries sum to the observed height error.
    """
    T = len(pos.total)
    weight = (T - np.arange(T)) * dt          # remaining integration time per tick
    out = {f"via v: {k}": float((s * weight).sum()) for k, s in vel.terms.items()}
    out.update({k: float(s.sum()) for k, s in pos.terms.items() if k != "PROP_CARRY"})
    return out


def _common_mode(rec: dict, n_contacts: int) -> dict[str, np.ndarray]:
    r"""Height the contact update pushed into ``ker H``.

    The update moves the base by ``Δp`` and each anchor by ``Δd_i``; the orthogonal
    projection onto ``C = {ξ_p = ξ_{d_i} = δ}`` is ``δ = (Δp + Σ Δd_i)/(N+1)``.
    Only the anchor MEAN is recorded, which is all this projection needs.
    """
    p, d = rec["p_stage"], rec["d_stage"]
    dp = p[:, 2, 2] - p[:, 1, 2]
    dd = d[:, 2, 2] - d[:, 1, 2]                      # mean over slots
    return {
        "cm_contact": (dp + n_contacts * dd) / (n_contacts + 1.0),
        "base_step": dp,
        "anchor_step": dd,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _report(name: str, b: Budget, unit: str, duration: float) -> None:
    tot = float(b.total.sum())
    print(f"\n  {name}  (closes to {b.closure:.2e})")
    print(f"    {'term':16s} {'cumulative':>14s} {'rate':>14s} {'share':>9s}")
    denom = abs(tot) if abs(tot) > 1e-12 else 1.0
    for k, series in sorted(b.terms.items(), key=lambda kv: -abs(kv[1].sum())):
        s = float(series.sum())
        print(f"    {k:16s} {s:14.5f} {s / duration:14.5f} {100 * s / denom:8.1f}%")
    print(f"    {'TOTAL':16s} {tot:14.5f} {tot / duration:14.5f}   [{unit}]")


def _edge_latency(truth: np.ndarray, trust: np.ndarray, rising: bool,
                  dt: float, horizon: int = 60) -> np.ndarray:
    r"""Ticks from each truth edge to the matching trust edge, per slot.

    Positive = trust LAGS truth.  Touchdown (rising) and toe-off (falling) are
    reported separately because `ContactTrust` is deliberately asymmetric —
    release is same-tick, entry serves the full dwell (`sensors.py:142-154`) — so
    a single mean would hide the very thing worth seeing.  Edges whose partner
    does not arrive within ``horizon`` ticks are dropped, not clamped.
    """
    out = []
    for s in range(truth.shape[1]):
        t, u = truth[:, s], trust[:, s]
        d_t = np.diff(t) > 0 if rising else np.diff(t) < 0
        d_u = np.diff(u) > 0 if rising else np.diff(u) < 0
        u_idx = np.flatnonzero(d_u)
        for i in np.flatnonzero(d_t):
            later = u_idx[u_idx >= i]
            if later.size and later[0] - i <= horizon:
                out.append(later[0] - i)
    return np.asarray(out) * dt


def _detection_report(rec: dict) -> None:
    """ContactTrust vs MuJoCo's own contact set: agreement and edge timing."""
    truth, trust = rec["contact_true"], rec["contact"]
    if truth.shape != trust.shape:
        raise AssertionError(
            f"contact_true {truth.shape} and contact {trust.shape} disagree; the "
            f"per-tick records are misaligned and every FP/FN and lag below would "
            f"be wrong. Do not silently truncate — fix the recorder.")
    dt_ctl = 1.0                                       # report in ticks, then scale
    fp = float(((trust > 0) & (truth == 0)).mean())
    fn = float(((trust == 0) & (truth > 0)).mean())
    if TracingSensorReader.contact_source == "oracle" and (fp or fn):
        raise AssertionError(
            f"oracle contact source but FP={fp:.4f} FN={fn:.4f}; they must be "
            f"exactly zero because the trust vector IS the truth vector. The "
            f"records are misaligned.")
    print("\n  CONTACT DETECTION — ContactTrust vs MuJoCo contact")
    print(f"    truth duty {truth.mean():.3f}   trust duty {trust.mean():.3f}")
    print(f"    false POSITIVE (trusted, not in contact) {100 * fp:6.2f}% of slot-ticks")
    print(f"    false NEGATIVE (in contact, not trusted) {100 * fn:6.2f}% of slot-ticks")
    for label, rising in (("touchdown", True), ("toe-off", False)):
        lat = _edge_latency(truth, trust, rising, dt_ctl)
        if lat.size:
            print(f"    {label:10s} lag: median {np.median(lat):5.1f} ticks  "
                  f"mean {lat.mean():5.1f}  p90 {np.percentile(lat, 90):5.1f}  "
                  f"(n={lat.size})")
        else:
            print(f"    {label:10s} lag: no matched edges")


def analyse(rec: dict, dt: float, g: np.ndarray, n_contacts: int,
            r_body: np.ndarray | None = None, *, strict: bool = True):
    T = rec["p_stage"].shape[0]
    duration = T * dt
    lever = None if r_body is None else _lever_acceleration(rec, r_body, dt)
    pos = _position_budget(rec, dt)
    vel = _velocity_budget(rec, dt, g, lever)
    cm = _common_mode(rec, n_contacts)

    if strict:
        assert pos.closure < 1e-11, f"position budget does not close: {pos.closure:.3e}"
        assert vel.closure < 1e-11, f"velocity budget does not close: {vel.closure:.3e}"

    e_z = rec["p_stage"][:, 3, 2] - rec["p_true"][:, 2]
    ev_z = rec["v_stage"][:, 3, 2] - rec["v_true"][:, 2]
    print(f"\n=== z budget over {T} estimator ticks ({duration:.1f} s, dt={dt}) ===")
    print(f"  height error   e_z: {e_z[0]:+.4f} -> {e_z[-1]:+.4f} m "
          f"(mean rate {(e_z[-1] - e_z[0]) / duration:+.5f} m/s)")
    print(f"  velocity error e_v: mean {ev_z.mean():+.5f} m/s, final {ev_z[-1]:+.5f} m/s")

    if r_body is not None:
        print(f"  IMU lever arm r (body frame) = "
              f"({r_body[0]:+.4f}, {r_body[1]:+.4f}, {r_body[2]:+.4f}) m")

    _report("POSITION — where the metres come from", pos, "m", duration)
    _report("VELOCITY — why v̂_z is biased", vel, "m/s", duration)

    attrib = _height_attribution(pos, vel, dt)
    resid = e_z[-1] - e_z[0] - sum(attrib.values())
    print(f"\n  HEIGHT LEDGER — velocity terms folded in "
          f"(residual {resid:+.2e} m)")
    print(f"    {'source':24s} {'metres':>12s} {'m/s':>12s} {'share':>9s}")
    denom = abs(e_z[-1] - e_z[0]) or 1.0
    for k, s in sorted(attrib.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k:24s} {s:12.5f} {s / duration:12.5f} {100 * s / denom:8.1f}%")
    print(f"    {'TOTAL':24s} {e_z[-1] - e_z[0]:12.5f} "
          f"{(e_z[-1] - e_z[0]) / duration:12.5f}   [m]")

    print("\n  CONTACT UPDATE — base vs anchor split (the ker H deposit)")
    print(f"    base moved            {cm['base_step'].sum():+.5f} m")
    print(f"    mean anchor moved     {cm['anchor_step'].sum():+.5f} m")
    print(f"    common-mode deposit   {cm['cm_contact'].sum():+.5f} m  "
          f"({cm['cm_contact'].sum() / duration:+.5f} m/s) -- unobservable, never removed")

    nu = rec["contact_innovation"].reshape(T, n_contacts, 3)
    print("\n  CONTACT INNOVATION — the DC the update is reacting to [mm]")
    for i in range(n_contacts):
        m = nu[:, i] * 1e3
        print(f"    slot {i}: mean ({m[:, 0].mean():+7.3f},{m[:, 1].mean():+7.3f},"
              f"{m[:, 2].mean():+7.3f})   rms ({m[:, 0].std():6.3f},{m[:, 1].std():6.3f},"
              f"{m[:, 2].std():6.3f})")

    if "contact_true" in rec:
        _detection_report(rec)

    gap = rec["frame_gap"]
    Np = rec["Np_body"]
    ev = np.linalg.eigvalsh(0.5 * (Np[:, 0] + np.swapaxes(Np[:, 0], -1, -2)))
    aniso = ev[:, 2] / np.maximum(ev[:, 0], 1e-300)
    print("\n  CONTACT MEASUREMENT NOISE — frame and shape (slot 0)")
    print(f"    ‖R Np Rᵀ − Np‖/‖Np‖   mean {gap.mean():.4f}  max {gap.max():.4f}")
    print(f"    anisotropy λmax/λmin  median {np.median(aniso):.1f}  "
          f"p95 {np.percentile(aniso, 95):.1f}")
    print(f"    trace(Np)             mean {np.trace(Np[:, 0], axis1=1, axis2=2).mean():.3e} m²")
    sc = rec["sigma_c"]
    print(f"    Σ_C trace             mean {np.trace(sc[:, 0], axis1=1, axis2=2).mean():.3e}"
          f"  min {np.trace(sc[:, 0], axis1=1, axis2=2).min():.3e}"
          f"  max {np.trace(sc[:, 0], axis1=1, axis2=2).max():.3e}")

    qs = rec["quasi_static"]
    print(f"\n  gravity-leveling gate open on {100 * qs.mean():.2f}% of ticks "
          f"({int(qs.sum())} of {T})")
    return {"pos": pos, "vel": vel, "cm": cm, "e_z": e_z, "ev_z": ev_z}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default="baseline", choices=list(rp.POLICIES))
    ap.add_argument("--ticks", type=int, default=1500, help="control ticks (50 Hz)")
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--contacts-per-foot", type=int, choices=(1, 4), default=1)
    ap.add_argument("--stance-chol", type=float, default=1.0e-4)
    ap.add_argument("--swing-chol", type=float, default=1.0e1)
    ap.add_argument("--anchor-rate-gain", type=float, default=None)
    ap.add_argument("--contact-fk", choices=("measured", "pinned"), default="measured")
    ap.add_argument("--x0", type=float, default=0.0, metavar="M",
                    help="start the robot this far along +x, moving the WORLD ORIGIN "
                         "away from it; isolates the exp(-xi) rotation lever arm")
    ap.add_argument("--terrain", default="flat",
                    choices=("flat", "waves", "stepping_stones", "hard_stepping"),
                    help="run the closed loop over a heightfield instead of the "
                         "plane. The pools span four terrains but every closed-loop "
                         "drift number to date is flat, which is the main gap in any "
                         "contact-timing conclusion")
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument("--khz", action="store_true",
                    help="run the sim + estimator at 1 kHz, the rate ContactNet was "
                         "TRAINED at. Deployment normally runs 200 Hz, which stretches "
                         "the network's H=20 window from 19 ms to 95 ms")
    ap.add_argument("--source", nargs="*", default=None,
                    help="what the policy reads from the estimate. 'truth' makes the "
                         "GAIT identical across arms, so an arm-to-arm drift "
                         "difference is purely estimator quality and not a "
                         "different walk (the policy consumes base_ang_vel and "
                         "projected_gravity, so a changed estimate changes the gait)")
    ap.add_argument("--contact-floor", type=float, default=None,
                    help="override inekf.contact_floor (shipped 1e-4). At the "
                         "shipped value it swamps the analytic stance signal 4:1 "
                         "in decades, so Sigma_C is effectively a binary switch")
    ap.add_argument("--contact-meas-var", type=float, default=0.0,
                    help="isotropic floor folded into Sigma_q before the contact "
                         "update's R (shipped 0.0, i.e. R = J Sigma_q J^T alone)")
    ap.add_argument("--friction", type=float, default=None, metavar="MU",
                    help="OOD: slide friction on feet+floor (pool sampled [0.45, 1.2])")
    ap.add_argument("--payload", type=float, default=0.0, metavar="KG",
                    help="OOD: extra pelvis mass (never randomised in training)")
    ap.add_argument("--push-z", type=float, default=0.0, metavar="N",
                    help="OOD: vertical pelvis forcing at 0.7 Hz (training pushes "
                         "are horizontal-only)")
    ap.add_argument("--contact-source", choices=("trust", "oracle"), default=None,
                    help="'oracle' drives Sigma_C from MuJoCo's own contact set at "
                         "zero latency instead of the Schmitt/dwell ContactTrust — the "
                         "causal test for detection. Either value also records the "
                         "contact truth and normalised load per tick.")
    ap.add_argument("--dwell", type=float, default=None, metavar="S",
                    help="override ContactTrust dwell [s] (default 0.04)")
    ap.add_argument("--enter", type=float, default=None,
                    help="override ContactTrust enter threshold (default 0.35)")
    ap.add_argument("--stay", type=float, default=None,
                    help="override ContactTrust stay threshold (default 0.25)")
    ap.add_argument("--imu-noise", action="store_true",
                    help="enable sensor noise; with --noise-seed this is what makes "
                         "repeat runs independent samples rather than one rollout")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--rotate-R", action="store_true",
                    help="apply R_hat Np R_hat^T before the contact update — the "
                         "body->world rotation the shipped scan path omits")
    ap.add_argument("--contactnet", default=None, metavar="PARAMS.npz",
                    help="drive Sigma_C from a trained ContactNet instead of the "
                         "analytic stance/swing heuristic")
    ap.add_argument("--contactnet-norm", default=None, metavar="NORM.npz")
    ap.add_argument("--mask-k", choices=("none", "rot", "vel", "rotvel"), default="none",
                    help="EXPERIMENT: zero these tangent rows of the CONTACT update's K, "
                         "severing the cross-covariance path from a measurement whose H "
                         "has no rotation and no velocity columns")
    ap.add_argument("--verify", action="store_true",
                    help="assert the tracing step reproduces the shipped step")
    ap.add_argument("--out", default=None, help="write the raw per-tick record to this .npz")
    args = ap.parse_args()

    if args.khz:
        # ContactNet is TRAINED at 1 kHz (`scripts/run_contactnet.py` sets
        # rp.DT=0.001, rp.DECIMATION=20) but `run_estimator` deploys it at 200 Hz
        # (rp.DT=0.005, DECIMATION=4). The H=20 feature window therefore spans
        # 19 ms in training and 95 ms in deployment, and the `v` channel is a
        # causal first difference `/dt`. Must be set before `make_estimated_loop`
        # builds the model, which reads `rp.DT` for the timestep.
        rp.DT, rp.DECIMATION = 0.001, 20
        print("  1 kHz regime: rp.DT=0.001 rp.DECIMATION=20 "
              "(matches the ContactNet training rate)")

    if args.verify and (args.rotate_R or args.mask_k != "none"):
        ap.error("--verify compares against the SHIPPED step, which by definition "
                 "differs under --rotate-R / --mask-k; run it on a clean config")

    # -- swap in the tracing step + runtime + loop, keeping the shipped build path --
    orig_build = me.build_alex_fused_estimator_from_urdf
    holder: dict[str, object] = {}

    def traced_build(*a, **kw):
        fused = orig_build(*a, **kw)
        if args.contact_floor is not None:
            # Applied BEFORE the step is built, because the step closes over `ekf`.
            # `contact_floor` is ADDITIVE (`inEKF/contact.py:91`), and at the shipped
            # 1e-4 it is four orders above the analytic stance signal (Σ = 1e-8), so
            # in stance Σ_C is the floor and nothing else.
            fused = dataclasses.replace(fused, ekf=fused.ekf._replace(
                params=fused.ekf.params._replace(contact_floor=args.contact_floor)))
            print(f"  contact_floor = {args.contact_floor}")
        holder["fused"] = fused
        return dataclasses.replace(
            fused, inekf_step=make_tracing_inekf_step(
                fused.ekf, fused.kinematics, args.mask_k, args.rotate_R))

    me.build_alex_fused_estimator_from_urdf = traced_build
    rest.EstimatorRuntime = TracingRuntime
    rest.ContactNetRuntime = TracingContactNetRuntime
    if args.contact_source is not None:
        # Patched only when asked, so runs that do not use it keep the exact reader
        # (and the exact recorded key set) they had before this existed.
        TracingSensorReader.contact_source = args.contact_source
        rest.SimSensorReader = TracingSensorReader
    TracingLoop.ood_friction = args.friction
    TracingLoop.ood_payload = args.payload
    TracingLoop.ood_push_z = args.push_z
    if args.terrain != "flat":
        from invariant_estimation.sim import terrain as terr
        field = terr.sample_field(args.terrain, args.terrain_seed)
        TracingLoop.terrain_field = field
        # `make_estimated_loop` builds the model itself, so inject the field by
        # wrapping the builder rather than duplicating the build.
        _orig_model = rp.build_sim_model
        rp.build_sim_model = (
            lambda *a, **k: _orig_model(*a, **{**k, "terrain": field}))
        print(f"  terrain: {args.terrain}/seed{args.terrain_seed}")
    rest.EstimatedLoop = (TracingLoop if not args.x0
                          else lambda *a, **k: TracingLoop(*a, x0=args.x0, **k))

    loop = rest.make_estimated_loop(
        args.policy, with_visuals=False,
        sources=rest.DEFAULT_SOURCES if args.source is None else tuple(args.source),
        stance_chol=args.stance_chol, swing_chol=args.swing_chol,
        contact_fk_unfiltered=(args.contact_fk == "measured"),
        contacts_per_foot=args.contacts_per_foot,
        anchor_rate_gain=args.anchor_rate_gain,
        contact_meas_var=args.contact_meas_var,
        contactnet=args.contactnet, contactnet_norm=args.contactnet_norm,
        noise=IMUNoise(seed=args.noise_seed) if args.imu_noise else None,
    )
    fused = holder["fused"]
    for name, val in (("enter", args.enter), ("stay", args.stay), ("dwell", args.dwell)):
        if val is not None:
            setattr(loop.reader.trust, name, float(val))
            print(f"  ContactTrust.{name} = {val}")
    rest.run_headless(loop, args.ticks, cmd=(args.vx, args.vy, args.yaw))

    rec = loop.stacked()
    dt = float(fused.ekf.params.dt)
    g = np.asarray(fused.ekf.params.g)
    out = analyse(rec, dt, g, int(fused.n_contacts), _imu_lever_arm(fused))

    if args.verify:
        _verify(loop, fused, rec)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez_compressed(
            args.out, dt=dt, g=g, n_contacts=int(fused.n_contacts),
            **rec,
            **{f"pos_{k}": v for k, v in out["pos"].terms.items()},
            **{f"vel_{k}": v for k, v in out["vel"].terms.items()},
            **{f"cm_{k}": v for k, v in out["cm"].items()},
        )
        print(f"\n  wrote {args.out}")
    return 0


def _imu_lever_arm(fused) -> np.ndarray:
    r"""``r = ᴮR_Wᵀ(p_IMU − p_B)`` at ``qpos0`` — accelerometer offset in body axes.

    The InEKF integrates the specific force as if it were measured at the body
    frame origin ``B`` (`base_body_site`), but it is measured at the base IMU site.
    A rigid offset ``r`` between them adds ``α×r + ω×(ω×r)`` to what the sensor
    reads.  Evaluated at ``qpos0`` with plain MuJoCo — both sites are welded to the
    same body, so this is a constant of the model, not a configuration-dependent
    quantity (`build_fused_estimator` computes `R_mount` the same way).
    """
    mj = fused.model.mj_model
    d = mujoco.MjData(mj)
    mujoco.mj_kinematics(mj, d)
    sid = np.asarray(fused.model.site_ids)
    p_b = d.site_xpos[sid[fused.base_body_site]]
    p_s = d.site_xpos[sid[fused.base_site]]
    R_b = d.site_xmat[sid[fused.base_body_site]].reshape(3, 3)
    return np.asarray(R_b.T @ (p_s - p_b))


def _verify(loop, fused, rec: dict) -> None:
    """The shipped `inf.make_step`, replayed on the recorded inputs, must agree.

    Re-scans the SHIPPED step over the exact boundary-input stream the traced run
    fed its own step, from the same seed carry, and compares the per-tick base
    positions.  Any divergence between the copied body in `make_tracing_inekf_step`
    and `inEKF/filter.make_step` shows up here as a nonzero difference.
    """
    print("\n  [verify] replaying the SHIPPED InEKF step on the recorded inputs ...")
    shipped = inf.make_step(fused.ekf, fused.kinematics)
    _, out = jax.lax.scan(shipped, loop.seed_carry, loop.stacked_inputs())
    got = np.asarray(out.state.p)
    want = rec["p_stage"][:, 3, :]
    err = float(np.max(np.abs(got - want)))
    print(f"  [verify] max |p_shipped - p_traced| = {err:.3e}")
    assert err < 1e-12, "tracing step has drifted from the shipped step"


if __name__ == "__main__":
    raise SystemExit(main())
