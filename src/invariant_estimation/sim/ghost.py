"""`Ghost` — a translucent second robot drawn at the ESTIMATED state.

The error tables say the estimate is off by 0.81 deg of tilt and 2.2 m of position over a 30 s
walk; they cannot say *where* it goes wrong. The ghost draws the estimate on top of the real
robot, so any disagreement reads as a separating shadow. The vertical sink is what it shows most
plainly -- reduced but not gone: the causal early release cut it 2.24x, -56.7 -> -25.3 cm over
30 s at vx = 0.6 (`--early-release`, RUNNING.md).

**Physics-free, structurally.** A second `MjData` on the SAME `MjModel`, never `mj_step`ped --
only `mj_kinematics`, which is position-level: reads `qpos`, writes `xpos`/`xquat`/`geom_xpos`,
touches nothing the integrator reads. It cannot perturb the real `d` even in principle, which is
what `tests/sim/test_ghost.py` pins at `atol=0`.

It knows exactly what the estimator knows: base pose from `est.p`/`est.R`, the filtered joints
from `est.q`, and the remaining unfiltered joints copied from the real `qpos` -- legitimate,
because on hardware those come straight off the encoders.

Modes: ``full`` the whole estimated pose (the default; it sinks underground as the position
estimate drifts, which is the thing worth seeing), ``attitude`` the same with the pelvis pinned
at the TRUE position so orientation error is isolated from the height drift, ``off`` nothing.

Cost, measured: 0.083 ms/frame (update 0.027 + draw 0.057), free against a 20 ms control budget.
"""

from __future__ import annotations

import mujoco
import numpy as np

__all__ = ["Ghost"]


class Ghost:
    """A translucent second robot drawn at the estimated state. Physics-free."""

    MODES = ("off", "full", "attitude")

    def __init__(self, m, maps, filtered_slots, *, rgba=(1.0, 0.45, 0.1, 0.35), offset=0.0,
                 mode="off"):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.m = m
        self.data = mujoco.MjData(m)
        self.base_bid = int(maps["BASE_BID"])
        # `filtered_slots` indexes the POLICY's joint order
        # (`run_estimator.EstimatedLoop.filtered_slots`), so it goes through QADR to become a
        # qpos address.
        self.filtered_qadr = np.asarray(maps["QADR"])[np.asarray(filtered_slots)]
        self.rgba = np.asarray(rgba, dtype=np.float32)
        self.offset = float(offset)
        self.mode = mode

        # A DEDICATED MjvOption, not the viewer's: the dynamic pass renders sites too, and a
        # second copy of this model's 20 sites is pure noise. (Collision geoms are group 3 and
        # already hidden by the default `geomgroup`.)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0
        # `mjv_addGeoms` requires a perturb object; one reused instance, never mutated.
        self._pert = mujoco.MjvPerturb()

    @property
    def on(self) -> bool:
        return self.mode != "off"

    def cycle(self) -> str:
        """off -> full -> attitude -> off. Returns the new mode."""
        self.mode = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]
        return self.mode

    def update(self, est, d_true) -> None:
        """Place the ghost at `est`, then run position-level FK. No-op when off."""
        if not self.on:
            return
        q = self.data.qpos
        # Start from truth so the 20 UNFILTERED joints are populated; base pose and the 9
        # filtered joints are overwritten from the estimate below.
        q[:] = d_true.qpos
        if self.mode == "attitude":
            # Pin position at truth; orientation still follows the estimate.
            q[0:3] = d_true.xpos[self.base_bid]
        else:
            q[0:3] = est.p
        # `qpos[3:7]` is the free joint's world-from-body quaternion, the same convention as
        # `est.R`. mju_mat2Quat wants a flat, contiguous 9-vector.
        mujoco.mju_mat2Quat(q[3:7], np.ascontiguousarray(est.R, dtype=np.float64).reshape(9))
        q[self.filtered_qadr] = est.q
        # Side-by-side offset, applied last so it cannot interact with the attitude-mode pin.
        q[1] += self.offset
        mujoco.mj_kinematics(self.m, self.data)

    def draw(self, scn) -> int:
        """Append the ghost's geoms to `scn` and tint them. Returns how many were added."""
        if not self.on:
            return 0
        n0 = scn.ngeom
        mujoco.mjv_addGeoms(self.m, self.data, self.opt, self._pert,
                            mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        for i in range(n0, scn.ngeom):
            scn.geoms[i].rgba[:] = self.rgba
        return scn.ngeom - n0
