"""`Ghost` — a translucent second robot drawn at the ESTIMATED state.

The error tables tell you the estimate is off by 0.81 deg of tilt and 2.2 m of position over a
30 s walk. They cannot tell you *where* it goes wrong, which is what makes an estimator bug
obvious. The ghost draws the estimate as a second, translucent robot on top of the real one: when
the estimate is good the ghost hides inside the real robot, and any disagreement reads as a
separating shadow. The known missing-touchdown-reseed drift, for instance, shows up as the ghost
steadily sinking through the floor.

Physics-free, and structurally so
---------------------------------
The ghost owns a second `MjData` on the SAME `MjModel` and is never `mj_step`ped -- only
`mj_kinematics`, which is position-level: it reads `qpos` and writes `xpos`/`xquat`/`geom_xpos`
and touches nothing the integrator reads. It cannot perturb the real `d` even in principle, which
is the property `tests/sim/test_ghost.py` pins at `atol=0`. (Same spare-`MjData` idiom as
`run_policy.foot_rest_height`.)

What the ghost is allowed to know
---------------------------------
Exactly what the estimator knows, no more:

* base pose from `est.p` / `est.R`,
* the filtered joints from `est.q`,
* the remaining unfiltered joints copied from the real `qpos` -- legitimate, because on hardware
  those come straight off the encoders and are not filter outputs.

Modes
-----
``full``     the whole estimated pose. The honest view, and the default: it sinks underground as
             the position estimate drifts, which is the thing worth seeing.
``attitude`` identical but the pelvis is pinned at the TRUE position, isolating orientation error
             from the height drift so the ghost stays on screen.
``off``      draws nothing.

Cost, measured: `mj_kinematics` 0.002 ms + `mjv_addGeoms` 0.002 ms per frame, i.e. free against a
20 ms control budget.
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
        # qpos addresses of the joints the estimator actually filters. `filtered_slots` indexes the
        # POLICY's joint order (`run_estimator.EstimatedLoop.filtered_slots`), so it has to go
        # through QADR to become a qpos address.
        self.filtered_qadr = np.asarray(maps["QADR"])[np.asarray(filtered_slots)]
        self.rgba = np.asarray(rgba, dtype=np.float32)
        self.offset = float(offset)
        self.mode = mode

        # A DEDICATED MjvOption, not the viewer's. The dynamic pass renders sites too, and this
        # model has 20 of them (the estimator's 8 IMU sites, the soles, ...) -- appending a second
        # copy of all of them is pure visual noise on top of an already-overlaid robot. Collision
        # geoms are group 3 and hidden by the default `geomgroup`, so they need no handling.
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0
        # `mjv_addGeoms` requires a perturb object; one reused instance, never mutated.
        self._pert = mujoco.MjvPerturb()

    # -- state ---------------------------------------------------------------

    @property
    def on(self) -> bool:
        return self.mode != "off"

    def cycle(self) -> str:
        """off -> full -> attitude -> off. Returns the new mode."""
        self.mode = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]
        return self.mode

    # -- per-frame -----------------------------------------------------------

    def update(self, est, d_true) -> None:
        """Place the ghost at `est`, then run position-level FK. No-op when off."""
        if not self.on:
            return
        q = self.data.qpos
        # Start from truth so the 20 UNFILTERED joints are populated; the base pose and the 9
        # filtered joints are then overwritten from the estimate below.
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
        # Lateral displacement for side-by-side viewing. Applied last so it cannot interact with
        # the attitude-mode position pin above.
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
