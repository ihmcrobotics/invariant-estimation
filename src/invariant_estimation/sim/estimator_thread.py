"""`ThreadedEstimator` — run the fused estimator off the render thread.

Why this exists -- and why you probably do not need it
-----------------------------------------------------
This was built to fix "a control tick costs ~35 ms against its 20 ms budget, so the viewer runs at
0.6x". **That problem was fixed by something else**: pinning the ONNX session to one non-spinning
thread (`run_policy._ort_session`) took the median tick to ~17 ms and the loop now keeps real time
on CPU without any threading at all.

Measured, this module against the synchronous path: **0.97x headless, 1.03x with a render per
tick** -- i.e. nothing. The 1.90x figure that motivated the design was a micro-benchmark of JAX
execution overlapped against `mj_step`; both do release the GIL, and that number is real, but in
the full loop the sim thread simply has too little work left to overlap once ORT stopped stealing
cores.

It is kept because it is correct, tested and accuracy-neutral at the default backlog, and because a
heavier sim (terrain, more envs) would change the arithmetic. It stays **opt-in, viewer-only and
off by default**: `run_headless` and every test call `rt.advance` directly, which is what
`test_sources_truth_bypasses_the_estimate` pins at `atol=0`. This wraps `EstimatorRuntime` rather
than modifying it, so `estimator_loop.py` remains that untouched synchronous path. A threaded run
is not bit-reproducible -- the estimate the policy reads is a few ticks stale, by design -- so
authoritative numbers still come from the synchronous headless path.

The four things that are easy to get wrong here
-----------------------------------------------
1. **Never drop a sample.** The filter is a sequential recursion over sensor samples; dropping one
   silently changes the estimate and would surface months later as a phantom tuning problem. The
   queue is a plain `deque` with **no `maxlen`** -- a `maxlen` deque discards the *oldest*, which
   is the worst possible choice here. When the backlog exceeds `max_backlog_ticks`, the producer
   BLOCKS until the worker drains it. That degrades to exactly today's sub-real-time behaviour --
   the honest failure mode -- and makes "no sample is ever lost" a hard invariant a test asserts.

2. **Fixed chunks of `substeps`, always.** `EstimatorRuntime._advance` is a `jax.jit(lax.scan)`
   compiled for exactly `substeps` samples. A variable-length "catch-up" chunk retraces XLA: an I7
   violation and an ~11 s stall mid-viewer. The worker therefore consumes in fixed chunks and
   leaves any remainder in the queue.

3. **Score against PAIRED truth.** Each queued item carries the truth snapshot taken when its
   sensors were read. Scoring a stale estimate against the CURRENT `MjData` would inflate every
   published error by the staleness and quietly corrupt the numbers in `RUNNING.md`.

4. **A dead worker must be loud, and must not wedge the producer.** If the worker raises and
   nobody notices, the estimate freezes and the robot keeps walking on it -- silently wrong. The
   exception is captured and re-raised on the sim thread at the next `submit`/`latest`. And
   crucially the back-pressure wait is gated on `_alive()`, not on `_error is None`: a worker that
   dies WITHOUT recording an error would otherwise block the producer forever. Mutation testing
   found that one, and the test that should have caught it hung instead of failing.

`reader.read` deliberately stays on the SIM thread: it mutates the `ContactTrust` state machine
and reads `MjData`, so moving it into the worker would race on both.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = ["ThreadedEstimator", "Published"]


@dataclass(frozen=True)
class Published:
    """One estimator output, handed to the sim thread as a single immutable reference.

    Frozen and swapped whole (never field-by-field) so a reader can never observe a torn object:
    the policy either sees the previous estimate or the next one, never half of each.
    """

    est: Any             # EstimateView
    truth: dict          # the truth snapshot PAIRED with the last sample of this chunk
    seq: int             # number of samples consumed when this was published


class ThreadedEstimator:
    """Runs an `EstimatorRuntime` on its own thread. Opt-in, viewer-only."""

    def __init__(self, runtime, *, max_backlog_ticks: int = 2):
        self.rt = runtime
        self.substeps = int(runtime.substeps)
        if max_backlog_ticks < 1:
            raise ValueError("max_backlog_ticks must be >= 1")
        # Backlog is measured in SAMPLES so it can be compared against the queue directly.
        self.max_backlog = int(max_backlog_ticks) * self.substeps

        self._q: deque = deque()          # NO maxlen -- see docstring
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._latest: Published | None = None
        self._consumed = 0
        self._thread: threading.Thread | None = None
        self._started = False

    # -- lifecycle -----------------------------------------------------------

    def prime(self, batch, truth) -> Published:
        """Run ONE chunk synchronously so `latest()` is never None, then start the worker.

        The policy needs an estimate on the very first control tick, and the sim has already
        primed exactly `substeps` samples by then, so there is nothing to wait for and no reason
        to invent a null estimate.
        """
        if len(batch) != self.substeps:
            raise ValueError(f"prime expects exactly {self.substeps} samples, got {len(batch)}")
        self._latest = Published(self.rt.advance(list(batch)), truth, self.substeps)
        self._consumed = self.substeps
        self.start()
        return self._latest

    def start(self) -> None:
        if self._thread is not None:
            return
        # Daemon so a wedged estimator can never hang process exit; `stop()` is still the
        # intended path and joins properly.
        self._thread = threading.Thread(target=self._run, name="estimator", daemon=True)
        self._started = True
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Idempotent, and safe to call before `start()`."""
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # -- producer side (sim thread) -----------------------------------------

    def submit(self, sample, truth) -> None:
        """Queue one sensor sample + its paired truth. Blocks if the backlog is too deep."""
        self._reraise()
        with self._cv:
            self._q.append((sample, truth))
            self._cv.notify()
            # BACK-PRESSURE, not dropping. Wait until the worker has drained us back under the
            # bound.
            #
            # The liveness condition is `_alive()`, NOT `_error is None`. Mutation testing found
            # the difference: a worker that dies WITHOUT recording an error (a swallowed
            # exception, a `sys.exit` in a library) leaves `_error` None forever, and a producer
            # gated on `_error` then blocks until the heat death of the universe -- the test that
            # should have caught it hung instead of failing. Keying on "is the thread actually
            # running" makes the exit structural: whatever kills the worker, the producer leaves.
            while (len(self._q) > self.max_backlog
                   and not self._stop.is_set()
                   and self._error is None
                   and self._alive()):
                self._cv.wait(timeout=0.1)
        self._reraise()

    def latest(self) -> Published | None:
        """The most recent published estimate. One atomic reference read."""
        self._reraise()
        with self._cv:
            return self._latest

    @property
    def backlog(self) -> int:
        with self._cv:
            return len(self._q)

    def _alive(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def _reraise(self) -> None:
        err = self._error
        if err is not None:
            raise RuntimeError("the estimator thread died") from err
        # A worker can also stop without recording anything. Silence here is the dangerous case:
        # the estimate would simply freeze while the robot kept walking on it.
        if self._started and not self._stop.is_set() and not self._alive():
            raise RuntimeError("the estimator thread stopped without reporting an error")

    # -- worker --------------------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                with self._cv:
                    while len(self._q) < self.substeps and not self._stop.is_set():
                        self._cv.wait(timeout=0.1)
                    if self._stop.is_set():
                        return
                    # Exactly `substeps`, never more: a variable-length chunk retraces XLA.
                    chunk = [self._q.popleft() for _ in range(self.substeps)]
                    self._cv.notify_all()          # a blocked producer may now proceed
                batch = [c[0] for c in chunk]
                truth = chunk[-1][1]               # paired with the sample `_view` reads
                # OUTSIDE the lock: this is the expensive call, and holding the lock across it
                # would serialise the producer against it and undo the whole point.
                est = self.rt.advance(batch)
                with self._cv:
                    self._consumed += self.substeps
                    self._latest = Published(est, truth, self._consumed)
        except BaseException as exc:               # noqa: BLE001 - resurfaced on the sim thread
            self._error = exc
        finally:
            # Whatever happened, never leave a producer blocked on back-pressure.
            with self._cv:
                self._cv.notify_all()
