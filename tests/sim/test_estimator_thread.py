"""`ThreadedEstimator` queue invariants, against a STUB runtime.

The estimator itself takes ~55 s to build and is exercised elsewhere; what needs testing here is
the concurrency contract, and that is fully determined by the queue. A stub runtime that simply
records what it was handed makes every test in this file run in milliseconds and, more
importantly, makes the assertions EXACT -- "sample 137 arrived once, in order, in a chunk of
exactly `substeps`" is checkable against a recording list and not against a filter output.

The decisive one is `test_no_sample_is_ever_lost`. Everything else in the threaded design is a
performance detail; that one is a correctness invariant, because the filter is a sequential
recursion and a silently dropped sample would show up months later as a phantom tuning problem.
"""

import threading
import time

import pytest

from invariant_estimation.sim.estimator_thread import ThreadedEstimator

SUBSTEPS = 4


class StubRuntime:
    """Records every batch it is handed and returns a trivially identifiable estimate."""

    def __init__(self, substeps=SUBSTEPS, delay=0.0, fail_on=None):
        self.substeps = substeps
        self.batches = []
        self.delay = delay
        self.fail_on = fail_on
        self._lock = threading.Lock()

    def advance(self, batch):
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.batches.append(list(batch))
            n = len(self.batches)
        if self.fail_on is not None and n == self.fail_on:
            raise ValueError("stub runtime exploded")
        return f"est-after-{batch[-1]}"

    @property
    def seen(self):
        return [s for b in self.batches for s in b]


def _drain(te, timeout=10.0):
    """Wait until the worker has consumed everything it can (< substeps left)."""
    t0 = time.time()
    while te.backlog >= te.substeps and time.time() - t0 < timeout:
        time.sleep(0.002)
    time.sleep(0.05)              # let the in-flight chunk publish


# ---------------------------------------------------------------------------
# The invariant the design rests on
# ---------------------------------------------------------------------------

def test_no_sample_is_ever_lost():
    """400 samples in -> every one arrives exactly once, in order, in chunks of exactly substeps."""
    rt = StubRuntime()
    te = ThreadedEstimator(rt, max_backlog_ticks=5)
    te.prime(list(range(SUBSTEPS)), {"i": SUBSTEPS - 1})
    try:
        for i in range(SUBSTEPS, 400):
            te.submit(i, {"i": i})
        _drain(te)
    finally:
        te.stop()

    seen = rt.seen
    # In order, no gaps, no duplicates, starting from the primed chunk.
    assert seen == list(range(len(seen)))
    # Everything except a sub-chunk remainder must have been consumed.
    assert 400 - len(seen) < SUBSTEPS
    # And EVERY chunk was exactly `substeps` long -- a short chunk would retrace XLA.
    assert {len(b) for b in rt.batches} == {SUBSTEPS}


def test_chunks_are_never_variable_length_even_under_a_burst():
    """A slow worker plus a fast producer is exactly when a naive 'catch up' would batch bigger."""
    rt = StubRuntime(delay=0.004)
    te = ThreadedEstimator(rt, max_backlog_ticks=50)
    te.prime(list(range(SUBSTEPS)), {})
    try:
        for i in range(SUBSTEPS, 200):
            te.submit(i, {})
        _drain(te)
    finally:
        te.stop()
    assert {len(b) for b in rt.batches} == {SUBSTEPS}
    assert rt.seen == list(range(len(rt.seen)))


# ---------------------------------------------------------------------------
# Back-pressure
# ---------------------------------------------------------------------------

def test_backpressure_bounds_the_backlog_instead_of_dropping():
    """A worker slower than the producer must SLOW THE PRODUCER, never discard."""
    rt = StubRuntime(delay=0.01)
    te = ThreadedEstimator(rt, max_backlog_ticks=2)      # bound = 8 samples
    te.prime(list(range(SUBSTEPS)), {})
    peak = 0
    try:
        for i in range(SUBSTEPS, 120):
            te.submit(i, {})
            peak = max(peak, te.backlog)
        _drain(te)
    finally:
        te.stop()

    # The bound is enforced on the way out of submit, so the queue may momentarily hold one more.
    assert peak <= te.max_backlog + 1, f"backlog reached {peak}, bound is {te.max_backlog}"
    assert rt.seen == list(range(len(rt.seen)))          # and still nothing lost


def test_backpressure_actually_blocks_the_producer():
    """MUTATION CHECK for the test above: without blocking, 120 submits would return instantly."""
    rt = StubRuntime(delay=0.01)
    te = ThreadedEstimator(rt, max_backlog_ticks=1)
    te.prime(list(range(SUBSTEPS)), {})
    try:
        t0 = time.time()
        for i in range(SUBSTEPS, 60):
            te.submit(i, {})
        elapsed = time.time() - t0
    finally:
        te.stop()
    # ~56 samples / 4 per chunk = 14 chunks * 10 ms = 140 ms of unavoidable worker time.
    assert elapsed > 0.05, f"submit never blocked (took {elapsed:.3f}s); back-pressure is absent"


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------

def test_prime_publishes_synchronously_so_latest_is_never_none():
    rt = StubRuntime()
    te = ThreadedEstimator(rt)
    try:
        pub = te.prime([0, 1, 2, 3], {"tag": "primed"})
        assert te.latest() is pub
        assert pub.est == "est-after-3"
        assert pub.truth == {"tag": "primed"}
        assert pub.seq == SUBSTEPS
    finally:
        te.stop()


def test_prime_rejects_a_wrong_sized_batch():
    te = ThreadedEstimator(StubRuntime())
    with pytest.raises(ValueError, match="exactly 4"):
        te.prime([0, 1, 2], {})
    te.stop()


def test_published_truth_is_paired_with_the_chunk_not_the_present():
    """The estimate must be scored against the truth SAMPLED WITH IT, not the current MjData."""
    rt = StubRuntime(delay=0.005)
    te = ThreadedEstimator(rt, max_backlog_ticks=50)
    te.prime([0, 1, 2, 3], {"i": 3})
    try:
        for i in range(SUBSTEPS, 40):
            te.submit(i, {"i": i})
        _drain(te)
        pub = te.latest()
    finally:
        te.stop()
    # `est-after-N` and the truth tag must refer to the SAME sample, however stale both are.
    assert pub.est == f"est-after-{pub.truth['i']}"
    assert pub.seq == pub.truth["i"] + 1


def test_latest_is_swapped_whole_so_a_reader_cannot_tear():
    """A reader polling concurrently must only ever see self-consistent (est, truth) pairs."""
    rt = StubRuntime(delay=0.001)
    te = ThreadedEstimator(rt, max_backlog_ticks=50)
    te.prime([0, 1, 2, 3], {"i": 3})
    bad = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            p = te.latest()
            if p is not None and p.est != f"est-after-{p.truth['i']}":
                bad.append(p)

    r = threading.Thread(target=poll, daemon=True)
    r.start()
    try:
        for i in range(SUBSTEPS, 300):
            te.submit(i, {"i": i})
        _drain(te)
    finally:
        stop.set()
        r.join(timeout=2.0)
        te.stop()
    assert not bad, f"observed {len(bad)} torn reads"


# ---------------------------------------------------------------------------
# Failure and shutdown
# ---------------------------------------------------------------------------

def test_worker_exception_resurfaces_on_the_sim_thread():
    """A dead estimator must not just freeze the estimate while the robot walks on."""
    rt = StubRuntime(fail_on=2)
    te = ThreadedEstimator(rt, max_backlog_ticks=2)
    te.prime([0, 1, 2, 3], {})
    with pytest.raises(RuntimeError, match="estimator thread died"):
        for i in range(SUBSTEPS, 400):
            te.submit(i, {})
            time.sleep(0.001)
    te.stop()


def test_a_dead_worker_does_not_wedge_a_blocked_producer():
    """The deadlock this design could plausibly have: producer waiting on a backlog nobody drains."""
    # fail_on=2, not 1: batch #1 is the synchronous `prime` call, so #2 is the worker's first.
    rt = StubRuntime(fail_on=2, delay=0.005)
    te = ThreadedEstimator(rt, max_backlog_ticks=1)
    te.prime([0, 1, 2, 3], {})
    done = threading.Event()

    def producer():
        try:
            for i in range(SUBSTEPS, 200):
                te.submit(i, {})
        except RuntimeError:
            pass
        done.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    assert done.wait(timeout=5.0), "the producer never returned -- back-pressure deadlocked"
    te.stop()


def test_a_worker_that_dies_silently_still_releases_the_producer():
    """The failure mode mutation testing exposed: liveness must not depend on `_error` being set.

    A worker killed WITHOUT recording an exception (a swallowed error, a library `sys.exit`)
    leaves `_error` None forever. A producer gated on `_error is None` blocks until the process
    is killed -- and the symptom is a HANG, which is exactly the failure that hides in CI. So the
    wait is gated on "is the thread alive", and this test kills the worker behind the class's back
    to prove it.
    """
    rt = StubRuntime(delay=0.005)
    te = ThreadedEstimator(rt, max_backlog_ticks=1)
    te.prime([0, 1, 2, 3], {})

    # Stop the worker without going through `stop()`, and without setting `_error`.
    te._stop.set()
    te._thread.join(timeout=2.0)
    assert not te._thread.is_alive()
    te._stop.clear()               # ... and hide the evidence, so only liveness can save us

    done = threading.Event()
    err = []

    def producer():
        try:
            for i in range(SUBSTEPS, 200):
                te.submit(i, {})
        except RuntimeError as e:
            err.append(e)
        done.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    assert done.wait(timeout=5.0), "the producer blocked forever on a silently dead worker"
    assert err and "without reporting an error" in str(err[0])
    te.stop()


def test_stop_is_clean_and_idempotent():
    te = ThreadedEstimator(StubRuntime())
    te.stop()                     # before start(): must not raise
    te.prime([0, 1, 2, 3], {})
    te.stop()
    te.stop()                     # twice: must not raise
    assert te._thread is None


def test_context_manager_stops_the_thread():
    rt = StubRuntime()
    with ThreadedEstimator(rt) as te:
        te.prime([0, 1, 2, 3], {})
        for i in range(SUBSTEPS, 40):
            te.submit(i, {})
        _drain(te)
        t = te._thread
    assert t is not None and not t.is_alive()


def test_rejects_a_nonsense_backlog_bound():
    with pytest.raises(ValueError, match="max_backlog_ticks"):
        ThreadedEstimator(StubRuntime(), max_backlog_ticks=0)
