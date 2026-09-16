r"""Build a `ChannelMap` from a real Alex log's own variable list.

`log_adapter.ChannelMap` deliberately takes exact channel names rather than
guessing them, which leaves one job to do per robot build: work out what those
names actually are. This module does it from the log itself, so the answer is
derived from the machine that produced the data instead of from memory.

**No decoder required.** Names live in ``handshake.yaml``, which is plain YAML;
only the per-tick sample values in ``robotData.bsz`` need the `ihmclog` binary
decoder. So a channel map can be built and checked against a real capture on a
machine that cannot yet decode one.

The trap this exists to remove
------------------------------
The estimator does not consume ``raw_q_*``. `SensorProcessing` publishes every
stage of its chain, and **which stage is last differs per joint and per
channel** on a real Alex log:

* leg joints have an elasticity stage -- ``stiff_q_LEFT_KNEE_Y_sp1``
* ``SPINE_Z`` has no elasticity stage -- ``filt_q_SPINE_Z_sp0``
* no joint has a ``stiff`` *velocity* -- ``filt_qd_..._sp0`` is last for all

Any uniform suffix rule is therefore wrong for some joint on every real log,
and wrong quietly: feeding a less-processed stage puts one filtering step
between this port and the Java estimator, which looks exactly like a filter bug
and is not one. Stage selection lives in
`replay.logsource.resolve_joint_channel` and nowhere else; this module calls it.
"""
from .log_adapter import ChannelMap
from ..replay.logsource import handshake_names, resolve_joint_channel

#: Per-IMU gyro triple, in the x,y,z order `ChannelMap` requires.
GYRO_FORMAT = "gyroscope_{imu}{axis}"
#: Base-IMU specific force, same ordering.
ACCEL_FORMAT = "accelerometer_{imu}{axis}"
#: Binary per-foot trust published by the estimator's own trust logic.
TRUST_FORMAT = "is{contact}FootTrusted"


def _triple(template, **kwargs):
    return tuple(template.format(axis=axis, **kwargs) for axis in "XYZ")


def alex_channel_map(names,
                     joint_names,
                     imu_names,
                     contact_names,
                     *,
                     base_imu,
                     sensor_processing,
                     trust_format=TRUST_FORMAT,
                     probability_format=None):
    """Resolve a `ChannelMap` against ``names``, a log's variable-name set.

    Parameters
    ----------
    names : iterable of str
        Every variable name in the log -- from `handshake_names`, or a live
        reader's handshake.
    joint_names, imu_names, contact_names : sequences
        Must match the session model's and build's own names exactly; the
        adapter re-checks this, but resolving against the wrong set here
        produces a confusing failure one layer further in.
    base_imu : str
        The IMU whose accelerometer is the base specific force.
    sensor_processing : str
        Provenance recorded in the map: which processing chain these names came
        from. Free text, but say something a later reader can act on -- the log
        directory and the robot build, not "default".
    probability_format : str, optional
        Template for a per-contact ``[0, 1]`` probability channel. **Omit it**
        unless such a channel is really logged: on the Alex logs checked so far
        no continuous contact probability is published, and the adapter
        explicitly permits trust and probability to share one binary channel.
        What it forbids is silently thresholding a smoothed signal into a
        binary one, which is why this is opt-in rather than a guess.

    Raises
    ------
    KeyError
        If any joint, IMU, or contact has no matching channel in ``names`` --
        loudly, naming the missing one, rather than returning a partial map
        that fails deep inside a training run.
    """
    names = frozenset(names)
    joint_names = tuple(joint_names)
    imu_names = tuple(imu_names)
    contact_names = tuple(contact_names)
    if not joint_names or not imu_names or not contact_names:
        raise ValueError("joint, IMU and contact names must all be nonempty")
    if base_imu not in imu_names:
        raise ValueError(f"base_imu {base_imu!r} is not among the IMUs {imu_names}")
    if not sensor_processing.strip():
        raise ValueError("record which processing chain these names came from")

    positions = {j: resolve_joint_channel(names, j, "q") for j in joint_names}
    velocities = {j: resolve_joint_channel(names, j, "qd") for j in joint_names}

    gyros = {}
    for imu in imu_names:
        triple = _triple(GYRO_FORMAT, imu=imu)
        missing = [n for n in triple if n not in names]
        if missing:
            raise KeyError(f"log has no gyro channels for IMU {imu!r}: missing {missing}")
        gyros[imu] = triple

    accel = _triple(ACCEL_FORMAT, imu=base_imu)
    missing = [n for n in accel if n not in names]
    if missing:
        raise KeyError(f"log has no accelerometer for base IMU {base_imu!r}: missing {missing}")

    trust = {}
    for contact in contact_names:
        name = trust_format.format(contact=contact)
        if name not in names:
            raise KeyError(f"log has no trust channel {name!r} for contact {contact!r}")
        trust[contact] = name

    if probability_format is None:
        # Shared with trust ON PURPOSE, and permitted: a binary trust signal is already a
        # valid [0,1] probability. The adapter's default contact process noise does not read
        # it anyway; it only matters if the probability heuristic is opted into, and that
        # needs a genuinely continuous channel, not this one re-labelled.
        probability = dict(trust)
    else:
        probability = {}
        for contact in contact_names:
            name = probability_format.format(contact=contact)
            if name not in names:
                raise KeyError(f"log has no probability channel {name!r} for contact {contact!r}")
            probability[contact] = name

    return ChannelMap(
        positions=positions,
        velocities=velocities,
        gyros=gyros,
        accel=accel,
        anchor_trust=trust,
        contact_probability=probability,
        sensor_processing=sensor_processing,
    )


def alex_channel_map_from_log(log_dir, joint_names, imu_names, contact_names, **kwargs):
    """`alex_channel_map` against a log directory's own ``handshake.yaml``.

    The provenance string defaults to that directory, which is the fact a later
    reader most wants: these names were resolved against *this* capture.
    """
    kwargs.setdefault("sensor_processing", f"resolved from {log_dir}/handshake.yaml")
    return alex_channel_map(handshake_names(log_dir), joint_names, imu_names,
                            contact_names, **kwargs)
