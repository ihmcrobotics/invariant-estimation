"""Evaluation of filter quality.

`consistency` scores a filter against its own innovations (NIS) and needs no
ground truth, so it runs on any robot log. Error-magnitude evaluation against
mocap (NEES, RMSE) lives on the Java side, where the full covariance is
reachable in-process -- see `InvariantEstimatorNeesChecker` in the `alex` repo.
"""
