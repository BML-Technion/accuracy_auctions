import numpy as np
from sklearn.metrics import accuracy_score
from utils import *


def check_and_plot(x, y, M, v_mod, target_idx, allocation_rule, c=1, plot = True):
    model = allocation_rule(x, y, M, v=v_mod, c=c)
    predictions = model.predict(x)
    alloc = int(predictions[target_idx] == y[target_idx])
    if plot:
        plot_svm_decision_boundary(model, x, y, v=v_mod,
                                   title=f"Decision Boundary @ v={v_mod[target_idx]:.5f}, target = {target_idx}",
                                   target_idx=target_idx)
    return alloc, model

def compute_critical_bid(x, y, M, v, target_idx, allocation_rule, tol=1e-10, max_iter=1000, c=1, plot = True):
    """
    Use binary search to find the minimal v in [0, max(v)] for which the allocation a = 1
    for a given target index. Plot at v=0, v=max, and every `plot_every` steps.
    """
    v_mod = v.copy()
    min_v, max_v = 0.0, v[target_idx]

    # Early check at v = 0
    v_mod[target_idx] = min_v
    alloc_0, model_0 = check_and_plot(x, y, M, v_mod, target_idx, allocation_rule, c=c, plot = plot)
    if alloc_0 == 1:
        return min_v, alloc_0, model_0

    # Early check at v = max
    v_mod[target_idx] = max_v
    alloc_1, model_1 = check_and_plot(x, y, M, v_mod, target_idx, allocation_rule, c=c, plot = plot)
    if alloc_1 == 0:
        return 0.0, alloc_1, model_1

    # Binary search
    low, high = min_v, max_v
    alloc_mid = None  # Initialize in case loop does not run
    model_mid = None
    critical_v = 0
    for i in range(max_iter):
        mid = (low + high) / 2.0
        v_mod[target_idx] = mid
        alloc_mid, model_mid = check_and_plot(x, y, M, v_mod, target_idx, allocation_rule, c=c, plot = plot)

        if alloc_mid == 1:
            high = mid
        else:
            low = mid

        if high - low < tol and alloc_mid == 1:
            critical_v = (low + high) / 2.0
            break

    return critical_v, alloc_mid if alloc_mid is not None else -1, model_mid if model_mid is not None else -1


