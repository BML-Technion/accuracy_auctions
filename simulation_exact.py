from utils import *


def check_and_plot(x, y, v_mod, target_idx, allocation_rule, 
                   c=1, loss = 'hinge', plot = True, fit_intercept=True):
    model = allocation_rule(x, y, v=v_mod, c=c, loss = loss, fit_intercept=fit_intercept)
    predictions = model.predict(x)
    alloc = int(predictions[target_idx] == y[target_idx])
    if plot:
        plot_svm_decision_boundary(model, x, y, v=v_mod,
                                   title=f"Decision Boundary @ v={v_mod[target_idx]:.5f}, target = {target_idx}",
                                   target_idx=target_idx)
    return alloc, model



def compute_critical_bid(x, y, v, target_idx, allocation_rule, loss = 'hinge', tol=1e-10, max_iter=1000, c=1,
                          plot = True, fit_intercept=True):
    """
    Use binary search to find the minimal v in [0, max(v)] for which the allocation a = 1
    for a given target index. Plot at v=0, v=max, and every `plot_every` steps.
    """
    v_coef = 1.5
    v_mod = v.copy()
    min_v, max_v = 0.0, v_coef * v[target_idx] 
    
    # Early check at v = 0
    v_mod[target_idx] = min_v
    alloc_0, model_0 = check_and_plot(x, y, v_mod, target_idx, allocation_rule, loss = 'hinge', c=c, 
                                      plot = plot, fit_intercept=fit_intercept)
    if alloc_0 == 1:
        return min_v, alloc_0 , model_0
    
    #no need for early check at v_max, all points entering here have alloc = 1

    # Binary search
    low, high = min_v, max_v
    alloc_mid = 0  # Initialize in case loop does not run
    model_mid = None
    critical_v = None
    for i in range(max_iter):
        mid = (low + high) / 2.0
        v_mod[target_idx] = mid
        alloc_mid, model_mid = check_and_plot(x, y, v_mod, target_idx, allocation_rule, c=c,
                                              loss = 'hinge', plot = plot, fit_intercept=fit_intercept)

        if alloc_mid == 1:
            high = mid
        else:
            low = mid

        if high - low < tol and alloc_mid == 1:
            critical_v = (low + high) / 2.0
            break
    
    if i == max_iter - 1:
        print("Warning: reached max iterations")
    if alloc_mid == 0 or critical_v is None:
        print("Warning: alloc_mid is still 0 or critical_v is none")
        print(f"alloc mid =", alloc_mid)
        print(f"high, low", high, low)
        print(f"critical_v =", critical_v)
        print(f"iter =", i)
    return critical_v, alloc_mid, model_mid 
