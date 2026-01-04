import warnings 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler 
import numpy as np
import pandas as pd
from randomness import *
from simulation_exact import *
from utils import *
from data_analysis import *
from sklearn.linear_model import LogisticRegression
from scipy.stats import beta


def train_soft_svm(x, y, v=None, c=1.0, loss = 'hinge', fit_intercept=True):
    """
    Train a linear SVM with:
      - bias fixed to zero (fit_intercept = False)
      - hinge loss averaged over sum(v) instead of N
    """
    
    # If no sample weights provided, use all weights = 1
    if v is None:
        v = np.ones(len(y))

    if loss == 'hinge':
        model = LinearSVC(
            C=c,
            loss='hinge',
            fit_intercept=fit_intercept,   # <--- forces b = 0
            dual=True,
            random_state=0,
        )
    elif loss == 'log':
        model = LogisticRegression(
        penalty='l2',
        C=c,
        solver='liblinear',
        max_iter=2000, 
        fit_intercept=fit_intercept,
        dual=False,
        random_state=0 
        )
    elif loss == 'squared_hinge':
        model = LinearSVC(
            C=c,
            loss='squared_hinge',
            fit_intercept=fit_intercept,   # <--- forces b = 0
            dual=True,
            random_state=0
        )


    model.fit(x, y, sample_weight=v)
    return model


def get_relevant_indices(svm_model, x, y, v, c, sigma_loss = 1.0, k=None, k_coef=1.0, tol = 0.05):
    # unpack model
    w = svm_model.coef_.ravel()
    b = svm_model.intercept_

    # decision values and margins
    decision_values = x @ w + b
    margin = y * decision_values

    # correctly classified *on-margin* support vectors: 0 < y*f(x) <= 1
    support_vector_mask = (margin > 0) & (margin <= 1)
    support_vector_indices = np.where(support_vector_mask)[0]

    # extract V for those points
    V_critical = v[support_vector_indices]

    # lambda = 1/C
    lam = 1.0 / c

    # compute k if not provided: max L2 norm of x
    if k is None:
        # print(f'k is none, computing from data')
        k = np.linalg.norm(x, axis=1).max()


    # compute beta values (vectorized)
    beta_values = ((k*k_coef)**2 * V_critical * sigma_loss) / (2 * lam)


    # select: margin < beta   (vectorized)
    selected_mask = margin[support_vector_indices] <  beta_values + tol
    relevant_indices = support_vector_indices[selected_mask]
    
    #print("margin, beta", support_vector_indices , margin[support_vector_indices] ,  beta_values[0])
    return relevant_indices, support_vector_indices


def get_throw(svm_model, x, y, v, c, sigma_loss = 1.0, k=None, k_coef=1.0, tol = 0.1):
    # unpack model
    w = svm_model.coef_.ravel()
    b = svm_model.intercept_

    # compute margins: y * f(x)
    decision_values = x @ w + b
    margin = y * decision_values

    # points outside the margin: y*f(x) > 1
    outside_margin_mask = margin > 1 
    outside_margin_indices = np.where(outside_margin_mask)[0]
   
    # extract V for those points
    V_critical_outside = v[outside_margin_indices]
    
    # lambda = 1/C
    lam = 1.0 / c

    # compute k if not provided: max L2 norm of x
    if k is None:
        k = np.linalg.norm(x, axis=1).max()

    # compute beta values (vectorized)
    beta_values_outside = ((k*k_coef)**2 * (V_critical_outside) * sigma_loss) / (2 * lam)

    # final outside selection: margin > 1 + beta
    selected_mask = margin[outside_margin_indices] > (1 + beta_values_outside) + tol
    throw = outside_margin_indices[selected_mask]
    #print("margin, beta", outside_margin_indices , margin[outside_margin_indices] ,  beta_values_outside[0])

    return throw


def update_indices(relevant_indices, removed_indices):
    if len(removed_indices) == 0:
        return relevant_indices
    
    removed = sorted(removed_indices)
    updated = []

    for idx in relevant_indices:
        shift = sum(r < idx for r in removed)
        updated.append(idx - shift)

    return updated


def run_exact(x,y,v,c,use_loss, sigma_loss = 1.0, plot = False, is_throw = False, k=None, k_coef=1.0, fit_intercept=False):
    M = np.sum(v)
    svm_model = train_soft_svm(x, y, v, c = (c/M), loss = use_loss, fit_intercept=fit_intercept)
    relevant_indices, support_idx = get_relevant_indices(svm_model, x, y, v, (c/M), sigma_loss =  sigma_loss, k=None, k_coef=k_coef)
    if is_throw:
        throw = get_throw(svm_model, x, y, v, (c/M), sigma_loss =  sigma_loss, k=None, k_coef=k_coef)
    else:
        throw = []

    records = []
    n = len(v)
    throw_set = set(throw)
    relevant_set = set(relevant_indices)
    support_set = set(support_idx)

    mask = np.ones(n, dtype=bool)
    mask[throw] = False

    new_x, new_y, new_v = x[mask], y[mask], v[mask]

    updated_relevant_indices = update_indices(relevant_indices, throw)
    real_to_updated = dict(zip(relevant_indices, updated_relevant_indices))

    # updated_relevant_indices = update_indices(support_idx, throw)
    # real_to_updated = dict(zip(support_idx, updated_relevant_indices))

    for target_idx in range(n):
        is_relevant = target_idx in relevant_set
        is_support = target_idx in support_set
        is_thrown = target_idx in throw_set

        if is_relevant: # or is_support
            updated_idx = real_to_updated[target_idx]
            critical_v, alloc, _ = compute_critical_bid(
                new_x, new_y, new_v, updated_idx,
                train_soft_svm,
                loss=use_loss,
                plot=plot,
                c=(c / M),
                fit_intercept=fit_intercept
            )
        else:
            alloc = int(svm_model.predict(x[target_idx].reshape(1, -1)) == y[target_idx])
            critical_v = 0

        records.append({
            "agent": target_idx,
            "allocation": alloc,
            "true_v": v[target_idx],
            "critical_v": critical_v,
            "welfare": v[target_idx] * alloc,
            "utility": v[target_idx] * alloc - critical_v,
            "is_relevant": int(is_relevant),
            "support": int(is_support),
            "is_throw": int(is_thrown),
        })

    df_exact = pd.DataFrame(records)
    return df_exact, svm_model


if __name__ == '__main__':
    pass






