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
            random_state=0
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


def generate_custom_distribution(
    n,
    beta1, beta2, 
    sigma=0.1,
    a1=-0.5, b1=0.5,
    a2=-0.5, b2=0.5,
):
    """
    Generates a dataset where:
    - x[:,0] is Gaussian with means beta1 and beta2 (low variance sigma)
    - x[:,1] is uniform in [a1,b1] for class +1 and [a2,b2] for class -1
    """
    np.random.seed(0)

    n1 = n // 2
    n2 = n - n1

    # First dimension: Gaussian with beta1 or beta2
    x1_dim1 = np.random.normal(beta1, sigma, size=n1)
    x2_dim1 = np.random.normal(beta2, sigma, size=n2)

    # First dimension: Gaussian with beta1 or beta2
    x1_dim1 = np.random.normal(beta1, sigma, size=n1)
    x2_dim1 = np.random.normal(beta2, sigma, size=n2)

    # Second dimension: Uniform within intervals
    x1_dim2 = np.random.uniform(beta1, beta2/4, size=n1)
    x2_dim2 = np.random.uniform(beta1/4, beta2, size=n2)

    # Stack them into (n,2)
    X1 = np.column_stack([x1_dim1, x1_dim2])
    X2 = np.column_stack([x2_dim1, x2_dim2])

    x = np.vstack([X1, X2])
    y = np.array([1]*n1 + [-1]*n2)
    v = np.ones(n, dtype=int)

    return x, y, v


def generate_data_centered(n):
    # Generate synthetic data
    x, y = make_blobs(n_samples=n, centers=2, random_state=0, cluster_std=2.0)
    y = np.where(y == 0, -1, y)
    v = np.abs(x[:, 0]) * 10 + np.random.normal(0, 0.1, n)

    # --- New Step: Center the data ---
    scaler = StandardScaler(with_std=False) # Only subtracts the mean, doesn't scale std
    x_centered = scaler.fit_transform(x)
    # ---------------------------------

    return x_centered, y, v


def generate_data_centered_1(n):
    if n < 4:
        raise ValueError("n must be at least 4.")

    # --- Determine counts ---
    n_lines = max(2, 49 * n // 50)          # 20% for vertical lines
    n_gauss = n - n_lines             # rest for Gaussians
    
    n_line_side = n_lines // 2
    n_gauss_side = n_gauss // 2

    # --- Vertical lines ---
    y_vals_pos = np.linspace(-1, 1, n_line_side)
    y_vals_neg = np.linspace(-1, 1, n_line_side)

    beta_2 = 1/(4*n)
    line_pos = np.column_stack([beta_2 * np.ones(n_line_side), y_vals_pos])
    line_neg = np.column_stack([-beta_2 * np.ones(n_line_side), y_vals_neg])

    X_lines = np.vstack([line_pos, line_neg])
    y_lines = np.array([1]*n_line_side + [-1]*n_line_side)

    # --- Gaussian clusters ---
    np.random.seed(0)
    gauss_pos = np.column_stack([
        np.random.normal( 0.5, 0.2, n_gauss_side),
        np.random.normal( 0, 0.2, n_gauss_side)
    ])
    gauss_neg = np.column_stack([
        np.random.normal(-0.5, 0.2, n_gauss_side),
        np.random.normal( 0, 0.2, n_gauss_side)
    ])

    X_gauss = np.vstack([gauss_pos, gauss_neg])
    y_gauss = np.array([-1]*n_gauss_side + [1]*n_gauss_side)

    # --- Combine ---
    X = np.vstack([X_lines, X_gauss])
    y = np.concatenate([y_lines, y_gauss])

    # --- Compute v (same rule as your function) ---
    v = 2* np.ones(len(y)) # np.abs(X[:, 0]) * 10 + np.random.normal(0, 0.1, len(X))

    # --- Center the data ---
    scaler = StandardScaler(with_std=False)  # subtract mean only
    X_centered = scaler.fit_transform(X)

    return X_centered, y, v


def update_indices(relevant_indices, removed_indices):
    removed = sorted(removed_indices)
    updated = []

    for idx in relevant_indices:
        shift = sum(r < idx for r in removed)
        updated.append(idx - shift)

    return updated


def get_outside_inside_relevant(svm_model, x, y, v, c, sigma_loss = 1.0, k=None, k_coef=1.0):
    # unpack model
    w = svm_model.coef_.ravel()
    b = svm_model.intercept_

    # compute margins: y * f(x)
    decision_values = x @ w + b
    margin = y * decision_values

    # points outside the margin: y*f(x) > 1
    outside_margin_mask = margin > 1
    outside_margin_indices = np.where(outside_margin_mask)[0]

    # correctly classified *on-margin* support vectors: 0 < y*f(x) <= 1
    support_vector_mask = (margin > 0) & (margin <= 1)
    support_vector_indices = np.where(support_vector_mask)[0]

    # extract V for those points
    V_critical_outside = v[outside_margin_indices]
    V_critical_inside = v[support_vector_indices]
    
    # lambda = 1/C
    lam = 1.0 / c

    # compute k if not provided: max L2 norm of x
    if k is None:
        k = np.linalg.norm(x, axis=1).max()

    # compute beta values (vectorized)
    beta_values_outside = ((k*k_coef)**2 * V_critical_outside * sigma_loss) / (2 * lam)
    beta_values_inside = ((k*k_coef)**2 * V_critical_inside * sigma_loss) / (2 * lam)

    # final outside selection: margin > 1 + beta
    selected_mask = margin[outside_margin_indices] > (1+ beta_values_outside) #- tol)
    throw = outside_margin_indices[selected_mask]

    # final inside: margin < beta   (vectorized)
    selected_mask = margin[support_vector_indices] < beta_values_inside
    relevant_indices = support_vector_indices[selected_mask]

    return relevant_indices,  throw


def get_relevant_indices(svm_model, x, y, v, c, sigma_loss = 1.0, k=None, k_coef=1.0):
    # unpack model
    w = svm_model.coef_.ravel()
    b = svm_model.intercept_

    if b != 0:
        print("THE INTERCEPT IS NOT 0")

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
        print(f'k is none, computing from data')
        k = np.linalg.norm(x, axis=1).max()


    # compute beta values (vectorized)
    beta_values = ((k*k_coef)**2 * V_critical * sigma_loss) / (2 * lam)


    # select: margin < beta   (vectorized)
    selected_mask = margin[support_vector_indices] < beta_values
    relevant_indices = support_vector_indices[selected_mask]

    return relevant_indices


def main_exact(x, y, v, relevant_indcies, plot = False, c = 1.0, svm = train_soft_svm, loss = 'hinge', fit_intercept=True):
    records = []
    for target_idx in relevant_indcies:
        critical_v, alloc, _ = compute_critical_bid(x, y, v, target_idx, svm, loss = loss, 
                                                    plot = plot, c=c, fit_intercept=fit_intercept)
        if alloc != 1:
            print(f"Warning: relevant target_idx {target_idx} has allocation {alloc} at critical bid computation.")
        records.append({
                "agent": target_idx,
                "allocation": alloc,
                "true_v": v[target_idx],
                "critical_v": critical_v,
                'welfare':  v[target_idx] *  alloc,
                'utility': v[target_idx] * alloc - critical_v ,
                'is_relevant': 1
            })

    df = pd.DataFrame(records)

    
    return df


def models_equivalent(m1, m2, tol=1e-2):
    same_coef = np.allclose(m1.coef_, m2.coef_, atol=tol)
    #same_intercept = np.allclose(m1.intercept_, m2.intercept_, atol=tol)
    #print(m1.coef_, m2.coef_,)
    return same_coef #and same_intercept



def run_exact(x,y,v,c,use_loss, sigma_loss = 1.0, plot = False, is_throw = True, k=None, k_coef=1.0, fit_intercept=False):
    M = np.sum(v)
    svm_model = train_soft_svm(x, y, v, c = (c/M), loss = use_loss, fit_intercept=fit_intercept)
    # accuracy = svm_model.score(x, y)
    # plot_svm_decision_boundary(svm_model, x, y, v, target_idx=None, title = "Original model with truthful weights")

    relevant_indcies, throw= get_outside_inside_relevant(svm_model, x, y, v, c/M, sigma_loss = sigma_loss, k=k, k_coef=k_coef)
    if len(relevant_indcies) == 0:
        df_exact = pd.DataFrame(columns=[
            "agent", "allocation", "true_v", "critical_v", "welfare", "utility", "is_relevant"
        ])
        df_exact = fill_in_df(df_exact, relevant_indcies, svm_model, x, y, v)
        return df_exact , svm_model
    
    if is_throw and len(throw) > 0:
        mask = np.ones_like(v, dtype=bool)
        mask[throw] = False
        new_x = x[mask]
        new_y = y[mask]
        new_v = v[mask]

        updated_relevant_indcies = update_indices(relevant_indcies, throw)
        df_exact = main_exact(new_x, new_y, new_v, updated_relevant_indcies, plot = plot, c = (c/M), svm = train_soft_svm, 
                              loss = use_loss, fit_intercept=fit_intercept)

    else:
        df_exact = main_exact(x, y, v, relevant_indcies, plot = plot, c = (c/M), svm = train_soft_svm, 
                              loss = use_loss, fit_intercept=fit_intercept)
    
    df_exact = fill_in_df(df_exact, relevant_indcies, svm_model, x, y, v)

    return df_exact, svm_model


def fill_in_df(df_exact, relevant_indcies, svm_model, x, y, v):
    for idx in range(len(v)):
        if idx not in relevant_indcies:
            alloc = int(svm_model.predict(x[idx].reshape(1, -1)) == y[idx])
            df_exact = pd.concat([df_exact, pd.DataFrame([{
                "agent": idx,
                "allocation": alloc,
                "true_v": v[idx],
                "critical_v": 0.0,
                'welfare':  v[idx] * alloc,
                'utility': v[idx] * alloc,
                'is_relevant': 0
            }])], ignore_index=True)
    return df_exact


if __name__ == '__main__':
    pass






