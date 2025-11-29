from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler # Import the scaler
import numpy as np
import pandas as pd
from randomness import *
from simulation_exact import *
from utils import *
from data_analysis import *


def compute_beta(x, v_critical, c, M, k=None):
    # convert to arrays
    v = np.asarray(v_critical)
    
    # lambda = 1/c
    lam = 1.0 / c

    if not k:
        # convert to arrays
        X = np.asarray(x)

        # compute norms of each x
        norms = np.linalg.norm(X, axis=1)

        # k = maximum norm (ensures ||x|| < k for all x)
        k = np.max(norms)

    # compute beta for each x
    beta = (k**2 * v) / (2 * lam )#* M)

    return beta, k


def check_equivalence(X, y, C, v, model1, model2, tol=1e-6):
    """
    Check if (C, v) is equivalent to (C', v') where
    v' = v / sum(v) and C' = C * sum(v).
    """
    # Normalized weights and scaled C
    v_prime = v / v.sum()
    C_prime = C * v.sum()

    # # Train first model
    # model1 = SVC(kernel="linear", C=C, random_state=0)
    # model1.fit(X, y, sample_weight=v)

    # # Train second model
    # model2 = SVC(kernel="linear", C=C_prime, random_state=0)
    # model2.fit(X, y, sample_weight=v_prime)

    # Extract parameters
    w1, b1 = model1.coef_.ravel(), model1.intercept_[0]
    w2, b2 = model2.coef_.ravel(), model2.intercept_[0]

    # Compare
    same_w = np.allclose(w1, w2, atol=tol)
    same_b = np.allclose(b1, b2, atol=tol)

    # Compare support vectors
    sv_idx1, sv_idx2 = model1.support_, model2.support_
    same_sv_idx = np.array_equal(sv_idx1, sv_idx2)

    # Compare dual coefficients (alphas)
    alpha1, alpha2 = model1.dual_coef_, model2.dual_coef_
    same_alpha = np.allclose(alpha1, alpha2, atol=tol)

    return same_w and same_b and same_sv_idx and same_alpha, {
        "same_w": same_w,
        "same_b": same_b,
        "same_sv_idx": same_sv_idx,
        "same_alpha": same_alpha,
        "diff_w_norm": np.linalg.norm(w1 - w2),
        "diff_b": abs(b1 - b2),
        "diff_alpha": np.linalg.norm(alpha1 - alpha2),
        "support_idx_1": sv_idx1,
        "support_idx_2": sv_idx2,
        "num_support_1": len(sv_idx1),
        "num_support_2": len(sv_idx2),
        "C_prime": C_prime,
        "v_prime_sum": v_prime.sum()
    }


def train_soft_svm(x, y, M, v=None, c=1.0):
    """
    Train a linear SVM with:
      - bias fixed to zero (fit_intercept = False)
      - hinge loss averaged over sum(v) instead of N
    """
    
    # If no sample weights provided, use all weights = 1
    if v is None:
        v = np.ones(len(y))

    # Scale C so that the objective uses sum(v)
    # Scikit-learn’s LinearSVC averages hinge loss as (1/N) * sum(max(0, 1 - y*w·x))
    # We want: (1/sum(v)) * sum(v_i * hinge_loss_i)
    C_scaled = c / M

    model = LinearSVC(
        C=C_scaled,
        loss='hinge',
        fit_intercept=False,   # <--- forces b = 0
        dual=True,
        random_state=0
    )

    model.fit(x, y, sample_weight=v)

    return model


# Generate or load data
def generate_data(n):
    # Generate synthetic data
    x, y = make_blobs(n_samples=n, centers=2, random_state=0, cluster_std=1.5)  #1.5
    y = np.where(y == 0, -1, y)
    v = np.abs(x[:, 0]) * 10 + np.random.normal(0, 0.1, n)
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


def main_random(n, x, y, v):
    T = 50000
    records = []
    for mu in np.linspace(0.7, 0.75, 1):
        for run in range(T):
            new_b, allocation, payments, chi = mechanism_3(x, y, v, mu, train_soft_svm)
            for i in range(n):
                records.append({
                    'run': run,
                    'agent': i,
                    'true_b': v[i], ##assumin g truthfulness
                    'new_b': new_b[i],
                    'allocation': allocation[i],
                    'payment': payments[i],
                    'chi': chi[i],
                    'mu': mu,
                    'welfare': v[i] * allocation[i], #- payments[i],
                    'utility': v[i] * allocation[i]  - payments[i],
                })

    # === Results DataFrame ===
    df = pd.DataFrame(records)
    df.to_csv(f"random_simulation_n={n}.csv", index=False)
    IR(df)
    print(f'check IR in expectation')
    IR(df, exp=True)
    return df


def get_relevant_indcies(svm_model, beta, v, er = 0.02):
    w = svm_model.coef_[0]
    b = svm_model.intercept_[0]

    # Decision values: y * (w·x + b)
    decision_values = y * (x @ w + b)

    # Support vector indices
    support_vector_indices = svm_model.support_

    # Decision values for support vectors
    sv_decision_values = decision_values[support_vector_indices]

    # Mask of correctly classified SVs
    mask_correct = sv_decision_values > 0
    correct_sv_indices = support_vector_indices[mask_correct]

    print(correct_sv_indices)

    # Extract w and b
    w = svm_model.coef_.ravel()
    
    # Compute distances
    norm_w = np.linalg.norm(w)
    distances_correct = np.abs(sv_decision_values[mask_correct]) / norm_w

    print(distances_correct)
    print(beta * v[correct_sv_indices])
    thresh = np.abs(distances_correct - beta * v[correct_sv_indices])
    print(thresh)
    filtered_indices = correct_sv_indices[thresh < er]
    return filtered_indices


def main_exact(x, y, v,  plot = False, c = 1.0):
    records = []
    correct_sv_indices = range(0,len(y))
    M = np.sum(v)
    for target_idx in correct_sv_indices:
        # print(f"\n=== Processing target {target_idx} ===")
        critical_v, alloc, _ = compute_critical_bid(x, y, M, v, target_idx, train_soft_svm, plot = plot)
        records.append({
                "agent": target_idx,
                "allocation": alloc,
                "true_v": v[target_idx],
                "critical_v": critical_v,
                'welfare':  v[target_idx] *  alloc, # - critical_v ,
                'utility': v[target_idx] * alloc - critical_v ,
            })


    df = pd.DataFrame(records)
    non_zero_df = df[df['critical_v'] != 0]
    non_zero_indices = non_zero_df['agent']
    
    print(f"Number of non-zero payments: {len(non_zero_indices)}")
    print(f"Indices with non-zero payments: {list(non_zero_indices)}")

    # Print critical_v and true_v for those indices
    print("Critical_v and True_v values:")
    print(non_zero_df[['agent', 'critical_v', 'true_v']])

    return df, svm_model


def main_exact_in_on_margin(x, y, v, tol = 1e-2):
    print(f'take out points that are out of margin')
    model = train_soft_svm(x, y, v)

    decision_values = model.decision_function(x)
    margin_distances = y * decision_values
    inside_or_on_margin_mask = margin_distances <= 1 + tol
    # Filter x, y, v to keep only those points
    x_filtered = x[inside_or_on_margin_mask]
    y_filtered = y[inside_or_on_margin_mask]
    v_filtered = v[inside_or_on_margin_mask]
    n_filtered = len(x_filtered)

    # Get indices of points outside the margin
    in_on_margin_indices = np.where(margin_distances <= 1 + tol)[0]
    print("Indices inside/on the margin:", in_on_margin_indices)

    print(f'{n_filtered} left out of {n}')
    df = main_exact(n_filtered, x_filtered, y_filtered, v_filtered)
    return df


def drop_one_by_one(x, y, v):
    print('Removing points outside the margin one-by-one (farthest first)')

    model = train_soft_svm(x, y, v)
    decision_values = model.decision_function(x)
    margin_distances = y * decision_values

    # Identify violations: points with margin_dist > 1
    outside_indices = np.where(margin_distances > 1 + 1e-2)[0]
    violations = margin_distances[outside_indices] - 1

    # Sort indices by margin violation (farthest from margin first)
    sorted_outside = outside_indices[np.argsort(-violations)]
    print("Sorted indices to remove:", sorted_outside)

    # Store data as a dict with index keys
    data_dict = {i: (x[i], y[i], v[i]) for i in range(len(x))}

    # Remove one-by-one based on sorted indices
    for idx in sorted_outside:
        print(f"Removing index {idx} with margin distance {margin_distances[idx]:.4f}")
        if idx in data_dict:
            del data_dict[idx]
            # Reconstruct filtered arrays
            remaining_indices = sorted(data_dict.keys())
            x_filtered = np.array([data_dict[i][0] for i in remaining_indices])
            y_filtered = np.array([data_dict[i][1] for i in remaining_indices])
            v_filtered = np.array([data_dict[i][2] for i in remaining_indices])
            n_filtered = len(x_filtered)
            print(f'{n_filtered} left out of {n}')
            df = main_exact(n_filtered, x_filtered, y_filtered, v_filtered)
            IR(df)
            exact_min_sum = df['critical_v'].sum()
            print(f'exact {n_filtered} sum: {exact_min_sum}')
    return True


def lvl_1(x,y,v):
    T = 5000
    allocation_model = train_soft_svm(x, y, v)
    pred = np.array(allocation_model.predict(x))
    allocations = (y == pred).astype(float)
    all_payments = []

    for _ in range(T):
        payments = mechanism_1(x, y, v, train_soft_svm, allocations)
        all_payments.append(payments)

    # Stack all payments (shape: [100, n_agents]) and take mean over axis 0
    all_payments = np.vstack(all_payments)
    avg_payments = np.mean(all_payments, axis=0)
    var_payments = np.var(all_payments, axis=0)

    return avg_payments, var_payments, np.sum(all_payments)


# Sample version of main_random for a fixed mu, and returns DataFrame
def run_for_T(n, x, y, v, T, mu=0.7):
    records = []
    for run in range(T):
        new_b, allocation, payments, chi = mechanism_3(x, y, v, mu, train_soft_svm)
        for i in range(n):
            records.append({
                'run': run,
                'agent': i,
                'payment': payments[i],
                'T': T
            })
    df = pd.DataFrame(records)
    df.to_csv(f"check_sd.csv", index=False)
    return df


if __name__ == '__main__':
    n = 30
    np.random.seed(42)
    x, y, v = generate_data_centered(n)
    c = 1.0
    M = np.sum(v)
    print("M:", M)

    print("begin intial training")
    svm_model = train_soft_svm(x, y, M, v, c = c)

    plot_svm_decision_boundary(svm_model, x, y, v, 0)

    print("extract critical x's")
    # unpack model
    w = svm_model.coef_.ravel()
    b = svm_model.intercept_

    if b!=0:
        print("THE INTERCEPT IS NOT 0")

    # decision values for all points
    decision_values = x @ w + b
    margin = y * decision_values

    # correctly classified SVs: y * f(x) >= 1
    critical_sv_idx = np.where(margin <= 1)[0]

    x_critical = x[critical_sv_idx]
    y_critical = y[critical_sv_idx]
    v_critical = v[critical_sv_idx]

    print("calculate beta per critical x")

    c_scaled = c/M
    beta_values, k = compute_beta(x, v_critical, c_scaled, M, k=None)
    print("k =", k)
    print("x's:", critical_sv_idx)
    print("betas:", beta_values)
    print("scores:", margin[critical_sv_idx])
    print("eval:", margin[critical_sv_idx] < beta_values)
    filtered_numbers = [num for num, select in zip(critical_sv_idx, margin[critical_sv_idx] < beta_values) if select]
    print("relevant indcies:", filtered_numbers)

    # print(f'begin exact simulation')
    df_exact, model_1 = main_exact(x, y, v, plot = False, c = 1.0)
    # #df_exact.to_csv(f"exact_simulation_n={n}.csv", index=False)
    # #IR(df_exact)
    exact_sum = df_exact['critical_v'].sum()
    exact_util = df_exact['utility'].sum()
    print(f'exact sum: {exact_sum}')
    print(f'exact util: {exact_util}')


    # print(f'begin exact simulation_prime')
    # v_prime = v / v.sum()
    # C_prime = 1.0 * v.sum()
    # df_exact, model_2 = main_exact(x, y, v_prime, plot = False, c = C_prime)
    # exact_sum = df_exact['critical_v'].sum()
    # exact_util = df_exact['utility'].sum()
    # print(f'exact sum: {exact_sum}')
    # print(f'exact util: {exact_util}')

    # ok, details = check_equivalence(x, y, C=1.0, v=v, model1 = model_1, model2 = model_2)
    # print(ok)
    # print(details)
