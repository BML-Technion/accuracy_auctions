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


def train_soft_svm(x, y, v=None, c=1.0):
    """
    Train a linear SVM with:
      - bias fixed to zero (fit_intercept = False)
      - hinge loss averaged over sum(v) instead of N
    """
    
    # If no sample weights provided, use all weights = 1
    if v is None:
        v = np.ones(len(y))

    model = LinearSVC(
        C=c,
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


def get_outside_margins(svm_model, x, y, v, c, k=None):
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
    V_critical = v[outside_margin_indices]

    # lambda = 1/C
    lam = 1.0 / c

    # compute k if not provided: max L2 norm of x
    if k is None:
        k = np.linalg.norm(x, axis=1).max()

    # compute beta values (vectorized)
    beta_values = (k**2 * V_critical) / (2 * lam)

    # final selection: margin > 1 + beta
    selected_mask = margin[outside_margin_indices] > (1+ beta_values) #- tol)
    throw = outside_margin_indices[selected_mask]
    return throw

    # mask = np.ones_like(v, dtype=bool)
    # mask[throw] = False

    # new_x = x[mask]
    # new_y = y[mask]
    # new_v = v[mask]

    
    # return new_x, new_y, new_v

def get_relevant_indices(svm_model, x, y, v, c, k=None):
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
        k = np.linalg.norm(x, axis=1).max()

    # compute beta values (vectorized)
    beta_values = (k**2 * V_critical) / (2 * lam)

    # select: margin < beta   (vectorized)
    selected_mask = margin[support_vector_indices] < beta_values
    relevant_indices = support_vector_indices[selected_mask]

    return relevant_indices


def main_random(x, y, v, relevant_indcies):
    T = 50000
    n = len(y)
    records = []
    for mu in np.linspace(0.7, 0.75, 1):
        for run in range(T):
            new_b, allocation, payments, chi = mechanism_3(x, y, v, mu, train_soft_svm, relevant_indcies)
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


def main_exact(x, y, v, relevant_indcies, plot = False, c = 1.0):
    records = []
    for target_idx in relevant_indcies:
        print(f"\n=== Processing target {target_idx} ===")
        critical_v, alloc, _ = compute_critical_bid(x, y, v, target_idx, train_soft_svm, plot = plot, c=c)
        records.append({
                "agent": target_idx,
                "allocation": alloc,
                "true_v": v[target_idx],
                "critical_v": critical_v,
                'welfare':  v[target_idx] *  alloc,
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


def models_equivalent(m1, m2, tol=1e-4):
    same_coef = np.allclose(m1.coef_, m2.coef_, atol=tol)
    #same_intercept = np.allclose(m1.intercept_, m2.intercept_, atol=tol)
    print(m1.coef_, m2.coef_,)
    return same_coef #and same_intercept


if __name__ == '__main__':
    n = 300
    np.random.seed(42)
    x, y, v = generate_data_centered(n)
    c = 1.0
    M = np.sum(v)

    print("begin intial training")
    svm_model = train_soft_svm(x, y, v, c = (c/M))
    # print("Throw out points outside of 1+beta margin that are correctly classified")
    # new_x, new_y, new_v = get_outside_margins(svm_model, x, y, v, c = c/M)
    # print("get relevant indcies")
    # relevant_indcies = get_relevant_indices(svm_model, new_x, new_y, new_v, c/M)
    # print(f'begin exact simulation')
    # df_exact, model_1 = main_exact(new_x, new_y, new_v, relevant_indcies, plot = False, c = (c/M))
    # exact_sum = df_exact['critical_v'].sum()
    # exact_util = df_exact['utility'].sum()
    # print(f'exact sum: {exact_sum}')
    # print(f'exact util: {exact_util}')




    plot_svm_decision_boundary(svm_model, x, y, v, 0)

    print("get relevant indcies")
    relevant_indcies = get_relevant_indices(svm_model, x, y, v, c/M)
    print("relevant indcies: ", relevant_indcies)

    print(f'begin exact simulation')
    df_exact, model_1 = main_exact(x, y, v, relevant_indcies, plot = False, c = (c/M))
    exact_sum = df_exact['critical_v'].sum()
    exact_util = df_exact['utility'].sum()
    print(f'exact sum: {exact_sum}')
    print(f'exact util: {exact_util}')

    print("Throw out points outside of 1+beta margin that are correctly classified")
    throw = get_outside_margins(svm_model, x, y, v, c = c/M)
    print("throw:", throw)

    mask = np.ones_like(v, dtype=bool)
    mask[throw] = False

    new_x = x[mask]
    new_y = y[mask]
    new_v = v[mask]

    print('train again')
    new_svm_model = train_soft_svm(new_x, new_y, new_v, c = (c/M))

    plot_svm_decision_boundary(new_svm_model, new_x, new_y, new_v, 0)

    print("get relevant indcies")
    relevant_indcies = get_relevant_indices(new_svm_model, new_x, new_y, new_v, c/M)
    print("relevant indcies: ", relevant_indcies)

    is_equ = models_equivalent(svm_model, new_svm_model)
    if is_equ:
        print("The models are equivelent")
    else:
        print("The models are NOT equivelent")

    print(f'begin exact simulation for minimized model')
    df_exact, model_1 = main_exact(new_x, new_y, new_v, relevant_indcies, plot = False, c = (c/M))
    exact_sum = df_exact['critical_v'].sum()
    exact_util = df_exact['utility'].sum()
    print(f'exact sum: {exact_sum}')
    print(f'exact util: {exact_util}')








