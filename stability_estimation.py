# import matplotlib.pyplot as plt
# import numpy as np
# from data_analysis import *
# from utils import *
# from sklearn.svm import SVC



# def sample_from_distribution(n, seed):
#     rng = np.random.default_rng(seed)
#     X_neg = rng.uniform(-1, 0.25, size=(int(n/2), 1))
#     y_neg = -1 * np.ones(len(X_neg))
#     X_pos = rng.uniform(-0.25, 1, size=(int(n/2), 1))
#     y_pos = np.ones(len(X_pos))
#     v = np.ones(n)
#     X = np.vstack([X_neg, X_pos])
#     y = np.concatenate([y_neg, y_pos])
#     return X, y

# def classification_stability_from_distribution(
#     n_train=50,
#     n_datasets=20,
#     C=1.0,
#     kernel="linear",
#     random_state=0,
#     p_pos=0.5,
#     verbose=True
# ):
#     rng_master = np.random.RandomState(random_state)
#     betas = []

#     for ds_idx in range(n_datasets):
#         rng = np.random.RandomState(rng_master.randint(2**31 - 1))

#         # sample train and large fresh test set from true distribution
#         X_train, y_train = sample_from_distribution(n_train, rng)
#         plt.scatter(X_train,y_train)

#         # full-data model
#         full_clf = SVC(C=C, kernel=kernel)
#         full_clf.fit(X_train, y_train)
#         f_full = full_clf.decision_function(X_train)  # shape (n_test,)


#         # for each i, compute sup_x |f_full(x) - f_loo(x)| approx over X_test
#         sup_diffs = np.zeros(n_train, dtype=float)

#         for i in range(n_train):
#             print(f'running for {i}')
#             mask = np.ones(n_train, dtype=bool)
#             mask[i] = False
#             X_sub, y_sub = X_train[mask], y_train[mask]

#             clf_loo = SVC(C=C, kernel=kernel)
#             clf_loo.fit(X_sub, y_sub)
#             f_loo = clf_loo.decision_function(X_train)
#             plot_svm_decision_boundary_2_models(full_clf, clf_loo, X_train, y_train, v= None, target_idx=i, title="Stability estimation")

#             # sup over test sample approximated as max absolute difference
#             sup_diffs[i] = np.max(np.abs(f_full - f_loo))

#         # dataset-level empirical beta = worst-case over i
#         beta_S = sup_diffs.max()
#         betas.append(beta_S)

#         if verbose:
#             print(f"[dataset {ds_idx+1}/{n_datasets}] beta_S = {beta_S:.6f} (max over {n_train} LOO models)")

#     betas = np.array(betas)
#     results = {
#         "n_train": n_train,
#         "n_datasets": n_datasets,
#         "C": C,
#         "kernel": kernel,
#         "empirical_beta_mean": betas.mean(),
#         "empirical_beta_std": betas.std(ddof=1),
#         "empirical_beta_all": betas,               # per-dataset betas
#         "theoretical_bound": C / (2 * n_train)
#     }

#     # print summary
#     print("\n=== Classification Stability Summary ===")
#     print(f"train size n = {n_train}, datasets = {n_datasets}")
#     print(f"empirical beta: mean = {results['empirical_beta_mean']:.6f}, std = {results['empirical_beta_std']:.6f}")
#     print(f"theoretical bound (C/n): {results['theoretical_bound']:.6f}")

#     return results


# res = classification_stability_from_distribution(
#     n_train=100,
#     n_datasets=10,
#     C=1.0,
#     kernel="linear",
#     random_state=42,
#     verbose=True
# )

import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt


def estimate_svm_stability(n, num_repeats=5, C=1.0, val_size=1000, random_state=None):
    """
    Empirically estimate the stability of a soft-margin SVM for a given sample size n.
    """
    rng = np.random.default_rng(random_state)
    stabilities = []

    for _ in range(num_repeats):
        # Generate training and validation data
        X, y = make_classification(
            n_samples=n, n_features=10, n_informative=8, n_redundant=2,
            n_classes=2, random_state=rng.integers(1e6)
        )
        X_val, y_val = make_classification(
            n_samples=val_size, n_features=10, n_informative=8, n_redundant=2,
            n_classes=2, random_state=rng.integers(1e6)
        )

        # Train SVM on full dataset
        full_svm = SVC(C=C, kernel='linear')
        full_svm.fit(X, y)
        f_full = full_svm.decision_function(X_val)

        # Leave-one-out retraining and measure sensitivity
        diffs = []
        for i in range(n):
            X_minus_i = np.delete(X, i, axis=0)
            y_minus_i = np.delete(y, i, axis=0)

            reduced_svm = SVC(C=C, kernel='linear')
            reduced_svm.fit(X_minus_i, y_minus_i)
            f_minus_i = reduced_svm.decision_function(X_val)

            diffs.append(mean_absolute_error(f_full, f_minus_i))

        stabilities.append(np.mean(diffs))

    return np.mean(stabilities)


if __name__ == "__main__":
    sample_sizes = [20, 40, 80, 160, 320]
    C = 1.0
    empirical = []

    for n in tqdm(sample_sizes, desc="Estimating empirical stability"):
        stability_n = estimate_svm_stability(n, num_repeats=3, C=C)
        empirical.append(stability_n)

    # Fit the proportional constant k to match empirical data scale
    empirical = np.array(empirical)
    theoretical = lambda n, k: k * C / n
    k_fit = np.mean(empirical * sample_sizes / C)

    # Compute theoretical curve
    theoretical_values = theoretical(np.array(sample_sizes), k_fit)

    # ---- Plot both curves ----
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, empirical, 'o-', label='Empirical Stability')
    plt.plot(sample_sizes, theoretical_values, 'r--', label=f'Theoretical ~ kC/n (k={k_fit:.3f})')
    plt.title("Soft SVM Stability vs. Sample Size")
    plt.xlabel("Sample size (n)")
    plt.ylabel("Average Stability (Δ prediction)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
