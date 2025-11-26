import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

def check_equivalence(X, y, C, v, tol=1e-6):
    """
    Check if (C, v) is equivalent to (C', v') where
    v' = v / sum(v) and C' = C * sum(v).
    """
    # Normalized weights and scaled C
    v_prime = v / v.sum()
    C_prime = C * v.sum()

    # Train first model
    model1 = SVC(kernel="linear", C=C, random_state=0)
    model1.fit(X, y, sample_weight=v)

    # Train second model
    model2 = SVC(kernel="linear", C=C_prime, random_state=0)
    model2.fit(X, y, sample_weight=v_prime)

    # Extract parameters
    w1, b1 = model1.coef_.ravel(), model1.intercept_[0]
    w2, b2 = model2.coef_.ravel(), model2.intercept_[0]

    # Compare
    same_w = np.allclose(w1, w2, atol=tol)
    same_b = np.allclose(b1, b2, atol=tol)

    return same_w and same_b, {
        "same_w": same_w,
        "same_b": same_b,
        "diff_w_norm": np.linalg.norm(w1 - w2),
        "diff_b": abs(b1 - b2),
        "w1": w1,
        "w2": w2,
        "b1": b1,
        "b2": b2,
        "C_prime": C_prime,
        "v_prime_sum": v_prime.sum()
    }

#acc 0.8 + payments
# Generate or load data
def generate_data(n):
    # Generate synthetic data
    x, y = make_blobs(n_samples=n, centers=2, random_state=0, cluster_std=1.5)  #1.5
    y = np.where(y == 0, -1, y)
    v = np.abs(x[:, 0]) * 10 + np.random.normal(0, 0.1, n)
    return x, y, v

n=30
x, y, v = generate_data(n)
ok, details = check_equivalence(x, y, C=1.0, v=v)
print(ok)
print(details)
