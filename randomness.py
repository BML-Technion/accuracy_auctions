from utils import *
import numpy as np


def mechanism_1(x, y, b, allocation_function, allocations):
    """
    Archer & Tardos randomized payment mechanism.

    Args:
        x: Features.
        y: True labels.
        b: Bids (valuations) of agents.
        mu: Unused for now, placeholder.
        allocation_function: Function returning model with predict() method.
        num_samples: Number of samples to estimate winning probability.

    Returns:
        allocations: Binary array (1 if agent wins, else 0).
        payments: Payment for each agent (0 if not allocated).
    """
    n = len(b)
    b = np.array(b)
    payments = np.zeros(n)

    for i in np.where(allocations == 1)[0]:
        # print(f'randomizing for agent {i}')
        # Step 2: Draw u ~ Uniform[0, b[i]]
        u = np.random.uniform(0, b[i])
        b_u = b.copy()
        b_u[i] = u

        # Step 3: Check if i still wins when bidding u
        model_u = allocation_function(x, y, b_u)
        pred_u = model_u.predict(x)
        wins_with_u = int(pred_u[i] == y[i])

        # Step 4: Compute Z
        Z = b[i] if wins_with_u else 0

        # Step 6: Random price
        payments[i] = b[i] - Z

    return payments


def mechanism_3(x, y, b, mu, allocation_function):
    """
    Implements Mechanism 3: transformation with resampling for positive types.

    Parameters:
    - b: array-like, bid vector (length n), where b_i > 0
    - mu: float in (0, 1), the resampling probability
    - allocation_function: function A(x) that takes a vector x and returns allocation vector of same length

    Returns:
    - x: modified bids vector
    - allocation: result of allocation_function(x)
    - payments: payments assigned to each agent
    - chi: scaling vector used for each agent
    """
    n, b, chi= len(b), np.array(b), np.ones(len(b))

    # Step 3–5: Generate rescaling factor chi_i for each agent
    for i in range(n):
        d = np.random.uniform(0, 1)
        if d < mu:
            gamma_i = np.random.uniform(0, 1)
            chi[i] = gamma_i ** (1 / (1 - mu))
        else:
            chi[i] = 1.0

    # Step 6: Construct modified bids x_i = chi_i * b_i
    new_b = chi * b

    # Step 7: Allocate using A(x)
    allocation_model = allocation_function(x, y, v = new_b)
    pred = np.array(allocation_model.predict(x))
    allocation = (y == pred).astype(float)
    # Step 8: Compute payments
    payments = np.zeros(n)
    for i in range(n):
        if chi[i] == 1.0:
            payments[i] = b[i] * allocation[i]
        else:
            payments[i] = b[i] * allocation[i] * (1 - 1 / mu)

    return new_b, allocation, payments, chi


def mechanism_2(x, y, b, mu, allocation_function):
    pass