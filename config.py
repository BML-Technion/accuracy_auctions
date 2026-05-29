import numpy as np
from numpy.random import SeedSequence, default_rng


# -----------------------
# Randomization
# -----------------------
MASTER_SEED = 12345       # master seed for reproducibility
N_TRIALS = 10      # number of independent trials

# Master random generator (used to generate independent trial seeds)
MASTER_SS = SeedSequence(MASTER_SEED)
TRIAL_SS = MASTER_SS.spawn(N_TRIALS)

# -----------------------
# SVM training configuration
# -----------------------
MODEL_C = 1.0
MODEL_FIT_INTERCEPT = True
MODEL_KERNEL = "rbf"
ONE_SHOT_MU = 0.5
PLOT = False

TRAIN_SVM_PARAMS = {
    "hinge": {
        "C": MODEL_C,
        "loss": "hinge",
        "max_iter": 100000,
        "fit_intercept": MODEL_FIT_INTERCEPT,
        "dual": False,
        "random_state": 0,
        "admissibility": 1
    },
    "squared_hinge": {
        "C": MODEL_C,
        "loss": "squared_hinge",
        "max_iter": 100000,
        "fit_intercept": MODEL_FIT_INTERCEPT,
        "dual": True,
        "random_state": 0,
        "admissibility": 1
    },
    "log": {
        "C": MODEL_C,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": 100000,
        "fit_intercept": MODEL_FIT_INTERCEPT,
        "random_state": 0,
        "admissibility": 1
    },
    "poly2": {
        "C": MODEL_C,
        "kernel": "poly",
        "degree": 2,
        "coef0": 1,
        "gamma": "auto",
        "max_iter": 600000,
        "random_state": 0,
        "admissibility": 1
    },
    "poly3": {
        "C": MODEL_C,
        "kernel": "poly",
        "degree": 3,
        "coef0": 1,
        "gamma": "auto",
        "max_iter": 1000000,
        "random_state": 0,
        "admissibility": 1
    },
    "rbf1": {
        "C": MODEL_C,
        "kernel": "rbf",
        "gamma_multiplier": 2.0,
        "max_iter": 10000,
        "random_state": 0,
        "admissibility": 1
    },
    "rbf2": {
        "C": MODEL_C,
        "kernel": "rbf",
        "gamma_multiplier": 1.0,
        "max_iter": 10000,
        "random_state": 0,
        "admissibility": 1
    },
    "rbf3": {
        "C": MODEL_C,
        "kernel": "rbf",
        "gamma_multiplier": 0.5,
        "max_iter": 10000,
        "random_state": 0,
        "admissibility": 1
    },
}

# -----------------------
# Relevance and throw
# -----------------------
K_COEF = 1

IS_THROW = {
    "hinge": True,
    "squared_hinge": False,
    "log": False,
    "poly2": False,
    "poly3": False,
    "rbf1": False,
    "rbf2": False,
    "rbf3": False,
}

# -----------------------
# Vary mus - accuracy
# -----------------------
MU_LIST = np.linspace(3, 0, 21)
D_LIST = [2,4,8,16,32]

# -----------------------
# Vary c - calculate runtime
# -----------------------
C_LIST = np.logspace(-3, 3, num=20)
D_LIST = [2,4,8,16,32] 

# -----------------------
# KNN
# -----------------------
D = 2
M_LIST = [20, 50, 100, 150, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 10000, 15000, 20000, 25000]
K_LIST = [1,2,3,5,7]