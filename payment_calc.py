# %%
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

from main import *
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# %%
def generate_gaus_data(n_pos, n_neg, mu_pos, mu_neg, sigma_pos, sigma_neg, rng, d=1):
    X_neg = rng.normal(loc=mu_neg, scale=sigma_neg, size=(n_neg, d)) # label -1 
    X_pos = rng.normal(loc=mu_pos, scale=sigma_pos, size=(n_pos, d)) # label 1 
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([-1 * np.ones(n_neg),np.ones(n_pos)])

    v = np.concatenate([ np.ones(n_pos + n_neg)])  # generate_v_data(n_neg, n_pos, rng)
    return X, y, v

# %%
base_seed = 12345
loss_key = 'hinge' 
sigma_loss = 1.0 
c = 1
d = 4 # dimensions to run experiments on
T = 1 # 20  # number of trials
#-----

# choose parameters 
n_pos = 1000 # number of data points in a sample with label 1
n_neg = 1000 # number of data points in a sample with label -1
n_val_pos = 200
n_val_neg = 200
sigma_pos = 1.0
sigma_neg = 1.0
mu_scalar = 0.5
mu_pos = np.concatenate(([mu_scalar], np.zeros(d - 1)))
mu_neg = np.concatenate(([-mu_scalar], np.zeros(d - 1)))

#-----
show_plots = False # set to True to see plots of the binary search
is_throw = True # set to True to throw out points outside margin --> only when loss is lipschitz. 
k = None # if none will be set based on data max norm
k_coef = 1.0 # coefficient to multiply k with
fit_intercept = True # whether to fit intercept in SVM model

# %%
def compute_paymentss(t, N=50, oneshot_mus=[0.25, 0.4,  0.5, 0.6, 0.75]):
    dfs = []

    # ---- Base RNG for this trial
    ss = np.random.SeedSequence([base_seed, t, d, int(mu_scalar*1e6)])
    rng = np.random.default_rng(ss)

    # ---- Generate data once per trial
    x, y, v = generate_gaus_data(n_pos, n_neg, mu_pos, mu_neg, sigma_pos, sigma_neg, rng, d)

    # ---- Exact run (deterministic)
    df, _ = run_simulation(x, y, v, c, loss_key, algorithm="exact", n_random=1, rng=None, k=None)
    df['t'] = t
    df['d'] = d
    df['label'] = y[df['agent'].astype(int)]
    df['calc'] = 'exact'
    dfs.append(df)

    print("Accuracy of exact SVM:", (df['allocation'].mean()))

    # ---- Random runs and oneshot runs
    for n in range(1, N+1):
        for r in [1,3,6,9]:
            # Each run gets its own independent seed
            seed_random = int((t+1)*1e6 + n)
            rng_r = np.random.default_rng(seed_random)
            # Random mechanism using SVM
            df_random, _ = run_simulation(x, y, v, c, loss_key, algorithm="random", n_random=r, rng=rng_r)
            df_random['t'] = t
            df_random['d'] = d
            df_random['label'] = y[df_random['agent'].astype(int)]
            df_random['calc'] = f'random-{r}'
            df_random['n'] = n
            dfs.append(df_random)

        # Oneshoot runs with different mus
        for mu_val in oneshot_mus:
            df_oneshot, _ = run_simulation(x, y, v, c, loss_key, algorithm="one-shot", n_random=1, rng=rng_r)
            df_oneshot['t'] = t
            df_oneshot['d'] = d
            df_oneshot['label'] = y[df_oneshot['agent'].astype(int)]
            df_oneshot['calc'] = f'oneshot-{mu_val}'
            df_oneshot['n'] = n
            dfs.append(df_oneshot)

    return pd.concat(dfs, ignore_index=True)


# %%
df = compute_paymentss(1)

# %%
df.to_csv(f"payment_c={c}_d={d}_n={n_pos}.csv", index=False)
