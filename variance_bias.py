from utils import *
from main import *
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def gamma_mean1(rng, n, k):
    return rng.gamma(shape=k, scale=1/k, size=n)

def generate_gaus_data(n_pos, n_neg, mu_pos, mu_neg, sigma_pos, sigma_neg, rng, d=1):
    X_neg = rng.normal(loc=mu_neg, scale=sigma_neg, size=(n_neg, d)) # label -1 
    X_pos = rng.normal(loc=mu_pos, scale=sigma_pos, size=(n_pos, d)) # label 1 
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([-1 * np.ones(n_neg),np.ones(n_pos)])

    n = n_pos + n_neg

    v_const = np.ones(n)
    v_var_025 = gamma_mean1(rng, n, k=16)   # variance ≈ 0.0625
    v_var_05  = gamma_mean1(rng, n, k=4)    # variance ≈ 0.25
    v_var_1   = gamma_mean1(rng, n, k=1)    # variance = 1
    v_var_2   = gamma_mean1(rng, n, k=0.5)  # variance = 2
    v_var_3   = gamma_mean1(rng, n, k=0.33) # variance ≈ 3

    return X, y, v_const, v_var_025, v_var_05, v_var_1, v_var_2, v_var_3

def compute_v_type(df_xy, v, v_type, t, y, c=1, loss_key='hinge'):
    x, y = df_xy
    df, _ = run_simulation(x, y, v, c, loss_key, algorithm="exact")
    df['v_type'] = v_type
    df['t'] = t
    df['label'] = y[df['agent'].astype(int)]
    df["m"] = len(y)
    df['pays'] = (df['critical_v'] > 0).astype(int)
    return df

def compute_metrics(t):
    dfs = []
    sigma_pos = sigma_neg = 1.0
    d = 8
    mu_scalar = 0.5 
    mu = np.concatenate(([mu_scalar], np.zeros(d - 1)))
    rng = np.random.default_rng(t)
    for m in ms:
        # generate x,y and all vs only once
        x, y, v_const, v_var_025, v_var_05, v_var_1, v_var_2, v_var_3 = \
            generate_gaus_data(m, m, mu, -mu, sigma_pos, sigma_neg, rng, d)
        
        vs_dict = dict(zip(vs_types, [v_const, v_var_025, v_var_05, v_var_1, v_var_2, v_var_3]))
        
        # run each v_type in parallel
        dfs_m = Parallel(n_jobs=len(vs_types))(
            delayed(compute_v_type)((x, y), v, v_type, t, y) 
            for v_type, v in vs_dict.items()
        )
        dfs.extend(dfs_m)
    return pd.concat(dfs, ignore_index=True)


res = []
ms = [1000, 2500, 5000, 10000]
vs_types = ['const', 'var_025', 'var_05', 'var_1', 'var_2', 'var_3']
    
num_cpus = 100
results = Parallel(n_jobs=num_cpus)(
    delayed(compute_metrics)(t) for t in range(20)
)

df_res = pd.concat(results, ignore_index=True)
df_res.to_csv('simulation_results.csv', index=False)

