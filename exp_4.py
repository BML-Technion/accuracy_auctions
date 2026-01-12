
# %%
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

from randomness import *
from df_metrics import *
from simulation_exact import *
from utils import *
from data_analysis import *
from main import *
import numpy as np
from joblib import Parallel, delayed



# %%
use_loss = 'hinge' #'log' or 'hinge' or 'squared_hinge' 
sigma_loss = 1.0 
c = 1.0 # choose c without scaling 
show_plots = False # set to True to see plots of the binary search
is_throw = True # set to True to throw out points outside margin --> only when loss is lipschitz. 
k = None # if none will be set based on data max norm
k_coef = 1.0 # coefficient to multiply k with
fit_intercept = True # whether to fit intercept in SVM model
d = 16 #[2, 4, 8, 16, 32, 64] # dimensions to run experiments on
T = 50  # number of trials
ms = [2**i for i in range(18, 25)]
mu_pos_scalar = 0.2 # class 1 mean
mu_neg_scalar = -0.2 # class -1 mean
mu_pos =  np.concatenate(([mu_pos_scalar], np.zeros(d - 1)))
mu_neg =  np.concatenate(([mu_neg_scalar], np.zeros(d - 1)))
sigmas = [0.1, 0.2, 0.5, 1, 1.5, 2 , 2.5] #np.linspace(2, 0.1, 31) # different sigmas to run the experiments on

# %%
def generate_gaus_data(n_pos, n_neg, mu_pos, mu_neg, sigma_pos, sigma_neg, rng, d=1):
    X_neg = rng.normal(loc=mu_neg, scale=sigma_neg, size=(n_neg, d)) # label -1 
    X_pos = rng.normal(loc=mu_pos, scale=sigma_pos, size=(n_pos, d)) # label 1 
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([-1 * np.ones(n_neg),np.ones(n_pos)])

    v = np.concatenate([np.ones(n_pos),  np.ones(n_neg)])  # generate_v_data(n_neg, n_pos, rng)
    return X, y, v

# def compute_metrics_gaus_2(t):
#     dfs = []
#     for m in ms:
#         for sigma in sigmas:
#             seed = hash((t, m, sigma)) % (2**32)
#             rng = np.random.default_rng(seed)
#             x, y, v =  generate_gaus_data(m//2, m//2, mu_pos, mu_neg, sigma, sigma, rng, d)
#             #x_val, y_val, _ =  generate_gaus_data(n_val_pos, n_val_neg, mu_pos, mu_neg, sigma, sigma, rng, d)
            
#             df_exact, svm_model = run_exact(x, y, v, c, use_loss, sigma_loss=sigma_loss, plot=show_plots, 
#                                     is_throw=is_throw, k=k, k_coef=k_coef, fit_intercept=fit_intercept)
#             df_exact['t'] = t
#             df_exact['d'] = d
#             df_exact['sigma'] = sigma
#             df_exact['m_total'] = m
#             #df_exact['valid_acc'] = svm_model.score(x_val,y_val)
#             df_exact['label'] = y[df_exact['agent'].astype(int)]
#             dfs.append(df_exact)

#     return pd.concat(dfs, ignore_index=True)



# # %%
# mega_df_16_2 = pd.concat(  
#     Parallel(n_jobs=-1, backend="loky")(
#         delayed(compute_metrics_gaus_2)(t) for t in range(T)
#         ), 
#         ignore_index=True
# )

# mega_df_16_2.to_csv('exp_4.csv',index=False)


for m in ms:
    print(f"Running m = {m}...")

    def compute_metrics_for_m(t, m=m):  # fix m for each parallel call
        dfs = []
        for sigma in sigmas:
            seed = hash((t, m, sigma)) % (2**32)
            rng = np.random.default_rng(seed)
            x, y, v = generate_gaus_data(m//2, m//2, mu_pos, mu_neg, sigma, sigma, rng, d)

            df_exact, svm_model = run_exact(
                x, y, v, c, use_loss, sigma_loss=sigma_loss, plot=show_plots, 
                is_throw=is_throw, k=k, k_coef=k_coef, fit_intercept=fit_intercept
            )

            df_exact['t'] = t
            df_exact['d'] = d
            df_exact['sigma'] = sigma
            df_exact['m_total'] = m
            df_exact['label'] = y[df_exact['agent'].astype(int)]

            dfs.append(df_exact)

        return pd.concat(dfs, ignore_index=True)

    # Parallel run over trials
    mega_df_m = pd.concat(
        Parallel(n_jobs=-1, backend="loky")(
            delayed(compute_metrics_for_m)(t) for t in range(T)
        ),
        ignore_index=True
    )

    # Save after each m
    filename = f'exp_m_{m}.csv'
    mega_df_m.to_csv(filename, index=False)
    print(f"Saved results for m = {m} to {filename}")
