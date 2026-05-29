import warnings 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
import numpy as np
import pandas as pd
import time
from config import *
from utils import *

def process_agent_pay(i, x, y, v, new_x, new_y, new_v,
                      svm_model, base_preds,
                      relevant_set, throw_set,
                      real_to_updated, c, loss_key):

    alloc_0 = 0
    pays = 0

    is_throw = 1 if i in throw_set else 0
    is_relevant = 1 if i in relevant_set else 0

    if is_relevant:
        updated_idx = real_to_updated[i]

        tmp_v = new_v.copy()
        tmp_v[updated_idx] = 0.0

        model_0 = train_soft_svm(new_x, new_y, tmp_v, c, loss_key)
        alloc_0 = int(model_0.predict(x[i].reshape(1, -1))[0] == y[i])

        if alloc_0 == 0:
            pays = 1

    return {
        "agent": i,
        "allocation": int(base_preds[i] == y[i]),
        "true_v": v[i],
        "is_throw": is_throw,
        "is_relevant": is_relevant,
        "alloc 0 success": int(alloc_0 == 1) if is_relevant else 0,
        "pays": pays,
    }

def predict_knn(knn, x, y, v, k):
    distances, indices = knn.kneighbors(x, n_neighbors = k)
    preds = []
    for i, neigh_idx in enumerate(indices):
        weighted_sum = np.sum(y[neigh_idx] * v[neigh_idx])
        preds.append(np.sign(weighted_sum))
    preds = np.array(preds)
    # print(preds)
    
    return preds

def train_knn(x, y, v, k, plot):
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    knn.fit(x, y)
    if plot:
        plot_knn_2d(knn, x, y, v, k=k)
    return knn

def train_soft_svm(x, y, v= None, c=1 ,loss="hinge"):
    # print("training model")
    if v is None:
        print("v is none")
        v = np.ones(len(y))

    if loss not in TRAIN_SVM_PARAMS:
        raise ValueError(f"Unsupported loss: {loss}")

    params = TRAIN_SVM_PARAMS[loss]
    random_state = 0
    max_iter = 100000
    

    if loss == "hinge":
        model = LinearSVC(
            C=c,
            loss=params["loss"],
            max_iter=max_iter,
            fit_intercept=params["fit_intercept"],
            dual=True,
            random_state=random_state, 
            tol=1e-3
        )
    elif loss == "squared_hinge":
        # print("training for squared_hinge")
        model = LinearSVC(
            C=c,
            loss="squared_hinge",
            dual=False,
        )
    elif loss == "log":
        model = LogisticRegression(
            C=c,
            penalty=params["penalty"],
            solver=params["solver"],
            max_iter=max_iter,
            fit_intercept=params["fit_intercept"],
            random_state=random_state
        )

    elif loss in ["poly2", "poly3"]:
        model = SVC(
            C=c,
            kernel=params["kernel"],
            degree=params["degree"],
            coef0=params["coef0"],
            gamma=params["gamma"],
            max_iter=max_iter,
            random_state=random_state
        )

    elif loss in ["rbf1", "rbf2", "rbf3"]:
        gamma_scale = 1.0 / (x.shape[1] * x.var())
        gamma = params["gamma_multiplier"] * gamma_scale

        model = SVC(
            C=c,
            kernel=params["kernel"],
            gamma=gamma,
            max_iter=max_iter,
            random_state=random_state
        )
    
    model.fit(x, y, sample_weight=v)

    return model

def run_simulation(x, y, v, c, loss_key, algorithm="exact", n_random=1, rng=None, k=None):
    """
    Run a simulation using a specified algorithm:
    - "exact" : exact critical payments
    - "random": expected payments over n_random draws
    - "one-shot": random mechanism 3 - babaioff
    """
    if algorithm == "one-shot":
        return one_shot_payments(x, y, v, c, loss_key, ONE_SHOT_MU, rng)


    svm_model = train_soft_svm(x, y, v, c, loss_key)

    relevant_idx, throw_idx = get_relevant_throw_idx(svm_model, x, y, v, c, loss_key)
    # print(f"Relevant idx: {relevant_idx}")
    # print(f"Throw idx: {throw_idx}")
    n = len(y)
    #----
    # decision_values = svm_model.decision_function(x)
    # margin = y * decision_values
    # relevant_mask = (margin > 0) 
    # relevant_idx = np.where(relevant_mask)[0]
    # throw_idx = []
    #----
    mask = np.ones(n, dtype=bool)
    mask[throw_idx] = False
    new_x, new_y, new_v = x[mask], y[mask], v[mask]
    real_to_updated = update_indices(relevant_idx, throw_idx)
    #print(real_to_updated)
    
    throw_set = set(throw_idx)
    records = []

    for i in range(n):
        is_throw = 1 if i in throw_set else 0
        is_relevant = 1 if i in relevant_idx else 0
        if is_relevant:
            updated_idx = real_to_updated[i]

            # ---- Early check at v = 0 ----
            tmp_v = new_v.copy()
            tmp_v[updated_idx] = 0.0
            model_0 = train_soft_svm(new_x, new_y, tmp_v, c, loss_key)
            alloc_0 = int(model_0.predict(x[i].reshape(1, -1))[0] == y[i])
    
            if alloc_0 == 1:
                counter = 1
                alloc = 1
                critical_v = 0
            # ------------------------------

            elif algorithm == "exact":
                alloc, critical_v, counter = exact_payments(new_x, new_y, new_v, c, updated_idx, loss=loss_key)
                counter += 1
            elif algorithm == "random":
                alloc, critical_v, counter = random_payments(new_x, new_y, new_v, c, updated_idx, loss_key, n_random, rng)
                counter += 1
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        else:
            # print(x[i].shape, svm_model.predict(x[i].reshape(1, -1)))
            alloc = int(svm_model.predict(x[i].reshape(1, -1))[0] == y[i])
            critical_v = 0
            counter = 0
        
        records.append({
            "agent": i,
            "allocation": alloc,
            "true_v": v[i],
            "is_throw": is_throw,
            "is_relevant":is_relevant,
            "alloc 0 success": int(alloc_0 == 1) if is_relevant else 0,
            "critical_v": critical_v,
            "welfare": v[i] * alloc,
            "utility": v[i] * alloc - critical_v,
            "times_alloc_called": counter + 1/n
        })

    return pd.DataFrame(records), svm_model

def one_shot_payments(X, y, v, c, loss_key, mu= ONE_SHOT_MU, rng = None):
    n = len(v)

    gamma = rng.uniform(0, 1, size=n)
    chi = np.ones(n)
    resample_mask = rng.random(n) < mu
    chi[resample_mask] = gamma[resample_mask] ** (1.0 / (1.0 - mu))
    v_mod = chi * v

    model = train_soft_svm(X, y, v_mod, c, loss=loss_key)

    allocation = (model.predict(X) == y).astype(float)
    records = []
    for i in range(n):
        factor = 1 if chi[i] == 1 else 1 - (1 / mu)
        payments_i = v[i] * allocation[i] * factor
        records.append({
            "agent": i,
            "allocation": allocation[i],
            "true_v": v[i],
            "critical_v": payments_i,
            "welfare": v[i] * allocation[i],
            "utility": v[i] * allocation[i] - payments_i,
            "times_alloc_called": 1/n
        })

    return pd.DataFrame(records), model

def random_payments(new_x, new_y, new_v, c, target_idx, loss_key, n_random=1, rng=None):
    allocs, critical_vs = [], []
    v_temp = new_v.copy()
    for _ in range(n_random):
        v_temp[target_idx] = rng.uniform(0, new_v[target_idx])
        model_temp = train_soft_svm(new_x, new_y, v_temp, c, loss=loss_key)
        pred = model_temp.predict(new_x[target_idx].reshape(1, -1))[0]
        allocs.append(int(pred == new_y[target_idx]))
        critical_vs.append(0 if pred == new_y[target_idx] else new_v[target_idx])
    alloc = np.mean(allocs)
    critical_v = np.mean(critical_vs)
    return alloc, critical_v, n_random

def exact_payments(x, y, v, c, target_idx,
                   loss="hinge", tol=1e-6, max_iter=100, plot=PLOT):
    v_mod = v.copy()
    low, high = 0.0, v[target_idx]
    counter = 0

    while high - low > tol and counter < max_iter:
        # print(low,high,counter)
        counter += 1
        mid = (low + high) / 2.0
        v_mod[target_idx] = mid

        model = train_soft_svm(x, y, v_mod, c, loss=loss)
        alloc = int(model.predict(x[target_idx].reshape(1, -1))[0] == y[target_idx])
        if plot:
            plot_svm_decision_boundary(model, x, y,
                v=v_mod, title=f"Decision Boundary @ v={v_mod[target_idx]:.5f}, target={target_idx}",
                target_idx=target_idx)

        if alloc == 1:
            high = mid
        else:
            low = mid

    # Safeguard fallback
    if alloc ==0:
        # print(f"for {target_idx}")
        # print(f"counter = {counter} and max_iter = {max_iter}")
        # print(f"high-low = {high-low} and tol = {tol}")
        mid = high 
        alloc = 1

    return alloc, mid, counter

# ----- data and run -----
def generate_gaus_data(n_pos, n_neg, mu_pos, mu_neg, sigma_pos, sigma_neg, rng=None, d=1):
    X_neg = rng.normal(loc=mu_neg, scale=sigma_neg, size=(n_neg, d)) # label -1 
    X_pos = rng.normal(loc=mu_pos, scale=sigma_pos, size=(n_pos, d)) # label 1 
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([-1 * np.ones(n_neg),np.ones(n_pos)])

    v = np.concatenate([np.ones(n_pos + n_neg)])  # generate_v_data(n_neg, n_pos, rng)
    return X, y, v

def smallest_v_for_flip(knn, x, y, v, k):
    distances, indices = knn.kneighbors(x, n_neighbors=k)

    smallest_vs = []

    for i, neigh_idx in enumerate(indices):
        # Remove the point itself (distance = 0)
        neigh_idx = neigh_idx[1:]

        # Weighted sum from neighbors only
        S = np.sum(y[neigh_idx] * v[neigh_idx])
        # print(S)

        # Check if currently opposite sign (required for flip)
        if S * y[i] < 0:
            # Solve S + y_i * v = 0  ->  v = -S * y_i
            v_star = -S * y[i]

            # Check if valid in [0, v[i]]
            if 0 <= v_star <= v[i]:
                smallest_vs.append(v_star)
            else:
                smallest_vs.append(0)
        else:
            smallest_vs.append(0)

    return np.array(smallest_vs, dtype=float)

def compute_metrics_gaus(t, trial_seed_seq):
    dfs = []
    loss_key = 'hinge'
    n_pos = n_neg = 250
    #n_pos_val = n_neg_val = 500
    sigma_pos = sigma_neg = 1
    n_configs = len(D_LIST) * len(MU_LIST)
    config_seqs = trial_seed_seq.spawn(n_configs)
    idx = 0
    c = TRAIN_SVM_PARAMS[loss_key]["C"] / (n_neg+ n_pos)

    for d in D_LIST:
        for mu_scalar in MU_LIST:
            config_ss = config_seqs[idx]
            idx += 1
            train_ss, test_ss, pay_ss = config_ss.spawn(3)
            rng_train = default_rng(train_ss)
            rng_test = default_rng(test_ss)
            rng_pay  = default_rng(pay_ss)
            
            mu = np.concatenate(([mu_scalar], np.zeros(d - 1)))
            x, y, v = generate_gaus_data(n_pos, n_neg, mu, -mu, sigma_pos, sigma_neg, rng_train, d)
            if loss_key == 'log':
                y = (y + 1) // 2
            #x_test, y_test, _ = generate_gaus_data(n_pos_val, n_neg_val, mu, -mu, sigma_pos, sigma_neg, rng_test, d)
            df_exact, svm_model = run_simulation(x, y, v, c, loss_key, algorithm="exact", rng = rng_pay)
           
            df_exact['t'] = t
            df_exact['d'] = d
            df_exact['mu'] = mu_scalar
            df_exact['k'] = np.linalg.norm(x, axis=1).max() 
            df_exact['label'] = y[df_exact['agent'].astype(int)]
            #df_exact['test_acc'] = (svm_model.predict(x_test) == y_test).astype(float)
            dfs.append(df_exact)

    return pd.concat(dfs, ignore_index=True)

def vary_c(t, trial_seed_seq):
    dfs = []
    loss_key = 'poly2'
    n_pos = n_neg = 250
    n_pos_val = n_neg_val = 500
    sigma_pos = sigma_neg = 1
    mu_scalar = 0.25
    n_configs = len(D_LIST)
    config_seqs = trial_seed_seq.spawn(n_configs)
    idx = 0

    for d in D_LIST:
        config_ss = config_seqs[idx]
        idx += 1
        train_ss, test_ss, pay_ss = config_ss.spawn(3)
        rng_train = default_rng(train_ss)
        rng_test = default_rng(test_ss)
        rng_pay  = default_rng(pay_ss)
        
        mu = np.concatenate(([mu_scalar], np.zeros(d - 1)))
        x, y, v = generate_gaus_data(n_pos, n_neg, mu, -mu, sigma_pos, sigma_neg, rng_train, d)
        k_val = np.linalg.norm(x, axis=1).max() 
        if loss_key == 'log':
                y = (y + 1) // 2
        for c in C_LIST:
            start = time.perf_counter()
            x_test, y_test, _ = generate_gaus_data(n_pos_val, n_neg_val, mu, -mu, sigma_pos, sigma_neg, rng_test, d)
            df_exact, svm_model = run_simulation(x, y, v, c, loss_key, algorithm="exact", rng = rng_pay)
            elapsed = time.perf_counter() - start
            df_exact['runtime'] = elapsed
            df_exact['t'] = t
            df_exact['d'] = d
            df_exact["c"] = c
            # df_exact['mu'] = mu_scalar
            df_exact['k'] = k_val
            df_exact['label'] = y[df_exact['agent'].astype(int)]

            df_exact['test_acc'] = (svm_model.predict(x_test) == y_test).astype(float)
            dfs.append(df_exact)

    return pd.concat(dfs, ignore_index=True)

def vary_m(t, trial_seed_seq):
    dfs = []
    mu = np.array([2, 0])
    n_pos_val = n_neg_val = 500
    sigma_pos = sigma_neg = 1.25
    n_configs = len(K_LIST) * len(M_LIST)
    config_seqs = trial_seed_seq.spawn(n_configs)
    idx = 0

    for m in M_LIST:
        n_pos = n_neg = m
        n = 2 * m
        config_ss = config_seqs[idx]
        idx += 1
        train_ss, test_ss = config_ss.spawn(2)
        rng_train = default_rng(train_ss)
        rng_test = default_rng(test_ss)

        x, y, v = generate_gaus_data(n_pos, n_neg, mu, -mu, sigma_pos, sigma_neg, rng_train, D)
        x_test, y_test, _ = generate_gaus_data(n_pos_val, n_neg_val, mu, -mu, sigma_pos, sigma_neg, rng_test, D)
        
def knn_metrics(t, trial_seed_seq):
    dfs = []
    mu = np.array([2, 0])
    n_pos_val = n_neg_val = 500
    sigma_pos = sigma_neg = 1.25
    n_configs = len(K_LIST) * len(M_LIST)
    config_seqs = trial_seed_seq.spawn(n_configs)
    idx = 0

    for m in M_LIST:
        n_pos = n_neg = m
        n = 2 * m
        config_ss = config_seqs[idx]
        idx += 1
        train_ss, test_ss = config_ss.spawn(2)
        rng_train = default_rng(train_ss)
        rng_test = default_rng(test_ss)

        x, y, v = generate_gaus_data(n_pos, n_neg, mu, -mu, sigma_pos, sigma_neg, rng_train, D)
        x_test, y_test, _ = generate_gaus_data(n_pos_val, n_neg_val, mu, -mu, sigma_pos, sigma_neg, rng_test, D)
        
        for k in K_LIST:
            knn_model = train_knn(x, y, v, k, plot = PLOT)
            df = pd.DataFrame({'agent': range(n)})
            df['k_nbrs'] = k
            df['label'] = y[df['agent'].astype(int)]
            df['true_v'] = v[df['agent'].astype(int)]
            df['m_total'] = n_pos + n_neg
            df['d'] = D
            df['t'] = t
            df['times_alloc_called'] = 1/n
            df['test_acc'] = (predict_knn(knn_model, x_test, y, v, k) == y_test).astype(int).mean()
            preds = predict_knn(knn_model, x, y, v, k)
            df['allocation'] = (preds == y).astype(int)
            df['welfare'] = df['allocation'] * df['true_v']
            df['critical_v'] = smallest_v_for_flip(knn_model, x, y, v, k)
            df['utility'] = df['allocation'] * df['true_v'] - df['critical_v']
            dfs.append(df)

    df_t = pd.concat(dfs, ignore_index=True)
    df_t.to_csv(f"knn_res_2/knn_{t}.csv", index = False)
