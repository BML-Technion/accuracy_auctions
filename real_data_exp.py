import os

# 🔥 Prevent hidden threading (important when using joblib / sklearn)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import time
from numpy.random import default_rng

from config import *
from main import (
    train_soft_svm,
    get_relevant_throw_idx,
    update_indices,
    exact_payments,
    random_payments,
    one_shot_payments
)

# =========================
# DATA FUNCTION
# =========================
def get_data(df, m, n_test, rng_train, rng_test):
    target_col = 'PINCP'
    weight_col = 'ord__WKHP'

    remove = ['original_weight', 'ord__WKHP', 'PINCP']
    keep = [col for col in df.columns if col not in remove]

    pos_df = df[df[target_col] == 1]
    neg_df = df[df[target_col] == -1]

    pos_idx = rng_train.choice(pos_df.index, size=m, replace=False)
    neg_idx = rng_train.choice(neg_df.index, size=m, replace=False)

    train_idx = np.concatenate([pos_idx, neg_idx])
    train = df.loc[train_idx]

    x_train = train[keep].to_numpy()
    y_train = train[target_col].to_numpy()
    v_train = train[weight_col].to_numpy()

    remaining_idx = df.index.difference(train_idx)
    test_idx = rng_test.choice(remaining_idx, size=n_test, replace=False)
    test = df.loc[test_idx]

    x_test = test[keep].to_numpy()
    y_test = test[target_col].to_numpy()
    v_test = test[weight_col].to_numpy()

    return x_train, y_train, v_train, x_test, y_test, v_test


# =========================
# SIMULATION FUNCTION (YOUR LOGIC)
# =========================
def run_simulation(x, y, v, c, loss_key, algorithm="exact", n_random=1, rng=None):

    if algorithm == "one-shot":
        return one_shot_payments(x, y, v, c, loss_key, ONE_SHOT_MU, rng)

    svm_model = train_soft_svm(x, y, v, c, loss_key)

    relevant_idx, throw_idx = get_relevant_throw_idx(
        svm_model, x, y, v, c, loss_key
    )

    n = len(y)

    mask = np.ones(n, dtype=bool)
    mask[throw_idx] = False

    new_x, new_y, new_v = x[mask], y[mask], v[mask]
    real_to_updated = update_indices(relevant_idx, throw_idx)

    throw_set = set(throw_idx)
    relevant_set = set(relevant_idx)

    records = []

    for i in range(n):

        is_throw = int(i in throw_set)
        is_relevant = int(i in relevant_set)

        alloc_0 = 0  # default

        if is_relevant:

            updated_idx = real_to_updated[i]

            # ---- Early check at v = 0 ----
            tmp_v = new_v.copy()
            tmp_v[updated_idx] = 0.0

            model_0 = train_soft_svm(new_x, new_y, tmp_v, c, loss_key)
            alloc_0 = int(
                model_0.predict(x[i].reshape(1, -1))[0] == y[i]
            )

            if alloc_0 == 1:
                alloc = 1
                critical_v = 0
                counter = 1

            elif algorithm == "exact":
                alloc, critical_v, counter = exact_payments(
                    new_x, new_y, new_v, c, updated_idx, loss=loss_key
                )
                counter += 1

            elif algorithm == "random":
                alloc, critical_v, counter = random_payments(
                    new_x, new_y, new_v, c, updated_idx,
                    loss_key, n_random, rng
                )
                counter += 1

            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        else:
            alloc = int(
                svm_model.predict(x[i].reshape(1, -1))[0] == y[i]
            )
            critical_v = 0
            counter = 0

        records.append({
            "agent": i,
            "allocation": alloc,
            "true_v": v[i],
            "is_throw": is_throw,
            "is_relevant": is_relevant,
            "alloc_0_success": int(alloc_0 == 1) if is_relevant else 0,
            "critical_v": critical_v,
            "welfare": v[i] * alloc,
            "utility": v[i] * alloc - critical_v,
            "times_alloc_called": counter + 1/n
        })

    return pd.DataFrame(records), svm_model


# =========================
# SINGLE EXPERIMENT
# =========================
def run_single_loss(x, y, v, x_test, y_test, rng_pay, m, t, c, loss):

    file_path = f"run_t/m={m}_t={t}_c={c}_{loss}_p.csv"

    if os.path.exists(file_path):
        print(f"Skipping: {file_path} exists", flush=True)
        return

    start = time.perf_counter()

    # 🔥 USE NEW SIMULATION
    df_exact, svm_model = run_simulation(
        x, y, v,
        c=c,
        loss_key=loss,
        algorithm="exact",
        rng=rng_pay
    )

    elapsed = time.perf_counter() - start

    df_exact['runtime'] = elapsed
    df_exact["m_total"] = 2 * m
    df_exact['loss'] = loss
    df_exact['t'] = t
    df_exact["c"] = c
    df_exact['label'] = y[df_exact['agent'].astype(int)]
    df_exact['test_acc'] = np.mean(
        svm_model.predict(x_test) == y_test
    )

    df_exact.to_csv(file_path, index=False)

    print(f"[t={t}] DONE in {elapsed:.2f}s", flush=True)


# =========================
# BUILD TASKS
# =========================
def build_all_tasks(df):

    all_tasks = []

    for i in range(1):

        trial_seed_seq = TRIAL_SS[i]

        n_test = 1000
        ms = [2000, 5000]

        config_seqs = trial_seed_seq.spawn(len(ms))

        for idx, m in enumerate(ms):

            config_ss = config_seqs[idx]
            train_ss, test_ss, pay_ss = config_ss.spawn(3)

            rng_train = default_rng(train_ss)
            rng_test = default_rng(test_ss)
            rng_pay  = default_rng(pay_ss)

            x, y, v, x_test, y_test, _ = get_data(
                df, m, n_test, rng_train, rng_test
            )

            cs = [1]

            for c in cs:
                for loss in ['log', 'squared_hinge','hinge']:
                    all_tasks.append(
                        (x, y, v, x_test, y_test,
                         rng_pay, m, i, c, loss)
                    )

    return all_tasks


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    df = pd.read_csv("NJ_data_with_noise.csv")

    df['ord__WKHP'] = np.where(
        df['ord__WKHP'] < 40, 1,
        np.where(df['ord__WKHP'] > 40, 5, 2)
    )

    df['PINCP'] = np.where(df['PINCP'] == 0, -1, 1)

    all_tasks = build_all_tasks(df)

    print(f"Total tasks: {len(all_tasks)}", flush=True)

    for task in all_tasks:
        run_single_loss(*task)