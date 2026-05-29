import os

# 🔥 CRITICAL: prevent hidden threading from BLAS / sklearn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import time
from numpy.random import default_rng
from joblib import Parallel, delayed

from config import *
from main import process_agent_pay, train_soft_svm, get_relevant_throw_idx, update_indices


# =========================
# DATA FUNCTION (FIXED: pass df explicitly)
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
# SINGLE EXPERIMENT (ONLY PARALLEL HERE)
# =========================
def run_single_loss(x, y, v, x_test, y_test, rng_pay, m, t, c, loss):

    file_path = f"run_t/m={m}_t={t}_c={c}_{loss}_p.csv"
    if os.path.exists(file_path):
        print(f"Skipping: {file_path} exists", flush=True)
        return

    start = time.perf_counter()

    # Train base model
    svm_model = train_soft_svm(x, y, v, c, loss)
    base_preds = svm_model.predict(x)

    relevant_idx, throw_idx = get_relevant_throw_idx(
        svm_model, x, y, v, c, loss
    )

    print(f"[t={t}] Relevant idx: {len(relevant_idx)}", flush=True)

    n = len(y)

    mask = np.ones(n, dtype=bool)
    mask[throw_idx] = False
    new_x, new_y, new_v = x[mask], y[mask], v[mask]
    real_to_updated = update_indices(relevant_idx, throw_idx)

    throw_set = set(throw_idx)
    relevant_set = set(relevant_idx)

    records = Parallel(n_jobs=128, backend="loky", mmap_mode="r")(
        delayed(process_agent_pay)(
            i, x, y, v,
            new_x, new_y, new_v,
            svm_model, base_preds,
            relevant_set, throw_set,
            real_to_updated,
            c, loss
        )
        for i in range(n)
    )

    elapsed = time.perf_counter() - start

    df_exact = pd.DataFrame(records)
    df_exact['runtime'] = elapsed
    df_exact["m_total"] = 2 * m
    df_exact['loss'] = loss
    df_exact['t'] = t
    df_exact["c"] = c
    df_exact['label'] = y[df_exact['agent'].astype(int)]
    df_exact['test_acc'] = np.mean(svm_model.predict(x_test) == y_test)

    df_exact.to_csv(file_path, index=False)

    print(f"[t={t}] DONE in {elapsed:.2f}s", flush=True)


# =========================
# BUILD ALL TASKS (NO PARALLEL)
# =========================
def build_all_tasks(df):
    all_tasks = []

    for i in range(N_TRIALS):
        trial_seed_seq = TRIAL_SS[i]

        n_test = 1000
        ms = [4000]
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

            cs = [0.1, 1, 10, 100]

            for c in cs:
                for loss in ['hinge']:
                    all_tasks.append(
                        (x, y, v, x_test, y_test, rng_pay, m, i, c, loss)
                    )

    return all_tasks


# =========================
# MAIN ENTRY (SAFE FOR MULTIPROCESSING)
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