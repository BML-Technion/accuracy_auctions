import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import default_rng
from main import *

# =====================================================
# CONFIG
# =====================================================

T = 20
BASE_SEED = 42

M_LIST = np.linspace(20, 201, 19).astype(int) 
K_LIST = [2,4,6,8] 

LOSS_KEY = "hinge"

D = 2

MU = np.array([0.25, 0])
SIGMA = 1


# =====================================================
# KNN: NUMBER OF PAYERS
# =====================================================

def run_knn_num_payers(x, y, v, k):

    knn_model = train_knn(x, y, v, k, plot=False)

    critical_v = smallest_v_for_flip(knn_model, x, y, v, k)

    # number of agents with positive payment
    num_payers = np.sum(critical_v > 0)

    return num_payers


# =====================================================
# ONE FULL TRIAL
# =====================================================

def run_trial(t, seed_seq):

    dfs = []

    config_seq = seed_seq.spawn(len(M_LIST))

    idx = 0

    for m in M_LIST:

        print(f"  m = {m}")

        rng = default_rng(config_seq[idx])
        idx += 1

        n_pos = n_neg = m

        # -------------------------------------------------
        # Generate Gaussian data
        # -------------------------------------------------

        x, y, v = generate_gaus_data(
            n_pos=n_pos,
            n_neg=n_neg,
            mu_pos=MU,
            mu_neg=-MU,
            sigma_pos=SIGMA,
            sigma_neg=SIGMA,
            rng=rng,
            d=D
        )

        # -------------------------------------------------
        # SVM exact payments
        # -------------------------------------------------

        c = 1.5

        df_svm, svm_model = run_simulation(
            x,
            y,
            v,
            c=c,
            loss_key=LOSS_KEY,
            algorithm="exact",
            rng=rng
        )

        svm_num_payers = (df_svm["critical_v"] > 0).sum()

        svm_acc = (svm_model.predict(x) == y).mean()

        print(f"    SVM accuracy = {svm_acc:.4f}")
        print(f"    SVM num payers = {svm_num_payers}")

        dfs.append({
            "t": t,
            "m": m,
            "model": "SVM-exact",
            "num_payers": svm_num_payers
        })

        # -------------------------------------------------
        # KNN models
        # -------------------------------------------------

        for k in K_LIST:

            num_payers = run_knn_num_payers(x, y, v, k)

            knn_model = train_knn(x, y, v, k, plot=False)

            preds = predict_knn(knn_model, x, y, v, k)

            knn_acc = (preds == y).mean()

            print(f"    KNN k={k} accuracy = {knn_acc:.4f}")
            print(f"    KNN k={k} num payers = {num_payers}")

            dfs.append({
                "t": t,
                "m": m,
                "model": f"KNN-k={k}",
                "num_payers": num_payers
            })

    return pd.DataFrame(dfs)


# =====================================================
# MAIN EXPERIMENT
# =====================================================

def run_experiment():

    master_seed = np.random.SeedSequence(BASE_SEED)

    trial_seeds = master_seed.spawn(T)

    all_results = []

    for t in range(T):

        print("=" * 60)
        print(f"Running trial {t+1}/{T}")
        print("=" * 60)

        df_t = run_trial(t, trial_seeds[t])

        all_results.append(df_t)

    return pd.concat(all_results, ignore_index=True)


# =====================================================
# RUN EXPERIMENT
# =====================================================

df = run_experiment()

print("\nFinal dataframe:")
#print(df.head())


# =====================================================
# AGGREGATE OVER TRIALS
# =====================================================

summary = (
    df.groupby(["m", "model"])["num_payers"]
    .mean()
    .reset_index()
)

print("\nSummary:")
#print(summary.head())

summary.to_csv("summary_num_payers_2.csv", index=False)


# =====================================================
# PLOT
# =====================================================
summary = pd.read_csv("summary_num_payers_2.csv")
plt.figure(figsize=(6, 4))

for model in summary["model"].unique():

    sub = summary[summary["model"] == model]

    plt.plot(
        sub["m"],
        sub["num_payers"],
        marker="o",
        label=model
    )

#plt.xscale("log")
plt.xlim(20, 201)
plt.ylim(0, 100)
plt.xlabel("m (sample size per class)", fontsize=12)
plt.ylabel("Number of payers", fontsize=12)
plt.title("Number of Payers vs Dataset Size", fontsize=13)

plt.legend(loc = 'right')
plt.grid(True)

plt.savefig("num_payers_plot.png")
plt.savefig("num_payers_plot.pdf")

plt.show()