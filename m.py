import pandas as pd
import numpy as np
from utils import *
from main import *
from data_analysis import *

def generate_1d_data(n, seed=420):
    rng = np.random.default_rng(seed)
    X_neg = rng.uniform(-1, 0.25, size=(int(n/2), 1))
    y_neg = -1 * np.ones(len(X_neg))
    X_pos = rng.uniform(-0.25, 1, size=(int(n/2), 1))
    y_pos = np.ones(len(X_pos))
    v = np.ones(n)
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([y_neg, y_pos])
    return X, y, v


rng_master = np.random.RandomState(42)
n_list = [0]
ns = np.arange(20, 500, 10).astype(int)
paying_agents = []
T = 5000
for t in range(T):
    rng = np.random.RandomState(rng_master.randint(2**31 - 1))
    X_new, y_new, v_new = generate_1d_data(10, rng)
    for current_n in ns:
        print(f' running for {current_n}')
        X_1, y_1, _ = generate_1d_data(10, seed=current_n * (t+1))

        # Append to original dataset
        X_new = np.vstack([X_new, X_1])
        y_new = np.concatenate([y_new, y_1])
        v_new = np.ones(current_n)
        print(f'running for {len(y_new)}')

        summary_df, boundary, _ = main_exact(X_new, y_new, v_new)
        df_payments = summary_df[summary_df["critical_v"] > 0]
        for agent in df_payments["agent"].unique():
            distance = abs(X_new[agent][0] - boundary)

            paying_agents.append({
                't': t,
                "n": current_n,
                "agent": agent,
                "agent_location": float(X_new[agent][0]),
                "agent_label": y_new[agent],
                "critical_v": float(df_payments[df_payments["agent"] == agent]['critical_v']),
                "boundary": float(boundary),
                "distance": float(distance)
            })
        #print(f'{paying_agents}')

# build DataFrame at the end
df = pd.DataFrame(paying_agents)
df.to_csv('payments_n.csv', index = False)


