from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import pandas as pd
from randomness import *
from simulation_exact import *
from utils import *
from data_analysis import *

def train_soft_svm(x, y, v=None, c =1):
    # print(f'c is {c}')
    model = SVC(kernel='linear', C=c, random_state=0)
    if v is None:
        model.fit(x, y)
    else:
        model.fit(x, y, sample_weight=v)
    return model

#acc 0.8 + payments
# Generate or load data
def generate_data(n):
    # Generate synthetic data
    x, y = make_blobs(n_samples=n, centers=2, random_state=0, cluster_std=1.5)  #1.5
    y = np.where(y == 0, -1, y)
    v = np.abs(x[:, 0]) * 10 + np.random.normal(0, 0.1, n)
    # v = (v - np.min(v)) / (np.max(v) - np.min(v))
    return x, y, v


def main_random(n, x, y, v):
    T = 50000
    records = []
    for mu in np.linspace(0.7, 0.75, 1):
        for run in range(T):
            new_b, allocation, payments, chi = mechanism_3(x, y, v, mu, train_soft_svm)
            for i in range(n):
                records.append({
                    'run': run,
                    'agent': i,
                    'true_b': v[i], ##assumin g truthfulness
                    'new_b': new_b[i],
                    'allocation': allocation[i],
                    'payment': payments[i],
                    'chi': chi[i],
                    'mu': mu,
                    'welfare': v[i] * allocation[i], #- payments[i],
                    'utility': v[i] * allocation[i]  - payments[i],
                })

    # === Results DataFrame ===
    df = pd.DataFrame(records)
    df.to_csv(f"random_simulation_n={n}.csv", index=False)
    IR(df)
    print(f'check IR in expectation')
    IR(df, exp=True)
    return df


def main_exact(x, y, v,  plot = False):
    records = []

    # Train once
    svm_model = train_soft_svm(x, y, v)
    w = svm_model.coef_[0]
    b = svm_model.intercept_[0]

    # print(f'boundary={-b / w}')

    train_acc = svm_model.score(x, y)
    print(f'The accuracy is: {train_acc}')

    # Decision values: y * (w·x + b)
    decision_values = y * (x @ w + b)

    # Support vectors
    support_vector_indices = svm_model.support_

    # Keep only those that are correctly classified
    correct_sv_indices = support_vector_indices[decision_values[support_vector_indices] > 0]

    #print(f'going over {correct_sv_indices} points')
    for target_idx in correct_sv_indices:
        # print(f"\n=== Processing target {target_idx} ===")
        critical_v, alloc, _ = compute_critical_bid(x, y, v, target_idx, train_soft_svm, plot = plot)
        records.append({
                "agent": target_idx,
                "allocation": alloc,
                "true_v": v[target_idx],
                "critical_v": critical_v,
                'welfare':  v[target_idx] *  alloc, #- critical_v ,
                'utility': v[target_idx] * alloc - critical_v ,
            })


    df = pd.DataFrame(records)
    # non_zero_df = df[df['critical_v'] != 0]
    # non_zero_indices = non_zero_df['agent']
    #
    # print(f"Number of non-zero payments: {len(non_zero_indices)}")
    # print(f"Indices with non-zero payments: {list(non_zero_indices)}")

    # # Print critical_v and true_v for those indices
    # print("Critical_v and True_v values:")
    # print(non_zero_df[['agent', 'critical_v', 'true_v']])

    return df, -b / w, train_acc

def main_exact_in_on_margin(x, y, v, tol = 1e-2):
    print(f'take out points that are out of margin')
    model = train_soft_svm(x, y, v)

    decision_values = model.decision_function(x)
    margin_distances = y * decision_values
    inside_or_on_margin_mask = margin_distances <= 1 + tol
    # Filter x, y, v to keep only those points
    x_filtered = x[inside_or_on_margin_mask]
    y_filtered = y[inside_or_on_margin_mask]
    v_filtered = v[inside_or_on_margin_mask]
    n_filtered = len(x_filtered)

    # Get indices of points outside the margin
    in_on_margin_indices = np.where(margin_distances <= 1 + tol)[0]
    print("Indices inside/on the margin:", in_on_margin_indices)

    print(f'{n_filtered} left out of {n}')
    df = main_exact(n_filtered, x_filtered, y_filtered, v_filtered)
    return df

def drop_one_by_one(x, y, v):
    print('Removing points outside the margin one-by-one (farthest first)')

    model = train_soft_svm(x, y, v)
    decision_values = model.decision_function(x)
    margin_distances = y * decision_values

    # Identify violations: points with margin_dist > 1
    outside_indices = np.where(margin_distances > 1 + 1e-2)[0]
    violations = margin_distances[outside_indices] - 1

    # Sort indices by margin violation (farthest from margin first)
    sorted_outside = outside_indices[np.argsort(-violations)]
    print("Sorted indices to remove:", sorted_outside)

    # Store data as a dict with index keys
    data_dict = {i: (x[i], y[i], v[i]) for i in range(len(x))}

    # Remove one-by-one based on sorted indices
    for idx in sorted_outside:
        print(f"Removing index {idx} with margin distance {margin_distances[idx]:.4f}")
        if idx in data_dict:
            del data_dict[idx]
            # Reconstruct filtered arrays
            remaining_indices = sorted(data_dict.keys())
            x_filtered = np.array([data_dict[i][0] for i in remaining_indices])
            y_filtered = np.array([data_dict[i][1] for i in remaining_indices])
            v_filtered = np.array([data_dict[i][2] for i in remaining_indices])
            n_filtered = len(x_filtered)
            print(f'{n_filtered} left out of {n}')
            df = main_exact(n_filtered, x_filtered, y_filtered, v_filtered)
            IR(df)
            exact_min_sum = df['critical_v'].sum()
            print(f'exact {n_filtered} sum: {exact_min_sum}')
    return True

def lvl_1(x,y,v):
    T = 5000
    allocation_model = train_soft_svm(x, y, v)
    pred = np.array(allocation_model.predict(x))
    allocations = (y == pred).astype(float)
    all_payments = []

    for _ in range(T):
        payments = mechanism_1(x, y, v, train_soft_svm, allocations)
        all_payments.append(payments)

    # Stack all payments (shape: [100, n_agents]) and take mean over axis 0
    all_payments = np.vstack(all_payments)
    avg_payments = np.mean(all_payments, axis=0)
    var_payments = np.var(all_payments, axis=0)

    return avg_payments, var_payments, np.sum(all_payments)


# Sample version of main_random for a fixed mu, and returns DataFrame
def run_for_T(n, x, y, v, T, mu=0.7):
    records = []
    for run in range(T):
        new_b, allocation, payments, chi = mechanism_3(x, y, v, mu, train_soft_svm)
        for i in range(n):
            records.append({
                'run': run,
                'agent': i,
                'payment': payments[i],
                'T': T
            })
    df = pd.DataFrame(records)
    df.to_csv(f"check_sd.csv", index=False)
    return df

#
# def payment_asf_x(x,y,v, target_idx):
#     records = []
#
#     # Train once
#     svm_model = train_soft_svm(x, y, v)
#     w = svm_model.coef_[0]
#     b = svm_model.intercept_[0]
#
#
#     # print(f"\n=== Processing target {target_idx} ===")
#     critical_v, alloc, _ = compute_critical_bid(x, y, v, target_idx, train_soft_svm)
#     records.append({
#         "agent": target_idx,
#         "allocation": alloc,
#         "true_v": v[target_idx],
#         "critical_v": critical_v,
#         'welfare': v[target_idx] * alloc,  # - critical_v ,
#         'utility': v[target_idx] * alloc - critical_v,
#     })
#
#     df = pd.DataFrame(records)
#     non_zero_df = df[df['critical_v'] != 0]
#     non_zero_indices = non_zero_df.index
#
#     print(f"Number of non-zero payments: {len(non_zero_indices)}")
#     print(f"Indices with non-zero payments: {list(non_zero_indices)}")
#
#     # Print critical_v and true_v for those indices
#     print("Critical_v and True_v values:")
#     print(non_zero_df[['agent', 'critical_v', 'true_v']])
#
#     return df



if __name__ == '__main__':
    n = 30
    np.random.seed(42)
    x, y, v = generate_data(n)
    # v = np.ones(n)
    print(f'begin exact simulation')
    df_exact, boundary, train_acc = main_exact(x, y, v)
    df_exact.to_csv(f"exact_simulation_n={n}.csv", index=False)
    #IR(df_exact)
    exact_sum = df_exact['critical_v'].sum()
    exact_util = df_exact['utility'].sum()
    print(f'exact sum: {exact_sum}')
    print(f'exact util: {exact_util}')

    # Main loop for multiple T values
    # T_values =[50000] #[100, 1000, 5000, 10000, 25000, 50000] # From 100 to ~50000 in log scale
    # all_std_records = []

    # for T in T_values:
    #     print(f'Running T = {T}')
    #     df = run_for_T(n=n, x=x, y=y, v=v, T=T, mu=0.7)
    #
    #     # Compute std of payment for each agent
    #     agent_std = df.groupby('agent')['payment'].std().reset_index()
    #     agent_std['T'] = T
    #     all_std_records.append(agent_std)
    #
    # # Combine all results
    # df_std = pd.concat(all_std_records)

    # # Plotting
    # plt.figure(figsize=(12, 6))
    # df_std['T_str'] = df_std['T'].astype(str)  # For better tick labels
    # df_std.boxplot(column='payment', by='T_str')

    # plt.title("Distribution of Std Dev of Agent Payments vs T (mu=0.7)")
    # plt.suptitle("")
    # plt.xlabel("T (number of runs)")
    # plt.ylabel("Std Dev of Payments per Agent")
    # plt.grid(True)
    # plt.show()


    # model = train_soft_svm(x, y, v)
    #
    # # Predict on training data
    # pred = model.predict(x)
    #
    # # Compute and print accuracy
    # acc = accuracy_score(y, pred)
    # print(f"Model accuracy: {acc:.4f}")
    #
    #
    # print(f'begin exact simulation')
    # df_exact = main_exact(n, x, y, v)
    # df_exact.to_csv(f"exact_simulation_n={n}.csv", index=False)
    # IR(df_exact)
    # exact_sum = df_exact['critical_v'].sum()
    # exact_util = df_exact['utility'].sum()
    # print(f'exact sum: {exact_sum}')
    # print(f'exact util: {exact_util}')

    # avg_payments, var_payments , mean_payment = lvl_1(x, y, v)
    # print(f"Mean total payment: {mean_payment:.4f}")

    # Assuming avg_payments is a 1D array of length equal to len(df_exact)
    # df_exact["avg_payment"] = avg_payments
    # df_exact["payment_minus_critical"] = df_exact["avg_payment"] - df_exact["critical_v"]
    # df_exact["var_payment"] = var_payments

    # # (Optional) Display as requested
    # print("Average payments per agent (compared to critical values):")
    # print(df_exact[["critical_v", "avg_payment", "payment_minus_critical", 'var_payment']])


    # print(f'begin exact simulation with no out of margin points')
    # df_min_exact = main_exact_in_on_margin(x, y, v, tol = 0.6)
    # df_min_exact.to_csv(f"exact_min_simulation_n={n}.csv", index=False)
    # IR(df_min_exact)
    # exact_min_sum = df_min_exact['critical_v'].sum()
    # print(f'exact min sum: {exact_min_sum}')

    # print(f'begin random simulation')
    # df_random = main_random(n, x, y, v)
    # # exact_sum = 1.198636
    # print(f'begin Analysis')
    # analyze_utilities_and_payments(df_random,df_exact)
    # analyze_payments_by_mu(df_random,df_exact)
    # print(f'begin one by one drop')
    # drop_one_by_one(x, y, v)
