import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

def check_equivalence(X, y, C, v, tol=1e-6):
    """
    Check if (C, v) is equivalent to (C', v') where
    v' = v / sum(v) and C' = C * sum(v).
    """
    # Normalized weights and scaled C
    v_prime = v / v.sum()
    C_prime = C * v.sum()

    # Train first model
    model1 = SVC(kernel="linear", C=C, random_state=0)
    model1.fit(X, y, sample_weight=v)

    # Train second model
    model2 = SVC(kernel="linear", C=C_prime, random_state=0)
    model2.fit(X, y, sample_weight=v_prime)

    # Extract parameters
    w1, b1 = model1.coef_.ravel(), model1.intercept_[0]
    w2, b2 = model2.coef_.ravel(), model2.intercept_[0]

    # Compare
    same_w = np.allclose(w1, w2, atol=tol)
    same_b = np.allclose(b1, b2, atol=tol)

    return same_w and same_b, {
        "same_w": same_w,
        "same_b": same_b,
        "diff_w_norm": np.linalg.norm(w1 - w2),
        "diff_b": abs(b1 - b2),
        "w1": w1,
        "w2": w2,
        "b1": b1,
        "b2": b2,
        "C_prime": C_prime,
        "v_prime_sum": v_prime.sum()
    }

#acc 0.8 + payments
# Generate or load data
def generate_data(n):
    # Generate synthetic data
    x, y = make_blobs(n_samples=n, centers=2, random_state=0, cluster_std=1.5)  #1.5
    y = np.where(y == 0, -1, y)
    v = np.abs(x[:, 0]) * 10 + np.random.normal(0, 0.1, n)
    return x, y, v

n=30
x, y, v = generate_data(n)
ok, details = check_equivalence(x, y, C=1.0, v=v)
print(ok)
print(details)




#from main main
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



# def train_soft_svm(x, y, v=None, c =1):
#     model = SVC(kernel='linear', C=c, random_state=0)
#     if v is None:
#         model.fit(x, y)
#     else:
#         model.fit(x, y, sample_weight=v)
#     return model
