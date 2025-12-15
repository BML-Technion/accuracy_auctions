# import numpy as np
# from sklearn.svm import SVC
# from sklearn.datasets import make_blobs

# def check_equivalence(X, y, C, v, tol=1e-6):
#     """
#     Check if (C, v) is equivalent to (C', v') where
#     v' = v / sum(v) and C' = C * sum(v).
#     """
#     # Normalized weights and scaled C
#     v_prime = v / v.sum()
#     C_prime = C * v.sum()

#     # Train first model
#     model1 = SVC(kernel="linear", C=C, random_state=0)
#     model1.fit(X, y, sample_weight=v)

#     # Train second model
#     model2 = SVC(kernel="linear", C=C_prime, random_state=0)
#     model2.fit(X, y, sample_weight=v_prime)

#     # Extract parameters
#     w1, b1 = model1.coef_.ravel(), model1.intercept_[0]
#     w2, b2 = model2.coef_.ravel(), model2.intercept_[0]

#     # Compare
#     same_w = np.allclose(w1, w2, atol=tol)
#     same_b = np.allclose(b1, b2, atol=tol)

#     return same_w and same_b, {
#         "same_w": same_w,
#         "same_b": same_b,
#         "diff_w_norm": np.linalg.norm(w1 - w2),
#         "diff_b": abs(b1 - b2),
#         "w1": w1,
#         "w2": w2,
#         "b1": b1,
#         "b2": b2,
#         "C_prime": C_prime,
#         "v_prime_sum": v_prime.sum()
#     }

# #acc 0.8 + payments
# # Generate or load data
# def generate_data(n):
#     # Generate synthetic data
#     x, y = make_blobs(n_samples=n, centers=2, random_state=0, cluster_std=1.5)  #1.5
#     y = np.where(y == 0, -1, y)
#     v = np.abs(x[:, 0]) * 10 + np.random.normal(0, 0.1, n)
#     return x, y, v

# n=30
# x, y, v = generate_data(n)
# ok, details = check_equivalence(x, y, C=1.0, v=v)
# print(ok)
# print(details)


# def check_equivalence(X, y, C, v, model1, model2, tol=1e-6):
#     """
#     Check if (C, v) is equivalent to (C', v') where
#     v' = v / sum(v) and C' = C * sum(v).
#     """
#     # Normalized weights and scaled C
#     v_prime = v / v.sum()
#     C_prime = C * v.sum()

#     # # Train first model
#     # model1 = SVC(kernel="linear", C=C, random_state=0)
#     # model1.fit(X, y, sample_weight=v)

#     # # Train second model
#     # model2 = SVC(kernel="linear", C=C_prime, random_state=0)
#     # model2.fit(X, y, sample_weight=v_prime)

#     # Extract parameters
#     w1, b1 = model1.coef_.ravel(), model1.intercept_[0]
#     w2, b2 = model2.coef_.ravel(), model2.intercept_[0]

#     # Compare
#     same_w = np.allclose(w1, w2, atol=tol)
#     same_b = np.allclose(b1, b2, atol=tol)

#     # Compare support vectors
#     sv_idx1, sv_idx2 = model1.support_, model2.support_
#     same_sv_idx = np.array_equal(sv_idx1, sv_idx2)

#     # Compare dual coefficients (alphas)
#     alpha1, alpha2 = model1.dual_coef_, model2.dual_coef_
#     same_alpha = np.allclose(alpha1, alpha2, atol=tol)

#     return same_w and same_b and same_sv_idx and same_alpha, {
#         "same_w": same_w,
#         "same_b": same_b,
#         "same_sv_idx": same_sv_idx,
#         "same_alpha": same_alpha,
#         "diff_w_norm": np.linalg.norm(w1 - w2),
#         "diff_b": abs(b1 - b2),
#         "diff_alpha": np.linalg.norm(alpha1 - alpha2),
#         "support_idx_1": sv_idx1,
#         "support_idx_2": sv_idx2,
#         "num_support_1": len(sv_idx1),
#         "num_support_2": len(sv_idx2),
#         "C_prime": C_prime,
#         "v_prime_sum": v_prime.sum()
#     }

# #from main main
#     # Main loop for multiple T values
#     # T_values =[50000] #[100, 1000, 5000, 10000, 25000, 50000] # From 100 to ~50000 in log scale
#     # all_std_records = []

#     # for T in T_values:
#     #     print(f'Running T = {T}')
#     #     df = run_for_T(n=n, x=x, y=y, v=v, T=T, mu=0.7)
#     #
#     #     # Compute std of payment for each agent
#     #     agent_std = df.groupby('agent')['payment'].std().reset_index()
#     #     agent_std['T'] = T
#     #     all_std_records.append(agent_std)
#     #
#     # # Combine all results
#     # df_std = pd.concat(all_std_records)

#     # # Plotting
#     # plt.figure(figsize=(12, 6))
#     # df_std['T_str'] = df_std['T'].astype(str)  # For better tick labels
#     # df_std.boxplot(column='payment', by='T_str')

#     # plt.title("Distribution of Std Dev of Agent Payments vs T (mu=0.7)")
#     # plt.suptitle("")
#     # plt.xlabel("T (number of runs)")
#     # plt.ylabel("Std Dev of Payments per Agent")
#     # plt.grid(True)
#     # plt.show()


#     # model = train_soft_svm(x, y, v)
#     #
#     # # Predict on training data
#     # pred = model.predict(x)
#     #
#     # # Compute and print accuracy
#     # acc = accuracy_score(y, pred)
#     # print(f"Model accuracy: {acc:.4f}")
#     #
#     #
#     # print(f'begin exact simulation')
#     # df_exact = main_exact(n, x, y, v)
#     # df_exact.to_csv(f"exact_simulation_n={n}.csv", index=False)
#     # IR(df_exact)
#     # exact_sum = df_exact['critical_v'].sum()
#     # exact_util = df_exact['utility'].sum()
#     # print(f'exact sum: {exact_sum}')
#     # print(f'exact util: {exact_util}')

#     # avg_payments, var_payments , mean_payment = lvl_1(x, y, v)
#     # print(f"Mean total payment: {mean_payment:.4f}")

#     # Assuming avg_payments is a 1D array of length equal to len(df_exact)
#     # df_exact["avg_payment"] = avg_payments
#     # df_exact["payment_minus_critical"] = df_exact["avg_payment"] - df_exact["critical_v"]
#     # df_exact["var_payment"] = var_payments

#     # # (Optional) Display as requested
#     # print("Average payments per agent (compared to critical values):")
#     # print(df_exact[["critical_v", "avg_payment", "payment_minus_critical", 'var_payment']])


#     # print(f'begin exact simulation with no out of margin points')
#     # df_min_exact = main_exact_in_on_margin(x, y, v, tol = 0.6)
#     # df_min_exact.to_csv(f"exact_min_simulation_n={n}.csv", index=False)
#     # IR(df_min_exact)
#     # exact_min_sum = df_min_exact['critical_v'].sum()
#     # print(f'exact min sum: {exact_min_sum}')

#     # print(f'begin random simulation')
#     # df_random = main_random(n, x, y, v)
#     # # exact_sum = 1.198636
#     # print(f'begin Analysis')
#     # analyze_utilities_and_payments(df_random,df_exact)
#     # analyze_payments_by_mu(df_random,df_exact)
#     # print(f'begin one by one drop')
#     # drop_one_by_one(x, y, v)



# # def train_soft_svm(x, y, v=None, c =1):
# #     model = SVC(kernel='linear', C=c, random_state=0)
# #     if v is None:
# #         model.fit(x, y)
# #     else:
# #         model.fit(x, y, sample_weight=v)
# #     return model



# # def drop_one_by_one(x, y, v):
# #     print('Removing points outside the margin one-by-one (farthest first)')

# #     model = train_soft_svm_hinge(x, y, v)
# #     decision_values = model.decision_function(x)
# #     margin_distances = y * decision_values

# #     # Identify violations: points with margin_dist > 1
# #     outside_indices = np.where(margin_distances > 1 + 1e-2)[0]
# #     violations = margin_distances[outside_indices] - 1

# #     # Sort indices by margin violation (farthest from margin first)
# #     sorted_outside = outside_indices[np.argsort(-violations)]
# #     print("Sorted indices to remove:", sorted_outside)

# #     # Store data as a dict with index keys
# #     data_dict = {i: (x[i], y[i], v[i]) for i in range(len(x))}

# #     # Remove one-by-one based on sorted indices
# #     for idx in sorted_outside:
# #         print(f"Removing index {idx} with margin distance {margin_distances[idx]:.4f}")
# #         if idx in data_dict:
# #             del data_dict[idx]
# #             # Reconstruct filtered arrays
# #             remaining_indices = sorted(data_dict.keys())
# #             x_filtered = np.array([data_dict[i][0] for i in remaining_indices])
# #             y_filtered = np.array([data_dict[i][1] for i in remaining_indices])
# #             v_filtered = np.array([data_dict[i][2] for i in remaining_indices])
# #             n_filtered = len(x_filtered)
# #             print(f'{n_filtered} left out of {n}')
# #             df = main_exact(n_filtered, x_filtered, y_filtered, v_filtered)
# #             IR(df)
# #             exact_min_sum = df['critical_v'].sum()
# #             print(f'exact {n_filtered} sum: {exact_min_sum}')
# #     return True


# # def lvl_1(x,y,v):
# #     T = 5000
# #     allocation_model = train_soft_svm(x, y, v)
# #     pred = np.array(allocation_model.predict(x))
# #     allocations = (y == pred).astype(float)
# #     all_payments = []

# #     for _ in range(T):
# #         payments = mechanism_1(x, y, v, train_soft_svm, allocations)
# #         all_payments.append(payments)

# #     # Stack all payments (shape: [100, n_agents]) and take mean over axis 0
# #     all_payments = np.vstack(all_payments)
# #     avg_payments = np.mean(all_payments, axis=0)
# #     var_payments = np.var(all_payments, axis=0)

# #     return avg_payments, var_payments, np.sum(all_payments)


# # # Sample version of main_random for a fixed mu, and returns DataFrame
# # def run_for_T(n, x, y, v, T, mu=0.7):
# #     records = []
# #     for run in range(T):
# #         new_b, allocation, payments, chi = mechanism_3(x, y, v, mu, train_soft_svm)
# #         for i in range(n):
# #             records.append({
# #                 'run': run,
# #                 'agent': i,
# #                 'payment': payments[i],
# #                 'T': T
# #             })
# #     df = pd.DataFrame(records)
# #     df.to_csv(f"check_sd.csv", index=False)
# #     return df

# # def train_soft_svm_log(x, y, v=None, c=1.0):
# #     """
# #     Train a linear SVM with:
# #       - bias fixed to zero (fit_intercept = False)
# #       - hinge loss averaged over sum(v) instead of N
# #     """
    
# #     # If no sample weights provided, use all weights = 1
# #     if v is None:
# #         v = np.ones(len(y))

# #     model = LogisticRegression(
# #     penalty='l2',
# #     C=c,
# #     solver='liblinear',
# #     max_iter=2000, 
# #     fit_intercept=False,
# #     dual=False,
# #     random_state=0 
# #     )

# #     model.fit(x, y, sample_weight=v)
# #     return model


# # def main_exact_in_on_margin(x, y, v, tol = 1e-2):
# #     print(f'take out points that are out of margin')
# #     model = train_soft_svm(x, y, v)

# #     decision_values = model.decision_function(x)
# #     margin_distances = y * decision_values
# #     inside_or_on_margin_mask = margin_distances <= 1 + tol
# #     # Filter x, y, v to keep only those points
# #     x_filtered = x[inside_or_on_margin_mask]
# #     y_filtered = y[inside_or_on_margin_mask]
# #     v_filtered = v[inside_or_on_margin_mask]
# #     n_filtered = len(x_filtered)

# #     # Get indices of points outside the margin
# #     in_on_margin_indices = np.where(margin_distances <= 1 + tol)[0]
# #     print("Indices inside/on the margin:", in_on_margin_indices)

# #     print(f'{n_filtered} left out of {n}')
# #     df = main_exact(n_filtered, x_filtered, y_filtered, v_filtered)
# #     return df


# # Generate or load data
# def generate_data(n):
#     # Generate synthetic data
#     x, y = make_blobs(n_samples=n, centers=2, random_state=0, cluster_std=1.5)  #1.5
#     y = np.where(y == 0, -1, y)
#     v = np.abs(x[:, 0]) * 10 + np.random.normal(0, 0.1, n)
#     return x, y, v


# for d in [2,4]:
#     T = 100
#     mus = np.linspace(1.0, 0.0, 31)
#     num_payers=[0]*len(mus)
#     for t in range(T):
#         #x,y,v = generate_data(n, mu1 = 1, mu2 = -1 , sigma =1.0)
#         for i, mu in enumerate(mus):
#             mu1 = mu
#             mu2 = -mu
#             sigma = 1.0
#             x,y,v = generate_data_d(n, mu1, mu2, sigma, rng, d=4)
#             #x_updated = update_data(x, y, mu1, mu2)    
#             df_exact, accuracy  = run_exact(x,y,v,c,use_loss, plot = show_plots)
#             #print(df_exact)
#             if df_exact is not None:
#                 num_payers[i] += len(df_exact)
#             # print("_________________") 
#     num_payers = np.array(num_payers) / T  # average over t runs
#     plot_num_payers_vs_dif_mus(mus, num_payers, n)

#     def plot_num_payers_vs_dif_mus(mus, num_payers, n):
#     dif_mus = 2*mus
#     num_payers = np.array(num_payers)
#     plt.plot(dif_mus, num_payers, marker='o')   
#     plt.xlabel("Difference in class means")
#     plt.ylabel("Number of payers")
#     plt.title("Number of payers vs Difference in class means n = {}".format(n))
#     plt.grid()
#     plt.show()


#def main_random(x, y, v, relevant_indcies):
#     T = 50000
#     n = len(y)
#     records = []
#     for mu in np.linspace(0.7, 0.75, 1):
#         for run in range(T):
#             new_b, allocation, payments, chi = mechanism_3(x, y, v, mu, train_soft_svm, relevant_indcies)
#             for i in range(n):
#                 records.append({
#                     'run': run,
#                     'agent': i,
#                     'true_b': v[i], ##assumin g truthfulness
#                     'new_b': new_b[i],
#                     'allocation': allocation[i],
#                     'payment': payments[i],
#                     'chi': chi[i],
#                     'mu': mu,
#                     'welfare': v[i] * allocation[i], #- payments[i],
#                     'utility': v[i] * allocation[i]  - payments[i],
#                 })

#     # === Results DataFrame ===
#     df = pd.DataFrame(records)
#     df.to_csv(f"random_simulation_n={n}.csv", index=False)
#     IR(df)
#     print(f'check IR in expectation')
#     IR(df, exp=True)
#     return df


#------------------------
    # n = 30
    # np.random.seed(42)
    # beta_1 = 1/(4*n)
    # beta_2 = -1/(4*n)
    # x, y, v = generate_data_centered(n) #, beta_1, beta_2)
    # c = 1.0

    # M = np.sum(v)
    # use_loss = 'hinge' #'log' or 'hinge' or 'squared_hinge'
    # # v = np.ones(len(y))

    # print("begin intial training")
    # svm_model = train_soft_svm(x, y, v, c = (c/M), loss = use_loss)

    # plot_svm_decision_boundary(svm_model, x, y, v, 0)

    # print("get relevant indcies")
    # relevant_indcies = get_relevant_indices(svm_model, x, y, v, c/M)
    # print("relevant indcies: ", relevant_indcies)

    # print(f'begin exact simulation')
    # df_exact = main_exact(x, y, v, relevant_indcies, plot = False, c = (c/M), svm = train_soft_svm, loss = use_loss)
    # exact_sum = df_exact['critical_v'].sum()
    # exact_util = df_exact['utility'].sum()
    # exact_welfare = df_exact['welfare'].sum()
    # print(f'exact sum: {exact_sum}')
    # print(f'exact util: {exact_util}')
    # print(f'exact welfare: {exact_welfare}')

    # print("Throw out points outside of 1+beta margin that are correctly classified")
    # throw = get_outside_margins(svm_model, x, y, v, c = c/M)
    # print("throw:", throw)

    # if len(throw) > 0:
    #     mask = np.ones_like(v, dtype=bool)
    #     mask[throw] = False

    #     new_x = x[mask]
    #     new_y = y[mask]
    #     new_v = v[mask]

    #     print('train again')
    #     new_svm_model = train_soft_svm(new_x, new_y, new_v, c = (c/M), loss = use_loss)

    #     plot_svm_decision_boundary(new_svm_model, new_x, new_y, new_v, 0)

    #     print("get relevant indcies")
    #     relevant_indcies = get_relevant_indices(new_svm_model, new_x, new_y, new_v, c/M)
    #     print("relevant indcies: ", relevant_indcies)

    #     is_equ = models_equivalent(svm_model, new_svm_model)
    #     if is_equ:
    #         print("The models are equivelent")
    #     else:
    #         print("The models are NOT equivelent")

    #     print(f'begin exact simulation for minimized model')
    #     df_exact = main_exact(new_x, new_y, new_v, relevant_indcies, plot = False, c = (c/M), svm = train_soft_svm, loss = use_loss)
    #     exact_sum = df_exact['critical_v'].sum()
    #     exact_util = df_exact['utility'].sum()
    #     exact_welfare = df_exact['welfare'].sum()
    #     print(f'exact sum: {exact_sum}')
    #     print(f'exact util: {exact_util}')
    #     print(f'exact welfare: {exact_welfare}')

    # else:
    #     print("no throw")


    ###
    # print("Throw out points outside of 1+beta margin that are correctly classified")
    # new_x, new_y, new_v = get_outside_margins(svm_model, x, y, v, c = c/M)
    # print("get relevant indcies")
    # relevant_indcies = get_relevant_indices(svm_model, new_x, new_y, new_v, c/M)
    # print(f'begin exact simulation')
    # df_exact, model_1 = main_exact(new_x, new_y, new_v, relevant_indcies, plot = False, c = (c/M))
    # exact_sum = df_exact['critical_v'].sum()
    # exact_util = df_exact['utility'].sum()
    # print(f'exact sum: {exact_sum}')
    # print(f'exact util: {exact_util}')


#------------------------
    # non_zero_df = df[df['critical_v'] != 0]
    # non_zero_indices = non_zero_df['agent']
    
    # print(f"Number of non-zero payments: {len(non_zero_indices)}")
    # print(f"Indices with non-zero payments: {list(non_zero_indices)}")

    #non_zero_df["x"] = non_zero_df["agent"].apply(lambda i: x[i])
    # Print critical_v and true_v for those indices
    # print("Critical_v and True_v values:")
    # print(non_zero_df[['agent', 'critical_v', 'true_v']])