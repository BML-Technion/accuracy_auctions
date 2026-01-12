import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#kappa
def plot_mean_kappa(mega_df, labels = [1,-1]):
    mean_accuracy_validation = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['d'])['k'].mean()
    )
    plt.plot(mean_accuracy_validation, 'o-')
    plt.ylabel("k")
    plt.xlabel(f"d")
    plt.title(f"Mean kappa as function of dimensions d")
    plt.show()

# number of payers metrics plots
def plot_mean_num_payers(mega_df, var, labels = [1, -1]):
    max_t = (max(mega_df['t'].unique())+1) 
    mean_num_payers = (
    (mega_df[(mega_df['critical_v'] > 0) & (mega_df['label'].isin(labels))])
    .groupby(['d', var])['critical_v'].count()
    .unstack(level=0).fillna(0)) / max_t 

    plt.plot(mean_num_payers)
    plt.ylabel("average number of payers per agent")
    plt.xlabel(f'{var}')
    plt.title(f"Average Number of Payers vs {var}s for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_num_payers.columns)
    plt.show()

def plot_mean_payment(mega_df, var, labels = [1, -1]):
    mean_payments = (
    (mega_df[mega_df['label'].isin(labels)])
    #mega_df
    .groupby(['t', 'd', var])['critical_v'].mean()
    .groupby(['d', var]).mean()
    .unstack(level=0)
)
    plt.plot(mean_payments)
    plt.ylabel("average payment per agent")
    plt.xlabel(f'{var}')
    plt.title(f"Average Payment vs {var}s for different dimensions d and label = {labels}")
    plt.legend(title='Dimension d', labels=mean_payments.columns)
    plt.show()

def plot_mean_payment_sd(mega_df, var, labels = [1, -1]):
    mean_payments_sd = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', var])['critical_v'].mean()
    .groupby(['d', var]).std()
    .unstack(level=0)
)
    plt.plot(mean_payments_sd)
    plt.ylabel("SD of average payment per agent")
    plt.xlabel(f"{var}")
    plt.title(f"SD of Average Payment vs {var}s for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_payments_sd.columns)
    plt.show()

#welfare metrics plots
def plot_mean_welfare(mega_df, var, labels = [1, -1]):
    mean_welfare = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', var])['welfare'].mean()
    .groupby(['d', var]).mean()
    .unstack(level=0)
)
    plt.plot(mean_welfare)
    plt.ylabel("average welfare per agent")
    plt.xlabel(f"{var}")
    plt.title(f"Average Welfare vs {var}s for different dimensions d and label = {labels}")
    plt.legend(title='Dimension d', labels=mean_welfare.columns)
    plt.show()

def plot_mean_welfare_sd(mega_df, var, labels = [1, -1]):
    mean_welfare_sd = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', var])['welfare'].mean()
    .groupby(['d', var]).std()
    .unstack(level=0)
)
    plt.plot(mean_welfare_sd)
    plt.ylabel("Mean Welfare sd")
    plt.xlabel(f"{var}")
    plt.title(f"Mean Welfare Standard Deviation vs {var}s for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_welfare_sd.columns)
    plt.show()

# utility metrics plots
def plot_mean_utility(mega_df, var, labels = [1, -1]):
    mean_sum_welfare = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', var])['utility'].mean()
    .groupby(['d', var]).mean()
    .unstack(level=0)
)
    plt.plot(mean_sum_welfare)
    plt.ylabel("Mean utility per agent")
    plt.xlabel(f"{var}")
    plt.title(f"Average utility vs {var}s for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_sum_welfare.columns)
    plt.show()

def plot_mean_utility_sd(mega_df, var, labels = [1, -1]):
    mean_utility_sd = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', var])['utility'].mean()
    .groupby(['d', var]).std()
    .unstack(level=0)
)
    plt.plot(mean_utility_sd)
    plt.ylabel("Mean Utility sd")
    plt.xlabel(f"{var}")
    plt.title(f"Mean Utility Standard Deviation vs {var}s for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_utility_sd.columns)
    plt.show()

# accuracy metrics plots
def plot_mean_accuracy_train(mega_df, var, labels = [1, -1]):
    mean_accuracy_train = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', var])['allocation'].mean()
    .groupby(['d', var]).mean()
    .unstack(level=0)
)
    plt.plot(mean_accuracy_train)
    plt.ylabel("mean accuracy train (out of 1)")
    plt.xlabel(f"{var}")
    plt.title(f"Mean Accuracy on Training Set vs {var}s for different dimensions d and label = {labels}")
    plt.legend(title='Dimension d', labels=mean_accuracy_train.columns)
    plt.show()

def plot_mean_accuracy_validation(mega_df, var, labels = [1, -1]):
    mean_accuracy_validation = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', var])['valid_acc'].mean()
    .groupby(['d', var]).mean()
    .unstack(level=0)
)
    plt.plot(mean_accuracy_validation)
    plt.ylabel("mean accuracy validation")
    plt.xlabel(f"{var}")
    plt.title(f"Mean Accuracy on Validation Set vs {var}s for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_accuracy_validation.columns)
    plt.show()

# percent of points in [0, beta] interval 
def plot_percent_relevant(mega_df, var, labels = [1, -1]):
    percent_relevant = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['d', var])['is_relevant'].mean()
    .unstack(level=0)
)
    plt.plot(percent_relevant)
    plt.ylabel(f"percent of points in [0, beta] interval")
    plt.xlabel(f"{var}")
    plt.title(f"Percent of Points in [0, beta] vs {var}s for different dimensions d")
    plt.legend(title='Dimension d', labels=percent_relevant.columns)
    plt.show()

# percent of points in [0, beta] interval 
def plot_percent_relevant_from_payers(mega_df, var, labels = [1, -1]):
    mask = (
        (mega_df["is_relevant"] > 0)
        & (mega_df["label"].isin(labels))
    )

    df_f = mega_df.loc[mask]

    agg = (
        df_f
        .groupby(["d", var])
        .agg(
            num_relevant=("is_relevant", "size"),
            num_relevant_and_critical=("critical_v", lambda x: (x > 0).sum())
        )
        .unstack("d", fill_value=0)
    )

    ratio = (
        agg["num_relevant_and_critical"]
        .div(agg["num_relevant"])
        .fillna(0)
    )



    plt.plot(ratio)
    plt.ylabel(f"percent of points in [0, beta] interval")
    plt.xlabel(f"{var}")
    plt.title(f"Percent of Points in [0, beta] vs {var}s for different dimensions d")
    plt.legend(title='Dimension d', labels=ratio.columns)
    plt.show()


# mixes
def acc_num_payers(mega_df, var, T, labels = [1, -1]):
    mean_accuracy_train = mega_df.groupby(['d', var])['allocation'].mean().unstack(level=0)
    mean_num_payers = (
                        mega_df[mega_df['label'].isin(labels)]
                        .assign(pos=lambda df: df['critical_v'] > 0)
                        .groupby([var, 'd'])['pos']
                        .sum()
                        .unstack('d', fill_value=0)
                        / T
                    )
    plt.plot(mean_accuracy_train, mean_num_payers, 'o-')
    plt.ylabel("Average number of payers (out of 100)")
    plt.xlabel(f'Training Accuracy')
    plt.title(f"Average Number of Payers and Training Accuracy for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_num_payers.columns)
    plt.show()

def acc_2(mega_df, var, T, ds, labels = [1, -1]):
    mean_accuracy_train = mega_df[(mega_df['label'].isin(labels)) 
                                  & (mega_df['d'].isin(ds))].groupby(['d', var])['allocation'].mean().unstack(level=0)
    mean_num_payers = (
                        mega_df[(mega_df['label'].isin(labels)) & (mega_df['d'].isin(ds))]
                        .assign(pos=lambda df: df['critical_v'] > 0)
                        .groupby([var, 'd'])['pos']
                        .sum()
                        .unstack('d', fill_value=0)
                        / T
                    )
    mean_num_relevants = (
                        mega_df[(mega_df['label'].isin(labels)) & (mega_df['d'].isin(ds))]
                        .groupby([var, 'd'])['is_relevant'].sum()
                        .unstack('d', fill_value=0) / T
                    )

    ratio = mean_num_payers / mean_num_relevants
    
    fig, ax1 = plt.subplots(figsize=(8,5))

    # --- First y-axis ---
    ax1.plot(mean_accuracy_train, mean_num_payers, 'o-', label='number of payers', color='tab:blue')
    ax1.plot(mean_accuracy_train, mean_num_relevants, 'o-', label='number of relevants', color='tab:orange')
    ax1.set_xlabel('Training Accuracy')
    ax1.set_ylabel('Average number of agents (out of 100)')
    ax1.set_title(f"Payers, Relevant Agents, and Their Ratio vs Training Accuracy for d = {ds[0]}")
    ax1.set_ylim(0, 100)  # force y-axis from 0 to n
    # Combine legends for first axis
    lines1, labels1 = ax1.get_legend_handles_labels()

    # --- Second y-axis ---
    ax2 = ax1.twinx()  # share x-axis
    ax2.plot(mean_accuracy_train, ratio, 'o-', label='payers/relevants', color='tab:green')
    ax2.set_ylabel('Payers / Relevants Ratio')
    ax2.set_ylim(0, 1)  # force y-axis from 0 to 1


    # Combine legends from both axes
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.show()

def sigma_ratio(mega_df, T, var = 'sigma_pos', ds = [1]):
    mean_num_payers_1 = (
                        mega_df[(mega_df['label'].isin([1])) & (mega_df['d'].isin(ds))]
                        .assign(pos=lambda df: df['critical_v'] > 0)
                        .groupby([var, 'd'])['pos']
                        .sum()
                        .unstack('d', fill_value=0)
                        / T
                    )
    mean_num_payers_0 = (
                        mega_df[(mega_df['label'].isin([-1])) & (mega_df['d'].isin(ds))]
                        .assign(pos=lambda df: df['critical_v'] > 0)
                        .groupby([var, 'd'])['pos']
                        .sum()
                        .unstack('d', fill_value=0)
                        / T
                    )
    
    mean_payments_1 = ((mega_df[mega_df['label'].isin([1]) & (mega_df['d'].isin(ds))])
                    .groupby(['t', 'd', var])['critical_v'].mean()
                    .groupby(['d', var]).mean()
                    .unstack(level=0))

    mean_payments_0 = ((mega_df[mega_df['label'].isin([-1]) & (mega_df['d'].isin(ds))])
                    .groupby(['t', 'd', var])['critical_v'].mean()
                    .groupby(['d', var]).mean()
                    .unstack(level=0))


    fig, ax1 = plt.subplots(figsize=(8,5))

    # --- Plot ratio of payers (positive / negative) ---
    ax1.plot(mean_num_payers_1 / mean_num_payers_0, 'o-', color='tab:blue', label='Payer Ratio (Positive / Negative)')
    ax1.plot(mean_payments_1 / mean_payments_0, 'o-', color='tab:green', label='Payment Ratio (Positive / Negative)')

    # --- Axis labels ---
    ax1.set_xlabel('Relative Variance (σ⁺ / σ⁻)', fontsize=12)
    ax1.set_ylabel('Ratio (Positive / Negative)', fontsize=12)

    # --- Title ---
    ax1.set_title(f"Ratio of Positive to Negative Payers vs Relative Variance for d = {ds[0]}", fontsize=14)

    # --- Legend ---
    ax1.legend(loc='best', fontsize=11)

    # Optional: improve grid for readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def sigma_ratio_relevants(mega_df, T, var = 'sigma_pos', ds = [1]):
    mean_num_relevants_1 = (
                        mega_df[(mega_df['label'].isin([1])) & (mega_df['d'].isin(ds))]
                        .groupby([var, 'd'])['is_relevant'].sum()
                        .unstack('d', fill_value=0) / T
                    )
    mean_num_relevants_0 = (
                        mega_df[(mega_df['label'].isin([-1])) & (mega_df['d'].isin(ds))]
                        .groupby([var, 'd'])['is_relevant'].sum()
                        .unstack('d', fill_value=0) / T
                    )

    mean_payments_1 = ((mega_df[mega_df['label'].isin([1]) & (mega_df['d'].isin(ds))])
                    .groupby(['t', 'd', var])['critical_v'].mean()
                    .groupby(['d', var]).mean()
                    .unstack(level=0))

    mean_payments_0 = ((mega_df[mega_df['label'].isin([-1]) & (mega_df['d'].isin(ds))])
                    .groupby(['t', 'd', var])['critical_v'].mean()
                    .groupby(['d', var]).mean()
                    .unstack(level=0))

    
    fig, ax1 = plt.subplots(figsize=(8,5))

    # --- Plot ratio of payers (positive / negative) ---
    ax1.plot(mean_num_relevants_1 / mean_num_relevants_0, 'o-', color='tab:blue', label='Relevants Ratio (Positive / Negative)')
    ax1.plot(mean_payments_1 / mean_payments_0, 'o-', color='tab:green', label='Payments Ratio (Positive / Negative)')

    # --- Axis labels ---
    ax1.set_xlabel('Relative Variance (σ⁺ / σ⁻)', fontsize=12)
    ax1.set_ylabel(' Ratio (Positive / Negative)', fontsize=12)

    # --- Title ---
    ax1.set_title(f"Ratio of Positive to Negative Payers vs Relative Variance for d = {ds[0]}", fontsize=14)

    # --- Legend ---
    ax1.legend(loc='best', fontsize=11)

    # Optional: improve grid for readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def relevant_ratio(mega_df, T, var = 'sigma_pos', ds = [1]):
    mean_num_payers_1 = (
                        mega_df[(mega_df['label'].isin([1])) & (mega_df['d'].isin(ds))]
                        .assign(pos=lambda df: df['critical_v'] > 0)
                        .groupby([var, 'd'])['pos']
                        .sum()
                        .unstack('d', fill_value=0)
                        / T
                    )
    mean_num_payers_0 = (
                        mega_df[(mega_df['label'].isin([-1])) & (mega_df['d'].isin(ds))]
                        .assign(pos=lambda df: df['critical_v'] > 0)
                        .groupby([var, 'd'])['pos']
                        .sum()
                        .unstack('d', fill_value=0)
                        / T
                    )
    mean_num_relevants_1 = (
                        mega_df[(mega_df['label'].isin([1])) & (mega_df['d'].isin(ds))]
                        .groupby([var, 'd'])['is_relevant'].sum()
                        .unstack('d', fill_value=0) / T
                    )
    mean_num_relevants_0 = (
                        mega_df[(mega_df['label'].isin([-1])) & (mega_df['d'].isin(ds))]
                        .groupby([var, 'd'])['is_relevant'].sum()
                        .unstack('d', fill_value=0) / T
                    )

    mean_payments_1 = ((mega_df[mega_df['label'].isin([1]) & (mega_df['d'].isin(ds))])
                    .groupby(['t', 'd', var])['critical_v'].mean()
                    .groupby(['d', var]).mean()
                    .unstack(level=0))

    mean_payments_0 = ((mega_df[mega_df['label'].isin([-1]) & (mega_df['d'].isin(ds))])
                    .groupby(['t', 'd', var])['critical_v'].mean()
                    .groupby(['d', var]).mean()
                    .unstack(level=0))

    

    
    fig, ax1 = plt.subplots(figsize=(8,5))

    # --- Plot ratio of payers (positive / negative) ---
    ax1.plot(mean_num_relevants_1/mean_num_relevants_0, mean_num_payers_1 / mean_num_payers_0, 'o-', color='tab:blue', label='Payer Ratio (Positive / Negative)')
    ax1.plot(mean_num_relevants_1/mean_num_relevants_0, mean_payments_1 / mean_payments_0, 'o-', color='tab:green', label='Payments Ratio (Positive / Negative)')

    # --- Axis labels ---
    ax1.set_xlabel('Average relevants Ratio (Positive / Negative)', fontsize=12)
    ax1.set_ylabel('Ratio (Positive / Negative)', fontsize=12)

    # --- Title ---
    ax1.set_title(f"Ratio of Positive to Negative Payers vs Relevants for d = {ds[0]}", fontsize=14)

    # --- Legend ---
    ax1.legend(loc='best', fontsize=11)

    # Optional: improve grid for readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_mean_num_payers_m(mega_df, d, labels = [1, -1]):
    max_t = (max(mega_df['t'].unique())+1) 
    mean_num_payers = (
    (mega_df[(mega_df['critical_v'] > 0) & (mega_df['label'].isin(labels))
             & (mega_df['d'] == d)])
    .groupby([ 'sigma', 'm_total'])['critical_v'].count()
    .unstack(level=0).fillna(0)) / max_t 

    plt.plot(mean_num_payers, 'o-')
    plt.ylabel("average number of payers (out of m)")
    plt.xlabel(f'm_total')
    plt.title(f"Average Number of Payers vs m_total for d = {d}")
    plt.legend(title='sigma', labels=mean_num_payers.columns)
    plt.show()

def plot_mean_accuracy_train_v(mega_df, var, d):
    mean_accuracy_train = (
    (mega_df[(mega_df['label'].isin([1,-1]))
             & (mega_df['d'] == d)])
    .groupby(['t',  var])['allocation'].mean()
    .groupby([var]).mean()
    #.unstack(level=0)
)
    mean_accuracy_train1 = (
    (mega_df[(mega_df['label'].isin([1]))
             & (mega_df['d'] == d)])
    .groupby(['t',  var])['allocation'].mean()
    .groupby([var]).mean()
    #.unstack(level=0)
)
    mean_accuracy_train0 = (
    (mega_df[(mega_df['label'].isin([-1]))
             & (mega_df['d'] == d)])
    .groupby(['t',  var])['allocation'].mean()
    .groupby([var]).mean()
    #.unstack(level=0)
)
  
    plt.plot(mean_accuracy_train, label = 'y = {1,-1}')
    plt.plot(mean_accuracy_train0, label = 'y=-1')
    plt.plot(mean_accuracy_train1, label = 'y=1')
    plt.ylabel("mean accuracy train (out of 1)")
    plt.xlabel(f"Ration v_neg / v_pos")
    plt.title(f"Mean Accuracy on Training Set vs v ratio for d = {d}")
    plt.legend()
    plt.show()


def stack_acc_v(mega_df, var, d):
    mean_accuracy_train = (
    (mega_df[(mega_df['label'].isin([1,-1]))
             & (mega_df['d'] == d)])
    .groupby(['t',  var])['allocation'].mean()
    .groupby([var]).mean()
    #.unstack(level=0)
)
    mean_accuracy_train1 = (
    (mega_df[(mega_df['label'].isin([1]))
             & (mega_df['d'] == d)])
    .groupby(['t',  var])['allocation'].mean()
    .groupby([var]).mean() /2
    #.unstack(level=0)
)
    mean_accuracy_train0 = (
    (mega_df[(mega_df['label'].isin([-1]))
             & (mega_df['d'] == d)])
    .groupby(['t',  var])['allocation'].mean()
    .groupby([var]).mean() /2
    #.unstack(level=0)
)
    fig, ax = plt.subplots(figsize=(8,5))

    ax.stackplot(
        mean_accuracy_train.index.to_numpy(),
        mean_accuracy_train0.values, mean_accuracy_train1.values,
        labels=['y = -1', 'y = 1'],
        #alpha=0.7
    )

    # Optional: overlay the average/total as a line
    #ax.plot(mean_accuracy_train, color='black', linewidth=2, label='Average')
    # ax.plot(mean_accuracy_train, label = 'y = {1,-1}')
    # ax.plot(mean_accuracy_train0, label = 'y=-1')
    # ax.plot(mean_accuracy_train1 + mean_accuracy_train0/2, label = 'y=1')
    ax.set_xlabel(f"Ration v_neg / v_pos")
    ax.set_ylabel("mean accuracy train (out of 1)")
    ax.set_title(f"Mean Accuracy on Training Set vs v ratio for d = {d}")

    ax.legend(loc='upper left')
    plt.show()

def plot_mean_welfare_v(mega_df, var, d):
    mean_welfare = (
    (mega_df[mega_df['label'].isin([1,-1])
             & (mega_df['d'] == d)])
    .groupby(['t',  var])['welfare'].mean()
    .groupby([ var]).mean()
    #.unstack(level=0)
)
    mean_welfare_1 = (
    (mega_df[mega_df['label'].isin([1])
             & (mega_df['d'] == d)])
    .groupby(['t',  var])['welfare'].mean()
    .groupby([ var]).mean()
    #.unstack(level=0)
)    
    mean_welfare_0 = (
    (mega_df[mega_df['label'].isin([-1])
             & (mega_df['d'] == d)])
    .groupby(['t',  var])['welfare'].mean()
    .groupby([ var]).mean()
    #.unstack(level=0)
)
    plt.plot(mean_welfare, label = 'y = {1,-1}')
    plt.plot(mean_welfare_0, label = 'y=-1')
    plt.plot(mean_welfare_1, label = 'y=1')
    plt.ylabel(f"mean welfare")
    plt.xlabel(f"Ration v_neg / v_pos")
    plt.title(f"Mean Welfare  vs v ratio for d = {d}")
    plt.legend()
    plt.show()


def plot_mean_payment_v(mega_df, var, d):
    mean_payments = (
    (mega_df[mega_df['label'].isin([1,-1])
             & (mega_df['d'] == d)])
    .groupby(['t', var])['critical_v'].mean()
    .groupby([var]).mean()
    #.unstack(level=0)
)
    mean_payments0 = (
    (mega_df[mega_df['label'].isin([-1])
             & (mega_df['d'] == d)])
    .groupby(['t', var])['critical_v'].mean()
    .groupby([var]).mean()
    #.unstack(level=0)
)
    mean_payments1 = (
    (mega_df[mega_df['label'].isin([1])
             & (mega_df['d'] == d)])
    .groupby(['t', var])['critical_v'].mean()
    .groupby([var]).mean()
    #.unstack(level=0)
)
    # mask = (mean_payments.index) <= 1
    # mean_payments[mask]
    plt.plot(mean_payments, label = 'y = {1,-1}')
    plt.plot(mean_payments0, label = 'y=-1')
    plt.plot(mean_payments1, label = 'y=1')
    plt.ylabel("mean payment ")
    plt.xlabel(f"Ration v_neg / v_pos")
    plt.title(f"Mean Payment  vs v ratio for d = {d}")
    plt.legend()
    plt.show()
