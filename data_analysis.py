import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def summarize_metrics(df_random):
    """Group by mu and agent, compute mean and std of utility and payment."""
    return df_random.groupby(['mu', 'agent'])[['utility', 'payment']].agg(['mean', 'std']).reset_index()


def print_and_collect_summaries(grouped, df_exact):
    """Print per-mu summaries and collect data for plotting."""
    mu_values = []
    sum_utilities = []
    mean_std_utilities = []
    sum_payments = []
    mean_std_payments = []

    for mu in sorted(grouped['mu'].unique()):
        mu_df = grouped[grouped['mu'] == mu]
        mu_df.columns = ['mu', 'agent',
                         'utility_mean', 'utility_std',
                         'payment_mean', 'payment_std']

        print(f"\nμ = {mu:.4f}")
        for _, row in mu_df.iterrows():
            agent = int(row['agent'])
            u_avg = row['utility_mean']
            u_std = row['utility_std']
            p_avg = row['payment_mean']
            p_std = row['payment_std']

            print(f"  Agent {agent}: utility avg = {u_avg:.4f}, payment avg = {p_avg:.4f}, "
                  f"util std = {u_std:.4f}, pay std = {p_std:.4f}")

        mu_values.append(mu)
        sum_utilities.append(mu_df['utility_mean'].sum())
        mean_std_utilities.append(mu_df['utility_std'].mean())
        sum_payments.append(mu_df['payment_mean'].sum())
        mean_std_payments.append(mu_df['payment_std'].mean())

    return mu_values, sum_utilities, mean_std_utilities, sum_payments, mean_std_payments

def plot_payment_summaries(mu_values,
                   sum_payments, std_payments, exact_payment_sum):
    """Plot total utility and payment with error bars."""
    plt.figure(figsize=(10, 6))


    # Payment plot
    plt.errorbar(mu_values, sum_payments, yerr=std_payments,
                 fmt='s--', capsize=4, label='Total Payment ± Mean SD')

    if exact_payment_sum is not None:
        plt.axhline(y=exact_payment_sum, color='red', linestyle='--', label=f'Exact Payment Sum = {exact_payment_sum:.2f}')

    plt.xlabel("μ")
    plt.ylabel("Total Value")
    plt.title("Payment vs μ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_util_summaries(mu_values, sum_utilities, std_utilities, exact_utility_sum):
    """Plot total utility and payment with error bars."""
    plt.figure(figsize=(10, 6))

    # Utility plot
    plt.errorbar(mu_values, sum_utilities, yerr=std_utilities,
                 fmt='o-', capsize=4, label='Total Utility ± Mean SD')

    # # Payment plot
    # plt.errorbar(mu_values, sum_payments, yerr=std_payments,
    #              fmt='s--', capsize=4, label='Total Payment ± Mean SD')

    # if exact_payment_sum is not None:
    #     plt.axhline(y=exact_payment_sum, color='red', linestyle='--', label=f'Exact Payment Sum = {exact_payment_sum:.2f}')

    if exact_utility_sum is not None:
        plt.axhline(y=exact_utility_sum, color='black', linestyle='--', label=f'Exact Utility Sum = {exact_utility_sum:.2f}')

    plt.xlabel("μ")
    plt.ylabel("Total Value")
    plt.title("Utility vs μ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_utilities_and_payments(df_random, df_exact):
    grouped = summarize_metrics(df_random)
    exact_payment_sum = df_exact['critical_v'].sum()
    exact_utility_sum = df_exact['utility'].sum()
    mu_values, sum_utils, std_utils, sum_pays, std_pays = print_and_collect_summaries(grouped, df_exact)
    plot_util_summaries(mu_values, sum_utils, std_utils, exact_utility_sum)
    plot_payment_summaries(mu_values, sum_pays, std_pays, exact_payment_sum)
    # (mu_values, sum_utils, std_utils, sum_pays, std_pays, exact_payment_sum, exact_utility_sum)
    #


def analyze_payments_by_mu(df_random, df_exact):
    # Group by mu and agent: compute mean and standard deviation of payment per agent
    grouped = df_random.groupby(['mu', 'agent'])['payment'].agg(['mean', 'std']).reset_index()

    exact_sum = df_exact['critical_v'].sum()
    mu_values = []
    sum_payments = []
    mean_std_devs = []

    for mu in sorted(df_random['mu'].unique()):
        print(f"\nμ = {mu:.4f}")
        mu_df = grouped[grouped['mu'] == mu]

        sum_mu = mu_df['mean'].sum()  # total expected payment at this μ
        mean_std = mu_df['std'].mean()  # mean SD across agents at this μ

        for _, row in mu_df.iterrows():
            agent = int(row['agent'])
            avg_payment = row['mean']
            agent_std = row['std']
            exact_payment = df_exact[df_exact['agent'] == agent]['critical_v'].values[0]
            print(f"  Agent {agent}: empirical avg = {avg_payment:.4f}, "
                  f"exact_payment = {exact_payment}, payment_dif = {abs(exact_payment - avg_payment)}, std = {agent_std:.4f}")

        print(f"  Sum of avg payments for μ = {mu:.4f} → {sum_mu:.4f}, Mean agent SD = {mean_std:.4f}")
        mu_values.append(mu)
        sum_payments.append(sum_mu)
        mean_std_devs.append(mean_std)

    # First plot: Unbounded y-axis
    plt.figure(figsize=(8, 5))
    plt.errorbar(mu_values, sum_payments, yerr=mean_std_devs, fmt='o-', capsize=5, label='Empirical Sum ± Mean Agent SD')
    if exact_sum is not None:
        plt.axhline(y=exact_sum, color='red', linestyle='--', label=f'Exact Sum = {exact_sum}')
    plt.xlabel("μ")
    plt.ylabel("Sum of Average Payments")
    plt.title("Total Payment vs μ (Unbounded Y)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Second plot: Bounded y-axis
    if exact_sum is not None:
        plt.figure(figsize=(8, 5))
        plt.errorbar(mu_values, sum_payments, yerr=mean_std_devs, fmt='o-', capsize=5,
                     label='Empirical Sum ± Mean Agent SD')
        plt.axhline(y=exact_sum, color='red', linestyle='--', label=f'Exact Sum = {exact_sum}')
        plt.ylim(exact_sum - 5, exact_sum + 5)
        plt.xlabel("μ")
        plt.ylabel("Sum of Average Payments")
        plt.title("Total Payment vs μ (Y limited to exact_sum ± 5)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def IR(df, exp=False):
    col = 'utility'
    if exp:
        col = 'utility_mean'
        df = df.groupby(['mu', 'agent'])['utility'].mean().reset_index(name=col)

    result = 'IR holds' if (df[col] >= 0).all() else 'IR does not hold'
    print(result)


    #Our generic transformation exhibits high variability in payments, and includes an
# explicit tradeoff between the variability in payments and the loss in performance.
# Formally, variability can be expressed as variance, maximal absolute value, or (for
# positive types) maximal rebate.