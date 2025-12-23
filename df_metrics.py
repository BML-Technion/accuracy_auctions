import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# number of payers metrics plots
def plot_mean_num_payers(mega_df, labels = [1, -1]):
    max_t = (max(mega_df['t'].unique())+1) 
    mean_num_payers = (
    (mega_df[(mega_df['critical_v'] > 0) & (mega_df['label'].isin(labels))])
    .groupby(['d', 'sigma'])['critical_v'].count()
    .unstack(level=0).fillna(0)) / max_t 

    plt.plot(mean_num_payers)
    plt.ylabel("average number of payers per agent")
    plt.xlabel("sigma")
    plt.title("Average Number of Payers vs Sigma for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_num_payers.columns)
    plt.show()

def plot_mean_payment(mega_df, labels = [1, -1]):
    mean_payments = (
    (mega_df[mega_df['label'].isin(labels)])
    #mega_df
    .groupby(['t', 'd', 'sigma'])['critical_v'].mean()
    .groupby(['d', 'sigma']).mean()
    .unstack(level=0)
)
    plt.plot(mean_payments)
    plt.ylabel("average payment per agent")
    plt.xlabel("sigma")
    plt.title("Average Payment vs Sigma for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_payments.columns)
    plt.show()

def plot_mean_payment_sd(mega_df, labels = [1, -1]):
    mean_payments_sd = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', 'sigma'])['critical_v'].mean()
    .groupby(['d', 'sigma']).std()
    .unstack(level=0)
)
    plt.plot(mean_payments_sd)
    plt.ylabel("SD of average payment per agent")
    plt.xlabel("sigma")
    plt.title("SD of Average Payment vs Sigma for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_payments_sd.columns)
    plt.show()

#welfare metrics plots
def plot_mean_welfare(mega_df, labels = [1, -1]):
    mean_welfare = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', 'sigma'])['welfare'].mean()
    .groupby(['d', 'sigma']).mean()
    .unstack(level=0)
)
    plt.plot(mean_welfare)
    plt.ylabel("average welfare per agent")
    plt.xlabel("sigma")
    plt.title("Average Welfare vs Sigma for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_welfare.columns)
    plt.show()

def plot_mean_welfare_sd(mega_df, labels = [1, -1]):
    mean_welfare_sd = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', 'sigma'])['welfare'].mean()
    .groupby(['d', 'sigma']).std()
    .unstack(level=0)
)
    plt.plot(mean_welfare_sd)
    plt.ylabel("Mean Welfare sd")
    plt.xlabel("sigma")
    plt.title(f"Mean Welfare Standard Deviation vs Sigma for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_welfare_sd.columns)
    plt.show()

# utility metrics plots
def plot_mean_utility(mega_df, labels = [1, -1]):
    mean_sum_welfare = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', 'sigma'])['utility'].mean()
    .groupby(['d', 'sigma']).mean()
    .unstack(level=0)
)
    plt.plot(mean_sum_welfare)
    plt.ylabel("Mean utility per agent")
    plt.xlabel("sigma")
    plt.title("Average utility vs Sigma for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_sum_welfare.columns)
    plt.show()

def plot_mean_utility_sd(mega_df, labels = [1, -1]):
    mean_utility_sd = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', 'sigma'])['utility'].mean()
    .groupby(['d', 'sigma']).std()
    .unstack(level=0)
)
    plt.plot(mean_utility_sd)
    plt.ylabel("Mean Utility sd")
    plt.xlabel("sigma")
    plt.title(f"Mean Utility Standard Deviation vs Sigma for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_utility_sd.columns)
    plt.show()

# accuracy metrics plots
def plot_mean_accuracy_train(mega_df, labels = [1, -1]):
    mean_accuracy_train = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', 'sigma'])['allocation'].mean()
    .groupby(['d', 'sigma']).mean()
    .unstack(level=0)
)
    plt.plot(mean_accuracy_train)
    plt.ylabel("mean accuracy train")
    plt.xlabel("sigma")
    plt.title("Mean Accuracy on Training Set vs Sigma for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_accuracy_train.columns)
    plt.show()

def plot_mean_accuracy_validation(mega_df, labels = [1, -1]):
    mean_accuracy_validation = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', 'sigma'])['valid_acc'].mean()
    .groupby(['d', 'sigma']).mean()
    .unstack(level=0)
)
    plt.plot(mean_accuracy_validation)
    plt.ylabel("mean accuracy validation")
    plt.xlabel("sigma")
    plt.title("Mean Accuracy on Validation Set vs Sigma for different dimensions d")
    plt.legend(title='Dimension d', labels=mean_accuracy_validation.columns)
    plt.show()

# percent of points in [0, beta] interval 
def plot_percent_relevant(mega_df, labels = [1, -1]):
    percent_relevant = (
    (mega_df[mega_df['label'].isin(labels)])
    .groupby(['t', 'd', 'sigma'])['is_relevant'].mean()
    .groupby(['d', 'sigma']).mean()
    .unstack(level=0)
)
    plt.plot(percent_relevant)
    plt.ylabel(f"percent of points in [0, beta] interval")
    plt.xlabel("sigma")
    plt.title(f"Percent of Points in [0, beta] vs Sigma for different dimensions d")
    plt.legend(title='Dimension d', labels=percent_relevant.columns)
    plt.show()