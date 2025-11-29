import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, to_rgba


def plot_svm_decision_boundary_2d(clf, X, y, v=None, target_idx=None, title="SVC Decision Boundary"):
    """
    Plots the decision boundary, margins, and support vectors for a 2D SVC.

    Parameters:
    - clf: a trained sklearn.svm.SVC model
    - X: feature matrix (n_samples, 2)
    - y: labels (n_samples,)
    - v: optional weights for data points, used to scale marker size
    - target_idx: index of the data point to highlight
    - title: plot title
    """
    # Define two distinct colors for binary classification
    colors = ['tab:blue', 'tab:orange']
    cmap = ListedColormap(colors)
    light_colors = [to_rgba(col, alpha=0.15) for col in colors]
    light_cmap = ListedColormap(light_colors)

    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError("This function only supports binary classification.")

    plt.figure(figsize=(8, 6))

    # Plot all data points
    if v is not None:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=v * 10, cmap=cmap, edgecolors='k')
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')

    # Annotate each point with its index
    for i, (x0, x1) in enumerate(X):
        plt.text(x0 + 0.02, x1 + 0.02, str(i), fontsize=9, color='black')

    # Highlight the target point
    if target_idx is not None:
        plt.scatter(X[target_idx, 0], X[target_idx, 1], s=100, facecolors='none', edgecolors='red', linewidths=2)

    # Background classification regions
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500),
                         np.linspace(ylim[0], ylim[1], 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z_labels = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z_labels, alpha=0.15, cmap=light_cmap)

    # Decision boundary and margins
    Z = clf.decision_function(grid).reshape(xx.shape)
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1],
               linestyles=['--', '-', '--'], linewidths=1.5)

    # # Plot support vectors
    # plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
    #             facecolors='none', edgecolors='pink', linewidths=1.5, label='Support Vectors')

    # Custom legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Label {int(lbl)}',
                   markerfacecolor=col, markersize=10, markeredgecolor='k')
        for lbl, col in zip(unique_labels, colors)
    ]
    handles.append(
        plt.Line2D([0], [0], marker='o', color='w', label='Target',
                   markerfacecolor='none', markeredgecolor='r', markersize=10, linewidth=2)
    )
    # handles.append(
    #     plt.Line2D([0], [0], marker='o', color='w', label='Support Vectors',
    #                markerfacecolor='none', markeredgecolor='pink', markersize=10, linewidth=1.5)
    # )

    plt.legend(handles=handles)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True)
    plt.savefig('2d_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    # plt.close()


def plot_allocation_curve(v_vals, a_vals, target_idx, real_v):
    plt.figure(figsize=(8, 4))
    plt.plot(v_vals, a_vals, marker='o')
    plt.axvline(x=real_v, color='red', linestyle='--', linewidth=2, label=f'v[{target_idx}]')
    plt.xlabel(f'v (value weight) of agent {target_idx} ')
    plt.ylabel('a(v) - Allocation')
    plt.title(f'Allocation function a(v) for agent {target_idx}')
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    # plt.show()
    plt.close()

def plot_avg_welfare_curve(v_vals, welfare_vals, target_idx, real_v):
    plt.figure(figsize=(8, 4))
    plt.plot(v_vals, welfare_vals, marker='o', color='green')
    plt.axvline(x=real_v, color='red', linestyle='--', linewidth=2, label=f'v[{target_idx}]')
    plt.xlabel(f'v (value weight) of agent {target_idx}')
    plt.ylabel('Total Welfare (sum of v_i for correct predictions)')
    plt.title(f'Total Welfare vs. v for agent {target_idx}')
    plt.grid(True)
    # plt.show()
    plt.close()


def plot_accuracy_curve(v_vals, accuracy_vals, target_idx, real_v):
    plt.figure(figsize=(8, 4))
    plt.plot(v_vals, accuracy_vals, marker='o', color='green')
    plt.axvline(x=real_v, color='red', linestyle='--', linewidth=2, label=f'v[{target_idx}]')

    plt.xlabel(f'v (value weight) of agent {target_idx}')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. v for agent {target_idx}')
    plt.grid(True)
    # plt.show()
    plt.close()

def plot_losses_curve(v_vals, target_idx, real_v, zero_one_loss_vals = None, hinge_loss_vals = None):
    plt.figure(figsize=(8, 4))
    plt.axvline(x=real_v, color='red', linestyle='--', linewidth=2, label=f'v[{target_idx}]')
    if zero_one_loss_vals is not None:
        plt.plot(v_vals, zero_one_loss_vals, marker='o', color='green', label='0-1 Loss')
    if hinge_loss_vals is not None:
        plt.plot(v_vals, hinge_loss_vals, marker='o', color='blue', label='Hinge Loss')
    plt.xlabel(f'v (value weight) of agent {target_idx}')
    plt.ylabel('Average Loss')
    plt.title(f'Average Loss vs. v for agent {target_idx}')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.close()


def calculate_losses(y, y_pred, decision_values=None, v=None):
    """
    Compute 0-1 loss and hinge loss, with optional sample weights.

    Parameters:
    - y: true labels (0 or 1)
    - y_pred: predicted labels (0 or 1)
    - decision_values: raw decision function values (needed for hinge loss)
    - v: sample weights (default: None)

    Returns:
    - dict with 0-1 loss and hinge loss (weighted if v is given)
    """
    n = len(y)
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    if v is None:
        v = np.ones_like(y, dtype=float)
    else:
        v = np.asarray(v, dtype=float)
        assert v.shape == y.shape, "Weights v must match shape of y"

    # 1. 0-1 loss (weighted)
    incorrect = (y != y_pred).astype(float)
    zero_one_loss = np.sum(incorrect * v) / n

    # 2. Hinge loss (weighted)
    if decision_values is None:
        raise ValueError("decision_values must be provided to compute hinge loss.")

    # Convert y from {0,1} to {-1,1}
    y_signed = y
    unique_vals = np.unique(y_signed)
    if 0 in unique_vals:
        y_signed = 2 * y - 1

    margins = y_signed * decision_values
    hinge_losses = np.maximum(0, 1 - margins)
    hinge = np.sum(hinge_losses * v) / n
    return zero_one_loss, hinge

def plot_svm_decision_boundary_1d(clf, X, y, v=None, target_idx=None, title="SVC Decision Boundary"):
    """
    Plot decision boundary for an SVM trained on 1D data.

    Parameters:
    - clf: trained sklearn.svm.SVC model
    - X: 1D features (n_samples,) or (n_samples, 1)
    - y: labels (+1, -1), shape (n_samples,)
    - v: optional weights for marker sizes
    - target_idx: optional index of a point to highlight
    - title: plot title
    """
    # Flatten X to 1D if needed
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y)

    colors = ['tab:blue', 'tab:orange']
    cmap = ListedColormap(colors)
    light_colors = [to_rgba(c, alpha=0.15) for c in colors]

    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError("This function supports only binary classification.")

    plt.figure(figsize=(8, 2.5))

    # Plot points on x-axis (y=0)
    if v is not None:
        plt.scatter(X, np.zeros_like(X), c=y, s=v * 20, cmap=cmap, edgecolors='k', zorder=3)
    else:
        plt.scatter(X, np.zeros_like(X), c=y, cmap=cmap, edgecolors='k', zorder=3)

    # Annotate points by index
    for i, x_val in enumerate(X):
        plt.text(x_val, 0.05, str(i), ha='center', fontsize=9, color='black', zorder=4)

    # Highlight target point if specified
    if target_idx is not None:
        plt.scatter(X[target_idx], 0, s=150, facecolors='none', edgecolors='red', linewidths=2, zorder=5)

    ax = plt.gca()
    xlim = ax.get_xlim()

    # Create dense grid over x-axis
    xx = np.linspace(xlim[0], xlim[1], 500).reshape(-1, 1)
    Z_decision = clf.decision_function(xx)
    decision_boundary_x = xx[np.argmin(np.abs(Z_decision))][0]

    # Draw decision boundary
    ax.axvline(decision_boundary_x, color='k', linestyle='-')

    # Shade left/right classification regions
    ylim = ax.get_ylim()
    ax.fill_betweenx(ylim, xlim[0], decision_boundary_x, color=colors[0], alpha=0.15)
    ax.fill_betweenx(ylim, decision_boundary_x, xlim[1], color=colors[1], alpha=0.15)

    # Format plot
    plt.yticks([])  # Hide y-axis ticks
    plt.xlabel('Feature 1')
    plt.title(title)
    plt.grid(axis='x')

    # Legend
    legend_handles = [mpatches.Patch(color=colors[i], label=f'Label {int(lbl)}') for i, lbl in
                      enumerate(unique_labels)]
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Target',
                                     markerfacecolor='none', markeredgecolor='r', markersize=10, linewidth=2))
    plt.legend(handles=legend_handles, loc='upper right')

    plt.tight_layout()
    plt.savefig('1d_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_svm_decision_boundary(clf, X, y, v=None, target_idx=None, title="SVC Decision Boundary"):
    # if X.ndim == 1:
    #     plot_svm_decision_boundary_1d(clf, X, y, v, target_idx, title)
    # elif X.ndim == 2:
    #     plot_svm_decision_boundary_2d(clf, X, y, v, target_idx, title)

    if X.shape[1] == 1:
        # pass
        # shape (n,1) treat as 1D data
        plot_svm_decision_boundary_1d(clf, X[:, 0], y, v=v, target_idx=target_idx, title=title)
    elif X.shape[1] == 2:
        # shape (n,2)
        plot_svm_decision_boundary_2d(clf, X, y, v=v, target_idx=target_idx, title=title)
    else:
        raise ValueError(f"Only 1D or 2D features supported. Got shape {X.shape}")


def plot_svm_decision_boundary_2_models(clf_1, clf_2, X, y, v=None, target_idx=None, title="SVC Decision Boundary"):
    # Flatten X to 1D if needed
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y)

    colors = ['tab:blue', 'tab:orange']
    cmap = ListedColormap(colors)
    light_colors = [to_rgba(c, alpha=0.15) for c in colors]

    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError("This function supports only binary classification.")

    plt.figure(figsize=(8, 2.5))

    # Plot points on x-axis (y=0)
    if v is not None:
        plt.scatter(X, np.zeros_like(X), c=y, s=v * 20, cmap=cmap, edgecolors='k', zorder=3)
    else:
        plt.scatter(X, np.zeros_like(X), c=y, cmap=cmap, edgecolors='k', zorder=3)

    # Annotate points by index
    for i, x_val in enumerate(X):
        plt.text(x_val, 0.05, str(i), ha='center', fontsize=9, color='black', zorder=4)

    # Highlight target point if specified
    if target_idx is not None:
        plt.scatter(X[target_idx], 0, s=150, facecolors='none', edgecolors='red', linewidths=2, zorder=5)

    ax = plt.gca()
    xlim = ax.get_xlim()

    # Create dense grid over x-axis
    xx = np.linspace(xlim[0], xlim[1], 500).reshape(-1, 1)
    Z_decision = clf_1.decision_function(xx)
    decision_boundary_x = xx[np.argmin(np.abs(Z_decision))][0]

    Z_decision_2 = clf_2.decision_function(xx)
    decision_boundary_x2 = xx[np.argmin(np.abs(Z_decision_2))][0]

    # Draw decision boundary
    ax.axvline(decision_boundary_x, color='k', linestyle='-')
    ax.axvline(decision_boundary_x2, color='k', linestyle=':')

    # Shade left/right classification regions
    ylim = ax.get_ylim()
    ax.fill_betweenx(ylim, xlim[0], decision_boundary_x, color=colors[0], alpha=0.15)
    ax.fill_betweenx(ylim, decision_boundary_x, xlim[1], color=colors[1], alpha=0.15)

    # Format plot
    plt.yticks([])  # Hide y-axis ticks
    plt.xlabel('Feature 1')
    plt.title(title)
    plt.grid(axis='x')

    # Legend
    legend_handles = [mpatches.Patch(color=colors[i], label=f'Label {int(lbl)}') for i, lbl in
                      enumerate(unique_labels)]
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Target',
                                     markerfacecolor='none', markeredgecolor='r', markersize=10, linewidth=2))
    plt.legend(handles=legend_handles, loc='upper right')

    plt.tight_layout()
    plt.show()