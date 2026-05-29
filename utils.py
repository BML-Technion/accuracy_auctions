import numpy as np
from config import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, to_rgba
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def plot_knn_2d(knn_model, X, y, v, X_test=None, k=None):
    X = np.array(X)
    y = np.array(y)
    v = np.array(v)
    v_scaled = 200 * (v / v.max())  # scale for plotting

    # --- Grid for decision boundary ---
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # --- Plot decision boundary ---
    plt.figure(figsize=(7,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    # --- Plot training points per class ---
    classes = np.unique(y)
    for cls in classes:
        mask = y == cls
        plt.scatter(X[mask,0], X[mask,1],
                    s=v_scaled[mask],
                    label=f'Class {cls}',
                    edgecolors='k',
                    cmap='coolwarm')
        
    # --- Highlight test points and their neighbors ---
    if X_test is not None:
        X_test = np.array(X_test)
        plt.scatter(X_test[:,0], X_test[:,1], color='green', s=120, marker='X', label='Test point')
        if k is None:
            k = knn_model.n_neighbors
        distances, indices = knn_model.kneighbors(X_test, n_neighbors=k)
        for neigh_idx in indices:
            plt.scatter(X[neigh_idx,0], X[neigh_idx,1],
                        s=v_scaled[neigh_idx], facecolors='none', edgecolors='orange', linewidths=2,
                        label='Neighbors')

    # --- Labels & legend ---
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('KNN Decision Boundary with Point Sizes Correlated to v')
    plt.legend()
    plt.show()

def update_indices(relevant_indices, removed_indices):
    if len(removed_indices) == 0:
        return dict(zip(relevant_indices, relevant_indices))
    
    removed = sorted(removed_indices)
    updated = []

    for idx in relevant_indices:
        shift = sum(r < idx for r in removed)
        updated.append(idx - shift)

    return dict(zip(relevant_indices, updated))

def get_relevant_throw_idx(svm_model, x, y, v, c, loss, k=None, tol = 0.05):
    decision_values = svm_model.decision_function(x)
    margin = y * decision_values

    k = k if k is not None else np.linalg.norm(x, axis=1).max()
    lam = 1 / c #TRAIN_SVM_PARAMS[loss]['C']
    sigma_loss = TRAIN_SVM_PARAMS[loss]['admissibility']
    beta_values = ((k*K_COEF)**2 * v * sigma_loss**2) / (2 * lam)

    #get_relevant
    relevant_mask =  (margin > 0) if loss=="squared_hinge" else (margin > 0) & (margin <= beta_values + tol) 
    relevant_indices = np.where(relevant_mask)[0]

    #get_throw
    throw_idx = []

    if loss == 'hinge' and len(relevant_indices) > 0:
        max_v_index = np.argmax(v[relevant_indices])
        max_relevant_beta = beta_values[max_v_index]

        # final outside selection: margin > 1 + beta max
        throw_mask = margin > (1 + 2 * max_relevant_beta) + tol
        throw_mask[relevant_indices] = False  # ensure relevant points are not thrown
        throw_idx = np.where(throw_mask)[0]

    return set(relevant_indices), throw_idx

# plots 
def plot_svm_decision_boundary(clf, X, y, v=None, target_idx=None, title="SVC Decision Boundary", labels = None):
    if X.shape[1] == 1:
        # shape (n,1) treat as 1D data
        plot_svm_decision_boundary_1d(clf, X[:, 0], y, v=v, target_idx=target_idx, title=title) #, labels= labels)
    elif X.shape[1] == 2:
        # shape (n,2)
        plot_svm_decision_boundary_2d(clf, X, y, v=v, target_idx=target_idx, title=title, labels= labels)
    else:
        raise ValueError(f"Only 1D or 2D features supported. Got shape {X.shape}")

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
        # print("v is not none")
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
    # plt.savefig('1d_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_svm_decision_boundary_2d(clf, X, y, v=None, target_idx=None, title="SVC Decision Boundary", labels = None):
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
    if labels is not None and len(labels) > 0:
        for i, (x0, x1) in enumerate(X):
            plt.text(x0 + 0.02, x1 + 0.02, str(labels[i]), fontsize=9, color='black')
    else:
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