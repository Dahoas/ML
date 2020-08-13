import matplotlib.pyplot as plt
import numpy as np

import logistic_regression

def get_point_marker_size():
    return 10

def get_marker_edge_width():
    return 2

def get_line_width():
    return 2

def get_font_size():
    return 16

def plot_objective_contours(X, y, lamb, w_min=-8, w_max=8, title=None, colors=None,
        show_labels=True, new_figure=True, show_figure=True, save_filename=None):
    """
    Plots the logistic_regression.objective function with parameters
    X, y, lamb (lambda).

    X: Nx2 numpy ndarray, training input
    y: Nx1 numpy ndarray, training output
    lamb: Scalar lambda hyperparameter
    w_min (default=-8): Minimum of axes range
    w_max (default=8): Maximum of axes range
    title (default=None): Title of plot if not None
    colors (default=None): Color of contour lines. None will use default cmap.
    show_labels (default=True): Show numerical labels on contour lines
    new_figure (default=True): If true, calls plt.figure(), which create a 
        figure. If false, it will modify an existing figure (if one exists).
    show_figure (default=True): If true, calls plt.show(), which will open
        a new window and block program execution until that window is closed
    save_filename (defalut=None): If not None, save figure to save_filename 
    """
    N = 101
    
    w1 = np.linspace(w_min, w_max, N)
    w2 = np.linspace(w_min, w_max, N)
    W1, W2 = np.meshgrid(w1,w2)
    
    obj = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            w = np.array([[W1[i,j]], [W2[i,j]]])
            obj[i, j] = logistic_regression.objective(w, X, y, lamb)
    
    # Ploting contour 

    if new_figure:
        plt.figure(figsize=(8,8))

    ax = plt.gca()
    contour_plot = ax.contour(W1, W2, obj, levels=20, colors=colors)
    if show_labels:
        ax.clabel(contour_plot, inline=1, fontsize=get_font_size())
    plt.tick_params(labelsize=get_font_size())
    #ax.set_xlabel('w1', fontsize = get_font_size())
    #ax.set_ylabel('w2', fontsize = get_font_size())

    ax.axhline(0, color='lightgray')
    plt.axvline(0, color='lightgray')
    ax.set_axisbelow(True)

    if title is not None:
        plt.title(title)

    if save_filename is not None:
        plt.savefig(save_filename)

    if show_figure:
        plt.show()

def plot_optimization_path(point_list, color, linestyle='-', label=None):
    """
    Plot arrows stepping between points in the point list.

    point_list: List of 2D points, each of which is a 2x1 numpy ndarray
    color: matplotlib color
    linestyle: matplotlib linestyle
    label: Label to put in the plt.legend (plt.legend is not called in here)

    Does not call plt.figure() or plt.show()
    """
    X = []
    Y = []
    U = []
    V = []

    start = point_list[0]
    for point in point_list[1:]:
        X.append(start[0,0])
        Y.append(start[1,0])

        U.append(point[0,0]-start[0,0])
        V.append(point[1,0]-start[1,0])

        start = point

    plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1,
        color=color, linestyle=linestyle, linewidth=get_line_width(), label=label)


