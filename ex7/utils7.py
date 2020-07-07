import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

def displayData(X, example_width=None, figsize=(10, 10)):
    """Plots 2D data

    Args:
        X (array-like): mxn input data of m examples and n features
        example_width ([int], optional): Width of 2D image in pixels. Defaults to None.
        figsize (tuple, optional): Height and width of figure in inches. Defaults to (10, 10).
    """
    # compute rows, columns
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None] # promote to 2D array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = int(n / example_width)

    # compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_height, example_width, order='F'), cmap='gray')
        ax.axis('off')


def featureNormalize(X):
    """Normalizes features in X to have mean 0 and std dev 1.

    Args:
        X (array-like): A dataset which is an mxn matrix.
    """
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X, axis=0, ddof=1)
    X_norm /= sigma

    return X_norm, mu, sigma


def plotProgresskMeans(i, X, centroid_history, idx_history):
    """Displays progress of k-means as it is running.

    Args:
        i (int): Current iteration of k-means
        X (array-like): Dataset which is a mxn matrix
        centroid_history (list): List of computed centroids for all iterations
        idx_history (list): List of computed assigned indices for all iterations
    """
    K = centroid_history[0].shape[0]
    plt.gcf().clf()
    cmap = plt.cm.rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=2)

    for k in range(K):
        current = np.stack([c[k, :] for c in centroid_history[:i+1]], axis=0)
        plt.plot(current[:, 0], current[:, 1],
                '-Xk',
                mec='k',
                lw=2,
                ms=10,
                mfc=cmap(norm(k)),
                mew=2)

        plt.scatter(X[:, 0], X[:, 1],   
                    c=idx_history[i],
                    cmap=cmap,
                    marker='o',
                    s=8**2,
                    linewidths=1,)
    plt.grid(False)
    plt.title('Iteration number %d' % (i+1))


def runKmeans(X, centroids, findClosestCentroids, computeCentroids, max_iters=10, plot_progress=False):
    """Runs the K-means algorithm

    Args:
        X (array_like): A dataset of dimensions (m, n)
        centroids (array_like): Initial centroid location of each cluster
        findClosestCentroids (func): Function for computing cluster assignment
        computeCentroids (func): Function for calculating the centroid of each cluster
        max_iters (int, optional): Total iterations used to optimize K-means. Defaults to 10.
        plot_progress (bool, optional): Option to plot progress. Defaults to False.

    Returns:
        centroids: array_like
    """
    K = centroids.shape[0]
    idx = None
    idx_history = []
    centroid_history = []

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)

        if plot_progress:
            idx_history.append(idx)
            centroid_history.append(centroids)

        centroids = computeCentroids(X, idx, K)

    if plot_progress:
        fig = plt.figure()
        anim = FuncAnimation(fig, plotProgresskMeans,
                            frames=max_iters,
                            interval=500,
                            repeat_delay=2,
                            fargs=(X, centroid_history, idx_history))
        return centroids, idx, anim
    return centroids, idx