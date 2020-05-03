"""
objective : Contains the functions for tma map visualization
author(s) : Ashwin de Silva
date      : 
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tma.utils import plot_latent_space, load_tma_data
import matplotlib.pyplot as plt


def visualize_tma_time_series(data_path):
    """visualize the time seires of TMA maps"""

    X, y = load_tma_data(data_path)

    fig = plt.figure()
    ax = fig.add_subplot('111')

    for i in range(X.shape[0]):
        C = X[i, ...].reshape(X.shape[1], X.shape[2])
        l = y[i]
        ax.imshow(C, vmin=0, vmax=1)
        ax.set_title('Label : %i' % l)
        plt.pause(0.1)

    # labels = np.unique(y)
    # fig, axes = plt.subplots(figsize=(13, 4), ncols=4)
    # for i, l in enumerate(labels, start=0):
    #     idx = np.where(y == l)[0]
    #     temp = np.mean(X[idx, ...], axis=0)
    #     temp[:8, :] = temp[:8, :]*6
    #     pos = axes[i].imshow(temp, vmin=0, vmax=1)
    #     axes[i].set_title("Label : %i" % l)
    #     fig.colorbar(pos, ax=axes[i])
    # plt.show()


def reduce_dims(X=None, y=None, data_path=None, algorithm='pca', perplexity=50, labels=['M', 'R', 'HC', 'V', 'PO']):
    """visualize the data in a latent space with reduced dimensions"""

    if data_path is not None:
        X, y = load_tma_data(data_path)

    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    print("Number of Training Samples : ", X.shape[0])

    # standardize the data
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # reduce the dimensionality of the data
    if algorithm == 'pca':
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

    if algorithm == 'tsne':
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, verbose=True)
        X_reduced = tsne.fit_transform(X)

    # plot the latent space
    plot_latent_space(X_reduced, y, labels)
