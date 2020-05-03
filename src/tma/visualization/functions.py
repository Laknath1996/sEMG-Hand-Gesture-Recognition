"""
The MIT License (MIT)
Copyright (c) 2020, Ashwin De Silva and Malsha Perera

Other Contributors : Asma Naim, Kithmin Wickramasinghe, Thilina
Lalitharatne, Simon Kappel

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

objective : Containes functions to visualize TMA maps and visualize the
TMA maps in 2-D latent space

The code is based on the following paper :
[1] A. D. Silva, M. V. Perera, K. Wickramasinghe, A. M. Naim,
    T. Dulantha Lalitharatne and S. L. Kappel, "Real-Time Hand Gesture
    Recognition Using Temporal Muscle Activation Maps of Multi-Channel
    Semg Signals," ICASSP 2020 - 2020 IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP), Barcelona,
    Spain, 2020, pp. 1299-1303.
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
