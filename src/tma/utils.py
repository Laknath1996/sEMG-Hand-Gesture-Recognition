"""
The MIT License (MIT)
Copyright (c) 2020, Ashwin De Silva and Malsha Perera

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

objective : Contains the utility functions for EmgLEARN class

The code is based on the following paper :
[1] A. D. Silva, M. V. Perera, K. Wickramasinghe, A. M. Naim,
    T. Dulantha Lalitharatne and S. L. Kappel, "Real-Time Hand Gesture
    Recognition Using Temporal Muscle Activation Maps of Multi-Channel
    Semg Signals," ICASSP 2020 - 2020 IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP), Barcelona,
    Spain, 2020, pp. 1299-1303.
"""


# import the libraries
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

plt.style.use(
    '/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/src/tma/other/PaperDoubleFig.mplstyle')

from matplotlib import cm
import numpy as np
import pandas as pd
import h5py
import joblib
import os


def plot_latent_space(X_reduced, y_train, labels):
    """
    plot the reduced dim space
    """
    num_classes = len(np.unique(y_train))
    cmap = cm.get_cmap('jet')
    idx = np.linspace(0, 1, num_classes)
    colors = cmap(idx)

    for l, c in zip(np.unique(y_train), colors):
        plt.scatter(X_reduced[y_train == l, 0], X_reduced[y_train == l, 1], c=c, label=l)

    plt.xlabel('Dim 1')
    plt.ylabel('DIm 2')
    plt.legend(tuple(labels), loc='lower right')
    plt.show()


def plot_recording(data, fs):
    """
    plot the raw recordings from the acquisition system
    """
    fig = plt.figure()
    num_channels = data.shape[1]
    num_samples = data.shape[0]
    print(num_channels)
    axes = [fig.add_subplot('%i1' % num_channels + str(i)) for i in range(0, num_channels)]
    [(ax.set_ylim([-100, 100])) for ax in axes]
    t = np.array(list(range(num_samples))) / fs
    i = 0
    for ax in axes:
        ax.plot(t, data[:, i])
        i += 1
    plt.show()


def read_from_csv(file):
    """
    read the recordings from the .csv files
    """
    df = pd.read_csv(file, delimiter='\t')
    data = df.values
    data = data[:, 1:data.shape[1]]
    return data.T


def write_to_csv(file, data):
    """
    write data to .csv files
    """
    df = pd.DataFrame(data)
    df.to_csv(file, sep='\t')


def load_tma_data(data_path):
    """load tma map data"""
    dataset_file = h5py.File(data_path)
    X = np.array(dataset_file['data'])
    y = np.array(dataset_file['label'])
    return X, y


def combine_data(data_paths):
    """combine datasets from more than two data paths"""
    X, y = load_tma_data(data_paths[1])
    for data_path in data_paths[1:]:
        X1, y1 = load_tma_data(data_path)
        X = np.concatenate((X, X1), axis=0)
        y = np.concatenate((y, y1))
    return X, y


def load_SVM_model(model_path):
    sc = joblib.load(os.path.join(model_path, 'scaler.joblib'))
    clf = joblib.load(os.path.join(model_path, 'model.joblib'))
    return sc, clf


# def get_corr_plot(X, title):
#     """
#     visualize the correlation matrix
#     """
#     df = pd.DataFrame(X.T)
#     corr = df.corr()
#     cm = plt.imshow(corr, cmap='Blues', vmin=0, vmax=1)
#     cm.set_cmap('Blues')
#     plt.colorbar()
#     plt.xticks(range(len(corr.columns)), corr.columns)
#     plt.yticks(range(len(corr.columns)), corr.columns)
#     plt.title(title)
#     plt.show()


# def plot_ethogram(t, pred, gestures):
#     """
#     plot the action ethogram
#     """
#     num_classes = len(gestures)
#     classes = list(range(0, num_classes))
#     cmap = cm.get_cmap('jet')
#     idx = np.linspace(0, 1, num_classes)
#     colors = cmap(idx)
#     for l, c in zip(classes, colors):
#         idx = np.where(pred == l)[0]
#         t_i = t[idx]
#         p_i = (l + 1) * np.ones(len(t_i))
#         plt.scatter(t_i, p_i, c=c, label=l)
#     plt.ylim((0, num_classes + 2))
#     plt.legend(gestures)
#     plt.xticks(np.arange(0, 130, step=5))
#     plt.grid()
#     plt.show()
