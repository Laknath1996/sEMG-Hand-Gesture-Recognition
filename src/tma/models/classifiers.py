"""
objective : define the classifiers for TMA map classification
author(s) : Ashwin de Silva, Malsha Perera
date      : 14 Aug 2019
"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import h5py
import numpy as np
import joblib
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tma.tma_emg_utils import plot_latent_space
import matplotlib.pyplot as plt
from keras import backend as K
from models.nn_models import nn, cnn


def load_tma_data(data_path):
    """load transition evolution map data"""
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


def visualize_data(data_path):
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


def reduce_dims(X=None, y=None, data_path=None, algorithm='pca', perplexity=50):
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
    plot_latent_space(X_reduced, y, ['M', 'R', 'HC', 'V', 'PO'])


def fit_SVM_classifier(data_path, model_path):
    """fit an SVM classifier to the data"""
    X, y = load_tma_data(data_path)
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    # standardize data
    sc = StandardScaler()
    X = sc.fit_transform(X=X)

    # train svm model
    clf = SVC(C=1, kernel='rbf', probability=False)
    clf.fit(X, y)

    print("Training Accuracy : ", clf.score(X, y))

    # save scaler and classfier
    joblib.dump(clf, os.path.join(model_path, 'model.joblib'))
    joblib.dump(sc, os.path.join(model_path, 'scaler.joblib'))


def fit_DL_classifier(experiment, epochs, X=None, y=None, data_path=None, model_path=None, model='cnn'):
    """fit an CNN classifier to the data"""

    if data_path is not None:
        X, y = load_tma_data(data_path)

    num_classes = len(np.unique(y))

    if model == 'nn':
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        print(X.shape)
    elif model == 'cnn':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    if num_classes > 2:
        y_cat = to_categorical(y, num_classes=num_classes)
    else:
        y_cat = y

    # standardize data
    if model == 'nn':
        sc = StandardScaler()
        X = sc.fit_transform(X=X)
        joblib.dump(sc, os.path.join(model_path, 'scaler.joblib'))

    # train the model
    if model == 'nn':
        model = nn((experiment.H * experiment.T,), num_classes)
        model_checkpoint = ModelCheckpoint(model_path + '/nn_model.h5', monitor='loss', verbose=1, save_best_only=True)
    elif model == 'cnn':
        model = cnn((X.shape[1], X.shape[2], 1), num_classes)
        model_checkpoint = ModelCheckpoint(model_path + '/cnn_model.h5', monitor='loss', verbose=1, save_best_only=True)

    model.fit(X, y_cat,
              epochs=epochs,
              callbacks=[model_checkpoint],
              batch_size=32)

# def save_feature_vector(data_path, model_path):
#     X, y = load_tma_data(data_path)
#     num_classes = len(np.unique(y))
#
#     X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
#
#     if num_classes > 2:
#         y_cat = to_categorical(y, num_classes=num_classes)
#     else:
#         y_cat = y
#
#     model = cnn((X.shape[1], X.shape[2], 1), num_classes)
#     model.load_weights(model_path)
#
#     get_feature_vector = K.function([model.layers[0].input],
#                                     [model.layers[6].output])
#
#     X_f = get_feature_vector([X])[0]
#
#     print(X_f.shape)
#
#     reduce_dims(X=X_f, y=y, data_path=None, algorithm='tsne', perplexity=50)
