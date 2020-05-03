"""
objective : define the classifiers for TMA map classification
author(s) : Ashwin de Silva, Malsha Perera
date      : 14 Aug 2019
"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from tma.utils import load_tma_data
from tma.models.nn_models import nn, cnn


def fit_SVM(data_path, model_path):
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


def fit_NN(experiment, epochs, X=None, y=None, data_path=None, model_path=None, model='cnn'):
    """fit an CNN classifier to the data"""

    if data_path is not None:
        X, y = load_tma_data(data_path)

    num_classes = len(np.unique(y))

    if model == 'fc':
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        print(X.shape)
    elif model == 'cnn':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    if num_classes > 2:
        y_cat = to_categorical(y, num_classes=num_classes)
    else:
        y_cat = y

    # standardize data
    if model == 'fc':
        sc = StandardScaler()
        X = sc.fit_transform(X=X)
        joblib.dump(sc, os.path.join(model_path, 'scaler.joblib'))

    # train the model
    if model == 'fc':
        model = nn((experiment.H * experiment.T,), num_classes)
        model_checkpoint = ModelCheckpoint(model_path + '/fc_model.h5', monitor='loss', verbose=1, save_best_only=True)
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
