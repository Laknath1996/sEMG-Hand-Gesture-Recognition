"""
objective : Train the classification model
author(s) : Ashwin de Silva, Malsha Perera
date      : 14 Aug 2019

The code is based on the following paper :
[1] A. D. Silva, M. V. Perera, K. Wickramasinghe, A. M. Naim,
    T. Dulantha Lalitharatne and S. L. Kappel, "Real-Time Hand Gesture
    Recognition Using Temporal Muscle Activation Maps of Multi-Channel
    Semg Signals," ICASSP 2020 - 2020 IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP), Barcelona,
    Spain, 2020, pp. 1299-1303.
"""

from tma.models.classifiers import fit_NN
from tma.functions import *
from tma.utils import combine_data
from tma.visualization.functions import reduce_dims

experiment = EmgLearn(fs=200,
                      no_channels=8,
                      obs_dur=0.4)

# data path(s)
data_path_1 = '/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/data/subject_1001/trans_map_dataset_1.h5'
data_path_2 = '/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/data/subject_1001/trans_map_dataset_2.h5'

# model path (to save the trained model)
model_path = '/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/models/subject_1001/model/cnn_model.h5'

# combine the data from the data path(s)
X, y = combine_data([data_path_1, data_path_2])

# visulize the TMA maps in latent space with reduced dimensions
reduce_dims(X=X,
            y=y,
            data_path=None,
            algorithm='tsne',
            perplexity=50,
            labels=['M', 'R', 'HC', 'V', 'PO'])

# fit a Support Vector Machine (SVM_) model
# fit_SVM(data_path, model_path)

# fit a Deep Learning model
fit_NN(X=X,
       y=y,
       data_path=None,
       model_path=model_path,
       model='cnn',
       experiment=experiment,
       epochs=20)
