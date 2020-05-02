"""
objective : Train the classification model
author(s) : Ashwin de Silva, Malsha Perera
date      : 14 Aug 2019
"""

from models.classifiers import *
from tma.tma_emg_learn import *

experiment = TemporalMuscleActivationMapsEmgLearn(fs=200,
                                                  no_channels=8,
                                                  obs_dur=0.4)

# data path(s)
data_path_1 = 'data/subject_1001_Ashwin/trans_map_dataset_1.h5'
data_path_2 = 'data/subject_1001_Ashwin/trans_map_dataset_2.h5'

# model path (to save the trained model)
model_path = 'models/subject_1001_Ashwin/model/cnn_model.h5'

# combine the data from the data path(s)
X, y = combine_data([data_path_1, data_path_2])

# visualize the time seires of TMA maps
# visualize_data(data_path_1)

# visulize the TMA maps in latent space with reduced dimensions
# reduce_dims(X=X, y=y, data_path=None, algorithm='tsne', perplexity=50)

# fit a Support Vector Machine (SVM_) model
# fit_SVM_classifier(data_path, model_path)

# fit a Deep Learning model
fit_DL_classifier(X=X,
                  y=y,
                  data_path=None,
                  model_path=model_path,
                  model='cnn',
                  experiment=experiment,
                  epochs=20)
