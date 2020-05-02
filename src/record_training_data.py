"""
objective : Create the dataset of Temporal Muscle Activation maps for the
selected set of hand gestures.
author(s) : Ashwin de Silva, Malsha Perera
date      : 2019 Aug 14

The code is based on the following paper :
[1] A. D. Silva, M. V. Perera, K. Wickramasinghe, A. M. Naim,
    T. Dulantha Lalitharatne and S. L. Kappel, "Real-Time Hand Gesture
    Recognition Using Temporal Muscle Activation Maps of Multi-Channel
    Semg Signals," ICASSP 2020 - 2020 IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP), Barcelona,
    Spain, 2020, pp. 1299-1303.

"""

from tma.tma_emg_learn import *
import h5py

# define the TemporalMuscleActivationMap
experiment = TemporalMuscleActivationMapsEmgLearn(fs=200,
                                                  no_channels=8,
                                                  obs_dur=0.400)

####################################################
# Define useful parameters here.
####################################################

data_save_path = 'data/subject_1001_Ashwin'
flexion_gestures = ['M_1', 'R_1', 'HC_1', 'V_1', 'PO_1']  # gesture types
thresh = [3, 2, 2, 2, 2]  # onset detection threholds for each gesture type. This can be set manually or by using methods explained in [1].
delta = 0.3  # the half width of the neighborhood selected around each gesture onset points

# Note : If you are using the method in [1] to set the onset
# detection thresholds, please plot the onset points and verify their positions.
# If positions are incorrect, you can plot the difference signal and set
# an appropriate threshold manually.

####################################################
# Record Data
####################################################

experiment.record_gestures(gestures=flexion_gestures,
                           data_path=data_save_path,
                           recording_time=100,
                           plot=True)

# ####################################################
# Create the dataset
# ####################################################

# load the saved recordings
gesture_database = experiment.load_emg_data(data_path=data_save_path,
                                            gestures=flexion_gestures)

# extract the signal envelopes
filtered_gesture_database = experiment.filter_signal_database(gesture_database)

X = np.zeros((1, experiment.H, experiment.T))  # stores TMA maps
y = np.zeros((1,))  # stores gesture labels of the corresponding TMA maps

for i, gesture in enumerate(list(filtered_gesture_database.keys()), start=0):
    # detects the gesture onset points
    print("Sampling from the gesture : %s" % gesture)
    signal = filtered_gesture_database[gesture]
    trans = experiment.detect_onsets(signal=signal,
                                     obs_inc=0.100,
                                     threshold=thresh[i],
                                     refractory_period=8,
                                     max_dur=97,
                                     plot=True,
                                     plot_diffs=True)
    for t in trans:
        # stores the TMA maps from a neighborhood defined around each gesture onset points. These maps
        # together with their labels are used as the training data for the classification algorithm
        neighborhood = signal[:, (t - int(delta * experiment.fs)):(t + int(delta * experiment.fs))]
        trans_maps = experiment.get_tma_maps(neighborhood, obs_inc=0.010)
        X = np.concatenate((X, trans_maps), axis=0)
        y = np.concatenate((y, i * np.ones((trans_maps.shape[0],))))

X = X[1:, ...]  # TMA maps
y = y[1:]  # gesture labels assigned for the TMA maps

print("Data Shape : ", X.shape)
print("Label Shape : ", y.shape)

print("Labels : ", np.unique(y))

# ####################################################
# # Save the dataset
# ####################################################

dataset_file = h5py.File(data_save_path + '/trans_map_dataset_2.h5')
dataset_file.create_dataset('data', data=X)
dataset_file.create_dataset('label', data=y)
dataset_file.close()
