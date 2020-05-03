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

objective : Contains the primary functions that supports the computation
of TMA maps

The code is based on the following paper :
[1] A. D. Silva, M. V. Perera, K. Wickramasinghe, A. M. Naim,
    T. Dulantha Lalitharatne and S. L. Kappel, "Real-Time Hand Gesture
    Recognition Using Temporal Muscle Activation Maps of Multi-Channel
    Semg Signals," ICASSP 2020 - 2020 IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP), Barcelona,
    Spain, 2020, pp. 1299-1303.
"""


from tma.utils import *
import matplotlib.pyplot as plt

plt.style.use('/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/src/tma/other/PaperDoubleFig.mplstyle')
import numpy as np
from scipy.signal import butter, sosfilt
import os
from collections import deque
import myo
import time
from threading import Lock
import joblib

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# np.seterr(divide='ignore', invalid='ignore') # ignore the true division error


class EmgCollector(myo.DeviceListener):
    def __init__(self, start):
        self.emg_data = np.zeros((1, 8))
        self.lock = Lock()
        self.start = start

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        self.emg_data = np.concatenate((self.emg_data, np.reshape(np.array(event.emg), (1, 8))), axis=0)
        print(time.time() - self.start)


class EmgLearn(object):
    """
    contains the functions required to build the TMA based real-time hand gesture
    recognition algorithm
    """

    def __init__(self, fs=4000, no_replicates=10, no_channels=7,
                 hold_time=5, rest_time=5, obs_dur=0.200, obs_inc=0.100,
                 allowed_time=5):
        """

        :param fs: sampling frequency of the signals (for MyoArm band : 200Hz)
        :param no_replicates: number of replicates obtained per each gesture
        when collecting training data
        :param no_channels: number of sEMG channle (electrodes) present in
        the sEMG device (for MyoArm band : 8)
        :param hold_time: the time duration a gesture replication is held
        :param rest_time: the time duration the hand kept in neural position
        until the next gesture replicate
        :param obs_dur: the time width of the window considere for each
        TMA maps (M value in the paper)
        :param obs_inc: the time difference between two adjacent TMA maps
        (k value in the paper)
        :param allowed_time: The time until the first gesture replicate
        """
        self.fs = fs
        self.no_replicates = no_replicates
        self.no_channels = no_channels
        self.hold_time = hold_time
        self.rest_time = rest_time
        self.obs_dur = obs_dur
        self.obs_inc = obs_inc
        self.allowed_time = allowed_time
        self.no_bins = int((hold_time - obs_dur) / obs_inc + 1)

        L = self.no_channels
        self.T = int(self.obs_dur * self.fs)
        self.H = int(L * (L + 3) / 2)

    def record(self, save_path, recording_time=110, sdk_path='/Users/ashwin/FYP/sdk/myo.framework/myo', plot=False):
        """
        records the sEMG signals for the specified gesture type and saves
        the recording as a .csv file
        :param save_path: path where the dataset is saved
        :param recording_time: the duration of the recording
        :param sdk_path: the MyoSDK path
        :param plot: plot the recorded signals or not (boolean)
        :return: None
        """
        myo.init(sdk_path)
        hub = myo.Hub()

        input("PRESS ENTER TO RECORD")
        start = time.time()
        listener = EmgCollector(start)

        hub.run(listener, recording_time * 1000)
        data = listener.emg_data
        data = data[1:]
        if plot:
            plot_recording(data, self.fs)
        write_to_csv(save_path, data=listener.emg_data)

    def record_gestures(self, gestures, data_path, recording_time=110, plot=False,
                        sdk_path='/Users/ashwin/FYP/sdk/myo.framework/myo'):
        """
        Records the sEMG sigals for the specified gestures and saves the
        recordings as .csv files
        :param gestures: The list of specified gesture types
        :param data_path: The directory where the recordings are saved
        :param recording_time: the duration of the recordings
        :param plot: plot the recorded signals or not (boolean)
        :param sdk_path: the MyoSDK path
        :return: None
        """
        for gesture in gestures:
            print("Recording gesture type : %s" % gesture)
            file = os.path.join(data_path, gesture + '.csv')
            self.record(save_path=file, recording_time=recording_time, sdk_path=sdk_path, plot=plot)

    def load_emg_data(self, data_path, gestures):
        """
        loads the sEMG recordings from the provided data path. The recorded multi-channel signals
        for each gesture type is contained in a dictionary where the keys represent
        gesture types. The shape of the dataset per each gesture type is
        (no_channels, no_samples)
        :param data_path: The directory where the recordings are saved
        :param gestures: The list of specified gesture types
        :return: the dictionary containing the multi-channel sEMG signal data for each gesture type
        """
        os.chdir(data_path)
        gesture_database = {}
        for gesture in gestures:
            data = read_from_csv(gesture + '.csv')
            gesture_database[gesture] = data
        return gesture_database

    def plot_recordings(self, data):
        """
        plot the entire sEMG recording for a selected gesture type
        :param data: multi-channel sEMG signal data for a selected gesture type.
        shape = (no_channels, no_samples)
        :return: None
        """
        plot_recording(data.T, self.fs)

    def plot_signals(self, signal, window=[0, 99], type='combined'):
        """
        plot the sEMG recordings of a selected gesture type
        :param signal: multi-channel sEMG signal data for a selected gesture type
        :param window: time window of interest to plot the signals
        :param type: plotting all channels in a single axis or plotting each
        channel in separate axes
        :return: None
        """

        if type == 'separate':
            fig = plt.figure()
            axes = [fig.add_subplot('%i1' % self.no_channels + str(i)) for i in range(0, self.no_channels)]
            [(ax.set_ylim([0, 50])) for ax in axes]
            t = np.arange(window[0], window[1], 1 / self.fs)
            i = 0
            for ax in axes:
                ax.plot(t, signal[i, window[0] * self.fs:window[1] * self.fs])
                i += 1
            plt.show()

        if type == 'combined':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_ylim([0, 1])
            t = np.arange(0, signal.shape[1] / self.fs, 1 / self.fs)
            for i in range(self.no_channels):
                ax.plot(t, signal[i, :])
            plt.legend(np.arange(1, self.no_channels + 1, 1))
            plt.show()

    def filter_signals(self, signal):
        """
        Extracts the envelopes of the multi-channel sEMG signal as described in [1]
        :param signal: The multi channel sEMG signal, shape = (no_channels, no_samples)
        :return: the signal envelopes of the multi-channel signals, shape = (no_channels, no_samples)
        """
        signal = np.abs(signal)  # full wave rectification
        lpf = butter(2, 1, 'lowpass', analog=False, fs=self.fs,
                     output='sos')  # define the low pass filter (Butterworth 2-order fc = 1Hz)
        filtered_signal = sosfilt(lpf, signal)
        return filtered_signal

    def filter_signal_database(self, gesture_database):
        """
        Extracts the envelopes of the multi-channel sEMG signals in the
        gesture database
        :param gesture_database: the dictionary containing the multi-channel sEMG signal data for each gesture type
        :return: the dictionary containing the envelopes of multi-channel sEMG signal data for each gesture type
        """
        gestures = list(gesture_database.keys())
        filtered_gesture_database = {}
        for gesture in gestures:
            signal = gesture_database[gesture]
            filtered_signal = self.filter_signals(signal)
            filtered_gesture_database[gesture] = filtered_signal
        return filtered_gesture_database

    # def get_time_series_emg_pattern(self, filtered_signal):
    #     """
    #     get the time series EMG pattern
    #     """
    #     X = filtered_signal / np.sum(filtered_signal, axis=0)
    #     return X
    #
    # def get_time_series_emg_pattern_database(self, filtered_gesture_database):
    #     """
    #     get the time series EMG pattern in the gesture database
    #     """
    #     gestures = list(filtered_gesture_database.keys())
    #     time_series_pattern_gesture_database = {}
    #     for gesture in gestures:
    #         filtered_signal = filtered_gesture_database[gesture]
    #         X = self.get_time_series_emg_pattern(filtered_signal)
    #         time_series_pattern_gesture_database[gesture] = X
    #     return time_series_pattern_gesture_database

    def non_linear_transform(self, X):
        """
        Generates the TMA map for the multi-channel signals of the
        specified time window using the non-linear transform described in [1].
        :param X: the multi-channel signals of the
        specified time window
        :return: the TMA map for the multi-channel signals of the specified time window
        """
        U = np.zeros((self.H, self.T))  # define the TMA map
        for t in range(self.T):
            ut = X[:, t]
            ut = ut.reshape(self.no_channels, 1)
            cov = ut * ut.T  # obtain the covariance matrix
            idx = np.triu_indices(self.no_channels)
            temp = np.concatenate((np.squeeze(ut), cov[idx]))  # removed 1
            U[:, t] = temp
        # normalize the first and second order terms separately
        # U1 = U[:self.no_channels + 1, :]
        # U[:self.no_channels + 1, :] = (U1 - U1.min()) / (U1.max() - U1.min())
        # U2 = U[self.no_channels + 1:, :]
        # U[self.no_channels + 1:, :] = (U2 - U2.min()) / (U2.max() - U2.min())
        U1 = U[:8, :]
        U[:8, :] = (U1 - U1.min()) / (U1.max() - U1.min())
        U2 = U[8:, :]
        U[8:, :] = (U2 - U2.min()) / (U2.max() - U2.min())
        return U

    def get_tma_maps(self, signal, obs_inc=0.100, plot=False):
        """
        Generates (offline) the time series of the TMA maps for a given multi-channel
        sEMG signal.
        :param signal:  multi-channel sEMG signal
        :param obs_inc: time difference (k) between two adjacent TMA maps
        :param plot: plot the time series of TMA maps. (Helps visually observe the
        changes of TMA maps over time)
        :return: the time series of the TMA maps for a given multi-channel
        sEMG signal.
        """
        i = 0
        if plot:
            fig = plt.figure()
        action_evolution_maps = np.zeros((1, self.H, self.T))
        while True:
            obs_start = int(obs_inc * self.fs * i)
            obs_end = int(obs_inc * self.fs * i + self.obs_dur * self.fs)
            time = (obs_start + obs_end) / (2.0 * self.fs)
            if obs_end > signal.shape[1]:
                break
            obs = signal[:, obs_start:obs_end]
            # U = obs
            U = self.non_linear_transform(obs)
            action_evolution_maps = np.concatenate((action_evolution_maps, U.reshape(1, self.H, self.T)), axis=0)
            if plot:
                ax = fig.add_subplot(111)
                ax.imshow(U, vmax=1, vmin=0, cmap='jet')
                fig.suptitle("Time : %0.2f" % time)
                if time == 5.4 or time == 10.4:
                    plt.pause(20)
                else:
                    plt.pause(0.001)
                m = plt.cm.ScalarMappable(cmap='jet')
                plt.colorbar(m, orientation='horizontal')
                ax.remove()
            i += 1
        return action_evolution_maps[1:, ...]

    def detect_onsets(self, signal, obs_inc, threshold, refractory_period, max_dur, plot=True, plot_diffs=False):
        """
        Detects (offline) the gesture onsets with the use of the time series of
        TMA maps.
        :param signal: the mulit-channel sEMG signal envelopes of a recording
        :param obs_inc: time difference (k) between two adjacent TMA maps
        :param threshold: specified threshold imposed to detect onsets using
        the difference signal d(n) as described in [1]
        :param refractory_period: the period (r) in which the onset detection is
        paused after a new onset is detected as described in [1]
        :param max_dur: the time point in the recording on which the
        onset detection should stop.
        :param plot: mark the detected onset points on the sEMG recording
        :param plot_diffs: plot the difference signal d(n) for the sEMG recording
        :return:
        """
        i = 0
        trans = []
        elapsed_time = refractory_period * self.fs
        Differences = []
        t = []
        while True:
            obs_start = int(obs_inc * self.fs * i)
            obs_end = int(obs_inc * self.fs * i + self.obs_dur * self.fs)
            time = obs_end
            elapsed_time += int(obs_inc * self.fs)
            if obs_end > max_dur * self.fs:
                break
            obs = signal[:, obs_start:obs_end]
            U = self.non_linear_transform(obs)
            # U = obs
            if time < 3 * self.fs:
                prev_U = U
                i += 1
                continue
            diff = np.linalg.norm(prev_U - U, ord='fro')
            t.append(time)
            Differences.append(diff)
            if diff > threshold and elapsed_time >= refractory_period * self.fs:
                trans.append(time)
                elapsed_time = 0
            prev_U = U
            i += 1

        print(trans)
        print("stdev : ", np.std(Differences))

        if plot:
            t = np.array(t)
            plt.plot(t / self.fs, Differences, 'y')
            for t in trans:
                plt.axvline(t / self.fs, ls='--')
            plt.axhline(3, ls='-', c='r')
            plt.xticks(np.arange(0, max(trans) / self.fs, 5))
            plt.xlabel("time(s)")
            plt.ylabel("$d(n)$")
            plt.legend(['difference signal', 'onset'])
            plt.show()

        return trans

    # def offline_recognition(self, signal, gesture_dict, model_path, obs_inc, sensitivity, refractory_period, max_dur,
    #                         plot=True, plot_diffs=False):
    #     """
    #     perform offline hand gesture recognition
    #     :param signal: the mulit-channel sEMG signal envelopes of a recording
    #     :param gesture_dict: dictionary with gesture types and names
    #     :param model_path: model path
    #     :param obs_inc: time difference (k) between two adjacent TMA maps
    #     :param sensitivity:
    #     :param refractory_period: the period (r) in which the onset detection is
    #     paused after a new onset is detected as described in [1]
    #     :param max_dur: the time point in the recording on which the
    #     predictions should stop.
    #     :param plot: mark the detected onset points on the sEMG recording
    #     :param plot_diffs:
    #     :return: plot the difference signal d(n) for the sEMG recording
    #     """
    #     sc = joblib.load(os.path.join(model_path, 'scaler.joblib'))
    #     clf = joblib.load(os.path.join(model_path, 'model.joblib'))
    #
    #     # from muscle_synergy_2.deep_learning_models import cnn
    #     # model = cnn((self.H, self.T, 1), 2)
    #     # model.load_weights(model_path)
    #
    #     i = 0
    #     trans = []
    #     elapsed_time = refractory_period * self.fs
    #     Differences = []
    #     t = []
    #     predictions = []
    #     D = deque(maxlen=5)
    #     prev_pred = 4
    #
    #     while True:
    #         obs_start = int(obs_inc * self.fs * i)
    #         obs_end = int(obs_inc * self.fs * i + self.obs_dur * self.fs)
    #         time = obs_end
    #         elapsed_time += int(obs_inc * self.fs)
    #         if obs_end > max_dur * self.fs:
    #             break
    #         obs = signal[:, obs_start:obs_end]
    #         O = self.non_linear_transform(obs)
    #         if time < 3 * self.fs:
    #             prev_O = O
    #             i += 1
    #             continue
    #         diff = np.linalg.norm(prev_O - O, ord='fro')
    #         t.append(time)
    #         Differences.append(diff)
    #         if len(D) == 5:
    #             thresh = np.mean(D)
    #             if diff > sensitivity and elapsed_time >= refractory_period * self.fs:
    #                 trans.append(time)
    #                 elapsed_time = 0
    #
    #                 # logic
    #                 if prev_pred == 0:
    #                     pred = 2
    #                 elif prev_pred == 1:
    #                     pred = 3
    #                 else:
    #                     # # CNN architecture
    #                     # U = O.reshape(1, self.H, self.T, 1)
    #                     # score = model.predict(U)
    #                     # if score >= 0.35:
    #                     #     pred = 1
    #                     # else:
    #                     #     pred = 0
    #
    #                     # SVM archiecture
    #                     fv = sc.transform(O.reshape(1, self.H * self.T))
    #                     pred = clf.predict(fv)
    #                     pred = pred[0]
    #
    #                 predictions.append(gesture_dict[pred])
    #
    #                 prev_pred = pred
    #
    #         D.append(diff)
    #         prev_O = O
    #         i += 1
    #
    #     if plot_diffs:
    #         plt.plot(t, Differences, 'y')
    #         plt.show()
    #
    #     if plot:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #         for t in trans:
    #             ax.axvline(t / self.fs)
    #         plt.xticks(np.arange(0, max(trans) / self.fs, 5))
    #         plt.show()
    #
    #     print(trans)
    #     print(predictions)
