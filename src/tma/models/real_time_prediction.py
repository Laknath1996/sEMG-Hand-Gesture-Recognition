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

objective : Contains the methods to execute the real-time prediction of
hand gestures using TMA maps of multi-channel sEMG signal recordings
(work in progress)

The code is based on the following paper :
[1] A. D. Silva, M. V. Perera, K. Wickramasinghe, A. M. Naim,
    T. Dulantha Lalitharatne and S. L. Kappel, "Real-Time Hand Gesture
    Recognition Using Temporal Muscle Activation Maps of Multi-Channel
    Semg Signals," ICASSP 2020 - 2020 IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP), Barcelona,
    Spain, 2020, pp. 1299-1303.
"""

from tma.models.nn_models import cnn

from threading import Lock
import myo

# myo.init('/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/sdk/myo.framework/myo')
import numpy as np
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class EmgCollectorPrediction(myo.DeviceListener):
    """
    Collects EMG data in a queue with *n* maximum number of elements.
    """

    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = []  # deque(maxlen=n)
        self.emg_stream = []

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))
            self.emg_stream.append((event.timestamp, event.emg))


class Predict(object):
    """
    onset detection and plotting
    """

    def __init__(self, listener, emgLearn, gesture_dict, cnn_model_path, thresh):
        # connection with the device
        self.n = listener.n
        self.listener = listener
        self.emgLearn = emgLearn

        # prediction properties
        self.prediction = 'No Gesture'
        self.thresh = thresh

        # set the model path here
        # cnn_model_path = 'models/subject_1001_Ashwin/model/CNN_2/cnn_model.h5'
        # svm_model_path = '/Users/ashwin/FYP/scripts/muscle_synergy_2/datasets/subject_1001_Ashwin/model/SVM_1'

        # self.sc = joblib.load(os.path.join(svm_model_path, 'scaler.joblib'))
        # self.clf = joblib.load(os.path.join(svm_model_path, 'model.joblib'))

        self.gesture_dict = dict([(value, key) for key, value in gesture_dict.items()])

        self.cnn = cnn((emgLearn.H, emgLearn.T, 1), 5)
        self.cnn.load_weights(cnn_model_path)

        self.start = time.time()

    def main(self):
        prev_pred = 6
        start_time = 0
        prev_O = 0
        start = True

        while True:
            emg_data = self.listener.get_emg_data()
            emg_data = np.array([x[1] for x in emg_data]).T

            if emg_data.shape[0] == 0:
                continue

            if emg_data.shape[1] >= self.n:
                if start:
                    print("statring..")
                    start = False

                emg_data = self.emgLearn.filter_signals(emg_data)

                obs = emg_data[:, -self.emgLearn.obs_dur*self.emgLearn.fs:]  # M = 80 (obs_dur = 0.4)

                O = self.emgLearn.non_linear_transform(obs)

                diff = np.linalg.norm(prev_O - O, ord=2)

                if diff > self.thresh and time.time() - start_time >= 3:
                    start_time = time.time()

                    # logic
                    if prev_pred == 0:
                        pred = 5
                    elif prev_pred == 1:
                        pred = 5
                    elif prev_pred == 2:
                        pred = 5
                    elif prev_pred == 3:
                        pred = 5
                    elif prev_pred == 4:
                        pred = 5
                    else:
                        ### CNN architecture
                        U = O.reshape(1, self.emgLearn.H, self.emgLearn.T, 1)
                        score = self.cnn.predict(U)
                        pred = np.argmax(score)

                        ### SVM archiecture
                        # fv = self.sc.transform(O.reshape(1, self.emgLearn.H*self.emgLearn.T))
                        # pred = self.clf.predict(fv)
                        # pred = pred[0]

                    self.prediction = self.gesture_dict[pred]

                    print(self.prediction)

                    prev_pred = pred

                prev_O = O
                time.sleep(0.100)  # k value between two adjacent TMA maps

# def main():
#     """
#     execute
#     """
#
#     model_path = 'models/subject_1001_Ashwin/model/cnn_model.h5'
#     myo.init('/Users/ashwin/FYP/sdk/myo.framework/myo')  # enter the path of the sdk/myo.famework/myo
#     hub = myo.Hub()
#     el = EmgLearn(fs=200,
#                   no_channels=8,
#                   obs_dur=0.400)
#     listener = EmgCollector(n=512)
#
#     gesture_dict = {
#         'Middle_Flexion': 0,
#         'Ring_Flexion': 1,
#         'Hand_Closure': 2,
#         'V_Flexion': 3,
#         'Pointer': 4,
#         'Neutral': 5,
#         'No_Gesture': 6
#     }
#
#     live = Predict(listener=listener,
#                    emgLearn=el,
#                    gesture_dict=gesture_dict,
#                    cnn_model_path=model_path)
#
#     with hub.run_in_background(listener.on_event):
#         live.main()
#
#     print("Closing...")
#
#
# if __name__ == '__main__':
#     main()
