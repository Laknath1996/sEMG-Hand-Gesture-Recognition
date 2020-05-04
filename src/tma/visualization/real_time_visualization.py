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

objective : Execute real-time hand gesture recognition using TMA maps of
multi-channel sEMG signals. Currently, this version of the code supports
only MyoArm Band (Thalamic Labs, Canada)

The code is based on the following paper :
[1] A. D. Silva, M. V. Perera, K. Wickramasinghe, A. M. Naim,
    T. Dulantha Lalitharatne and S. L. Kappel, "Real-Time Hand Gesture
    Recognition Using Temporal Muscle Activation Maps of Multi-Channel
    Semg Signals," ICASSP 2020 - 2020 IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP), Barcelona,
    Spain, 2020, pp. 1299-1303.
"""

import matplotlib.pyplot as plt

# plt.switch_backend('Qt4Agg')
# plt.style.use('dark_background')
from collections import deque
from threading import Lock
import myo

# myo.init('/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/sdk/myo.framework/myo')
import numpy as np
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class EmgCollectorEmgSignals(myo.DeviceListener):
    """
    Collects EMG data in a queue with *n* maximum number of elements.
    """

    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = []  # deque(maxlen=n)
        self.emg_stream = deque(maxlen=n)

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    def get_emg_queue(self):
        with self.lock:
            return list(self.emg_stream)

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))
            self.emg_stream.append((event.timestamp, event.emg))


class PlotEmgSignals(object):
    """
    onset detection and plotting
    """

    def __init__(self, listener, emgLearn, gesture_dict, conn):
        # connection with the device
        self.n = listener.n
        self.listener = listener
        self.emgLearn = emgLearn

        # figure properties
        self.fig = plt.figure(tight_layout=True)
        self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
        self.axes[0].set_title('sEMG Signals ($\mu V$)')
        [(ax.set_ylim([-200, 200])) for ax in self.axes]
        self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]

        self.start = time.time()

    def update_plot(self):
        # get data
        emg_data = self.listener.get_emg_queue()
        emg_data = np.array([x[1] for x in emg_data]).T

        # extract the signal envelope
        # emg_data = self.emgLearn.filter_signals(emg_data)

        # plot the data
        for g, data in zip(self.graphs, emg_data):
            if len(data) < self.n:
                data = np.concatenate([np.zeros(self.n - len(data)), data])
            g.set_ydata(data)

        plt.draw()

    def main(self):
        while True:
            self.update_plot()
            plt.pause(0.001)


class EmgCollectorDiffSignal(myo.DeviceListener):
    """
    Collects EMG data in a queue with *n* maximum number of elements.
    """

    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))


class PlotDiffSignal(object):
    """
    onset detection and plotting
    """

    def __init__(self, listener, emgLearn):
        self.n = listener.n
        self.listener = listener
        self.emgLearn = emgLearn

        self.fig = plt.figure(figsize=(10, 8), tight_layout=True)
        self.ax = self.fig.add_subplot('111')
        self.ax.set_ylim([0, 5])
        self.ax.set_title("Difference Signal - D(t)")
        self.ax.grid()
        self.graph = self.ax.plot(np.arange(self.n), np.zeros(self.n))[0]

        self.start = time.time()
        self.prediction = 'No Gesture'

        self.Differences = deque(maxlen=self.n)

    def update_difference_plot(self):
        if len(self.Differences) < self.n:
            self.Differences = np.concatenate([np.zeros(self.n - len(self.Differences)), self.Differences])
        self.graph.set_ydata(self.Differences)
        plt.pause(0.001)

    def main(self):
        D = deque(maxlen=5)
        Diff = deque(maxlen=self.n)
        print("statring..")
        while True:
            self.update_difference_plot()
            emg_data = self.listener.get_emg_data()
            emg_data = np.array([x[1] for x in emg_data]).T
            if emg_data.shape[0] == 0 or emg_data.shape[1] < 512:
                continue
            emg_data = self.emgLearn.filter_signals(emg_data)
            obs = emg_data[:, -40:]
            O = self.emgLearn.non_linear_transform(obs)
            if time.time() - self.start < 3:
                prev_O = O
                continue
            diff = np.linalg.norm(prev_O - O, ord=2)
            D.append(diff)
            Diff.append(diff)
            self.Differences = Diff
            prev_O = O


