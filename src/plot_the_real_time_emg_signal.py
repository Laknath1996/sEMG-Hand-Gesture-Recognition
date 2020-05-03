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

objective : Plot the real-time multi-channel sEMG signals. Currently, this
version of the code supports only MyoArm Band (Thalamic Labs, Canada)

The code is based on the following paper :
[1] A. D. Silva, M. V. Perera, K. Wickramasinghe, A. M. Naim,
    T. Dulantha Lalitharatne and S. L. Kappel, "Real-Time Hand Gesture
    Recognition Using Temporal Muscle Activation Maps of Multi-Channel
    Semg Signals," ICASSP 2020 - 2020 IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP), Barcelona,
    Spain, 2020, pp. 1299-1303.
"""


from tma.functions import *
import matplotlib.pyplot as plt

plt.switch_backend('Qt4Agg')
plt.style.use('dark_background')
from collections import deque
from threading import Lock
import myo

myo.init('/Users/ashwin/FYP/sdk/myo.framework/myo')
import numpy as np
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class EmgCollector(myo.DeviceListener):
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


class Plot(object):
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
        X = [(8, 2, 1), (8, 2, 3), (8, 2, 5), (8, 2, 7), (8, 2, 9), (8, 2, 11), (8, 2, 13), (8, 2, 15),
             (4, 2, (2, 4)), (4, 2, (6, 8))]  # subplots
        self.axes = [self.fig.add_subplot(r, c, pn) for r, c, pn in X]
        [(ax.set_ylim([-100, 100])) for ax in self.axes[:-2]]
        self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes[:-2]]

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


def main():
    """
    execute
    """

    myo.init('/Users/ashwin/FYP/sdk/myo.framework/myo')  # enter the path of the sdk/myo.famework/myo
    hub = myo.Hub()
    el = EmgLearn(fs=200,
                  no_channels=8,
                  obs_dur=0.400)
    listener = EmgCollector(n=512)

    with hub.run_in_background(listener.on_event):
        Plot(listener, el, gesture_dict=None, conn=None).main()


if __name__ == '__main__':
    main()
