"""
objective : plot the difference signal in real-time
author(s) : Ashwin de Silva
date      : 
"""

from matplotlib import pyplot as plt

plt.style.use('dark_background')
import myo

myo.init('/Users/ashwin/FYP/sdk/myo.framework/myo')
from tma.functions import *


class EmgCollector(myo.DeviceListener):
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


class Plot(object):
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


def main():
    """
    execute
    """
    myo.init('/Users/ashwin/FYP/sdk/myo.framework/myo')  # enter the path of the sdk/myo.famework/myo
    hub = myo.Hub()
    el = EmgLearn(fs=200, no_channels=8, obs_dur=0.2)
    listener = EmgCollector(n=512)
    with hub.run_in_background(listener.on_event):
        Plot(listener, el).main()


if __name__ == '__main__':
    main()
