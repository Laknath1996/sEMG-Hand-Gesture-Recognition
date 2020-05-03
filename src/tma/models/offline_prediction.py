"""
objective : Using TMA maps to recognize the hand gestures offline
            (work in progress)
author(s) : Ashwin de Silva
date      : 
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from tma.functions import *
from tma.models.nn_models import *
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd


def get_animated_difference_signal(x, y, title, onsets):
    x = np.array(x)
    y = np.array(y)

    y_o = np.zeros((len(y)))

    l = 0
    for n in range(len(y)):
        if onsets[l] == x[n]:
            y_o[n] = 15
            l = l + 1
            if l == 6:
                break

    x = x / 200

    signal = pd.DataFrame(y, x)
    signal.columns = {title}

    signal_o = pd.DataFrame(y_o, x)
    signal_o.columns = {title}

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=2000)
    fig = plt.figure(figsize=(10, 6))
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.axhline(6)
    plt.xlabel('time(s)', fontsize=20)
    plt.ylabel('$d(n)$', fontsize=20)
    plt.xticks(np.arange(0, 58, 5))

    def animate(i):
        data = signal.iloc[:int(i + 1)]
        p = sns.lineplot(x=data.index, y=data[title], data=data, color="b")
        p.tick_params(labelsize=15)
        plt.setp(p.lines, linewidth=1)

        data_o = signal_o.iloc[:int(i + 1)]
        q = plt.vlines(x=data_o.index, ymin=0, ymax=data_o[title], colors="r", linestyles="--")
        plt.setp(q, linewidth=1)

    ani = animation.FuncAnimation(fig, animate, frames=58 * 200, repeat=True)

    plt.show()


# def get_animated_emg_signal(Y, window):
#     Y = Y[:, window]
#     print(Y)
#
#     Writer = animation.writers['ffmpeg']
#     writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=2000)
#
#     fig = plt.figure(figsize=(10, 8), tight_layout=True)
#     axes = [fig.add_subplot('81' + str(i)) for i in range(1, 9)]
#     [(ax.set_ylim([0, 50])) for ax in axes]
#     [(ax.set_xlim([0, 500])) for ax in axes]
#
#     def animate(i):
#         idx = np.array(list(range(i + 1)))
#         # axes[1].plot(idx, np.squeeze(Y[1, idx]))
#         [(g.plot(idx, data)) for g, data in zip(axes, Y[:, idx])]
#
#     ani = animation.FuncAnimation(fig, animate, frames=500, repeat=True)
#
#     plt.show()


def offline_recognition(emgLearn, signal, gesture_dict, model_path, obs_inc, thresh, refractory_period, max_dur,
                        plot=True, plot_diffs=False):
    """
    perform offline hand gesture recognition
    :param signal: the mulit-channel sEMG signal envelopes of a recording
    :param gesture_dict: dictionary with gesture types and names
    :param model_path: model path
    :param obs_inc: time difference (k) between two adjacent TMA maps
    :param sensitivity:
    :param refractory_period: the period (r) in which the onset detection is
    paused after a new onset is detected as described in [1]
    :param max_dur: the time point in the recording on which the
    predictions should stop.
    :param plot: mark the detected onset points on the sEMG recording
    :param plot_diffs:
    :return: plot the difference signal d(n) for the sEMG recording
    """

    cnn_model = cnn((emgLearn.H, emgLearn.T, 1), 5)
    cnn_model.load_weights(model_path)

    i = 0
    onsets = []
    elapsed_time = refractory_period * emgLearn.fs
    Differences = []
    t = []
    predictions = []
    D = deque(maxlen=5)
    prev_pred = 6

    obs_end = 0

    while True:
        obs_start = int(obs_inc * emgLearn.fs * i)
        obs_end = int(obs_inc * emgLearn.fs * i + emgLearn.obs_dur * emgLearn.fs)
        time = obs_end
        elapsed_time += int(obs_inc * emgLearn.fs)
        if obs_end > max_dur * emgLearn.fs:
            break
        obs = signal[:, obs_start:obs_end]
        O = emgLearn.non_linear_transform(obs)
        if time < 3 * emgLearn.fs:
            prev_O = O
            i += 1
            continue
        diff = np.linalg.norm(prev_O - O, ord='fro')
        t.append(time)
        Differences.append(diff)
        if diff > thresh and elapsed_time >= refractory_period * emgLearn.fs:
            onsets.append(time)
            elapsed_time = 0

            ### CNN architecture
            U = O.reshape(1, emgLearn.H, emgLearn.T, 1)
            score = cnn_model.predict(U)
            pred = np.argmax(score)
            print(pred)

            print(gesture_dict[pred])
            predictions.append(gesture_dict[pred])

            prev_pred = pred

        D.append(diff)
        prev_O = O
        i += 1

    # if plot_diffs:
    #     plt.plot(t, Differences, 'y')
    #     plt.show()

    get_animated_difference_signal(t, Differences, "d(n)", onsets=onsets)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for t in onsets:
            ax.axvline(t / emgLearn.fs)
        plt.xticks(np.arange(0, max(onsets) / emgLearn.fs, 5))
        plt.show()

    print(onsets)
    print(predictions)


def main():
    """
    execute
    """

    model_path = '/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/models/subject_1001/model/cnn_model.h5'

    el = EmgLearn(fs=200,
                  no_channels=8,
                  obs_dur=0.400)

    gesture_dict = {
        0: 'Middle_Flexion',
        1: 'Ring_Flexion',
        2: 'Hand_Closure',
        3: 'V_Flexion',
        4: 'Pointer',
        5: 'Neutral',
        6: 'No_Gesture'
    }

    data_save_path = '/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/data/subject_1001'  # where the data should be saved
    flexion_gestures = ['M_2', 'R_2', 'HC_2', 'V_2', 'PO_2']  # gesture types

    # load the saved recordings
    gesture_database = el.load_emg_data(data_path=data_save_path,
                                        gestures=flexion_gestures)

    # extract the signal envelopes
    filtered_gesture_database = el.filter_signal_database(gesture_database)

    signal = filtered_gesture_database['R_2']

    # get_animated_emg_signal(signal, list(range(2 * 200, 58 * 200 + 1)))

    offline_recognition(emgLearn=el,
                        signal=signal,
                        gesture_dict=gesture_dict,
                        model_path=model_path,
                        obs_inc=0.15,
                        thresh=6,
                        refractory_period=8,
                        max_dur=58,
                        plot=True,
                        plot_diffs=True)

    print("Closing...")


if __name__ == '__main__':
    main()
