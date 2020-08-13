function plot_signal(S, win, Fs)
    figure;
    t = win(1):1/Fs:win(2);
    T = S(Fs*win(1):Fs*win(2));
    plot(t, T(:, 1));