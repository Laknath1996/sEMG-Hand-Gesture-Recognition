function plot_multi_signals(S, win, Fs)
    figure;
    t = win(1):1/Fs:win(2);
    T = S(Fs*win(1):Fs*win(2), :);
    subplot(811); plot(t, T(:, 1));
    subplot(812); plot(t, T(:, 2));
    subplot(813); plot(t, T(:, 3));
    subplot(814); plot(t, T(:, 4));
    subplot(815); plot(t, T(:, 5));
    subplot(816); plot(t, T(:, 6));
    subplot(817); plot(t, T(:, 7));
    subplot(818); plot(t, T(:, 8));
    