%% onset detection based on time-varying graph-frequency

clear all
close all

%% params
Fs = 200;           % sampling frequency
fc = 1;             % cutoff frequency
sel_shift = 0;      % 0 for A, 1 for L

%% load the signal
file = 'subject_1002/M_1.csv';
S = readtable(file);
S = S(:, 2:end);
S = table2array(S);

%% plot the signal from the first electrode
plot_multi_signals(S, [1, 40], Fs); 

%% pre-processing steps (rectification + low pass filtering)
S = abs(S);
[z, p, k] = butter(2, fc/(Fs/2));
sos = zp2sos(z, p, k);
S_filt = sosfilt(sos, S);

%% plot the signal from the first electrode
plot_multi_signals(S_filt, [1, 40], Fs); 

%% load the adjacency matrix
load adj_matrix.mat

%% get the graph frequencies and spectral components
if sel_shift == 0
    [V, D] = eig(A);    % frequency based on adj matrix
else
    d = sum(A, 1);
    D = diag(d);
    L = D-A;
    [V, D] = eig(L);    % frequency based on laplace matrix
end

% % mistake
% disp(V);
% V_o = V;
V(V < 1e-5) = 0;
D(D < 1e-5) = 0;
% disp(V);

%% determine the ascending order of graph frequencies
if sel_shift == 0    
    A_norm = (1/max(diag(D)))*A;
else
    A_norm = (1/max(diag(D)))*L;
end

TV = [];
for k=1:size(A, 1)
    v_k = V(:, k); 
    TV_k = norm(v_k - A_norm*v_k, 1);
    TV(k) = TV_k;
end

lambda = diag(D);
TV = [TV' lambda];

lambda_sorted = sortrows(TV, 1);
lambda_sorted = lambda_sorted(:, 2);

%% compute the graph time-varying Graph Frequencies
if sel_shift == 0
    F = inv(V);             % Graph Fourier Transform Matrix based on adj matrix
else
    F = V;                  % Graph Fourier Transform Matrix
end

% obtaining the GFT of the pre-processed multi-channel sEMG signal
S_hat = F*S_filt';
plot_multi_signals(S_hat', [1, 40], Fs);



