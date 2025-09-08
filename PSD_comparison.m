% Inputs:
% chin_emg_raw  - raw chin EMG signal vector
% chin_emg_pre  - preprocessed chin EMG signal vector
% gg_emg_raw    - raw GG EMG signal vector
% gg_emg_pre    - preprocessed GG EMG signal vector
% fs = 2000;    - sampling frequency (Hz)

% Define variables accordingly:
chin_emg_raw = Chin_emg_signal_Tongue_protusion_2kHz;
chin_emg_pre = chin';
gg_emg_raw = GG_emg_signal_Tongue_extend_right_2kHz;
gg_emg_pre = gg';
% fs = 2000;

%% PSD parameters
window_length = 1024;  % you can adjust
noverlap = window_length / 2;
nfft = 2048;           % frequency resolution

freq_limit = 100;      % max frequency to plot (adjust as needed)

%% Compute PSDs using pwelch

% Chin EMG raw
[pxx_chin_raw, f_chin_raw] = pwelch(chin_emg_raw, window_length, noverlap, nfft, fs);

% Chin EMG preprocessed
[pxx_chin_pre, f_chin_pre] = pwelch(chin_emg_pre, window_length, noverlap, nfft, fs);

% GG EMG raw
[pxx_gg_raw, f_gg_raw] = pwelch(gg_emg_raw, window_length, noverlap, nfft, fs);

% GG EMG preprocessed
[pxx_gg_pre, f_gg_pre] = pwelch(gg_emg_pre, window_length, noverlap, nfft, fs);

%% Plot PSDs for Chin EMG (before and after preprocessing)
figure('Name','Chin EMG Power Spectrum');
plot(f_chin_raw, 10*log10(pxx_chin_raw), 'b-', 'LineWidth', 1.5); hold on;
plot(f_chin_pre, 10*log10(pxx_chin_pre), 'r-', 'LineWidth', 1.5);
xlim([0 freq_limit]);
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('Chin EMG Power Spectrum (Active Phase-Tongue Protusion)');
legend('Before Preprocessing', 'After Preprocessing');
grid on;

%% Plot PSDs for GG EMG (before and after preprocessing)
figure('Name','Genioglossus EMG Power Spectrum');
plot(f_gg_raw, 10*log10(pxx_gg_raw), 'b-', 'LineWidth', 1.5); hold on;
plot(f_gg_pre, 10*log10(pxx_gg_pre), 'r-', 'LineWidth', 1.5);
xlim([0 freq_limit]);
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('Genioglossus EMG Power Spectrum (Active Phase-Tongue Protusion)');
legend('Before Preprocessing', 'After Preprocessing');
grid on;