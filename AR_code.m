GG_emg_signal = data(datastart(1,1):dataend(1,1));
Chin_emg_signal =  data(datastart(2,1):dataend(2,1));
%Chin_emg_signal = data;
% Load or define your raw EMG signal and set sampling frequency, e.g.:
% load('your_emg_data.mat'); % raw_emg must be defined in the loaded file
fs = 2000;  % Sampling frequency in Hz

% Process the signal with plots enabled and saved to files
[gg, noise_segment, SNR_dB, f, P_clean] = preprocess_emg_complete_adv(GG_emg_signal_Tongue_Protusion_2kHz', fs, false, false);
[chin, noise_segment, SNR_dB, f, P_clean] = preprocess_emg_complete_adv(Chin_emg_signal_Tongue_protusion_2kHz', fs, false, false);
%% Step 1: Estimate AR models for both signals
% Choose an AR model order (this is a parameter you'll have to tune)
order = 4;  % for example, order 4
%AIC to select the correct the order
% Estimate AR coefficients using aryule
[a_chin, E_chin] = aryule(chin, order);
[a_gg, E_gg]     = aryule(gg, order);

%% Step 2: Compute Frequency Responses from the AR models
% In an AR model, the frequency response is given by 1/A(e^(jomega)).
% Define frequency vector for evaluation
nfft = 2^nextpow2(length(chin));  % or choose a value suitable for your resolution

% Calculate frequency response for chin and GG signals
[H_chin, f] = freqz(1, a_chin, nfft, fs);
[H_gg, ~]   = freqz(1, a_gg, nfft, fs);

%% Step 3: Compute the Parametric Transfer Function
% Using the derivation above, the transfer function relationship can be formed as:
% H_est = H_gg / H_chin 
H_est = H_gg ./ H_chin;

%% Step 4: Plot the Transfer Function Components in the Frequency Band of Interest
% Here we limit the plots to 10-80 Hz as your band of interest.
figure;
subplot(2,1,1);
plot(f, abs(H_est), 'b-');
xlabel('Frequency (Hz)');
ylabel('Magnitude (Gain)');
title('Parametric Transfer Function Magnitude Estimate');
grid on;
xlim([20 80]);

subplot(2,1,2);
plot(f, angle(H_est), 'g-');
xlabel('Frequency (Hz)');
ylabel('Phase (radians)');
title('Parametric Transfer Function Phase Estimate');
grid on;
xlim([20 80]);

% Assuming your single, long signal is in the variable 'Chin_emg_signal'
% and the other is in 'GG_emg_signal'
% Note: The transfer function 'H_est' is assumed to be already calculated.

%% Step 5: Process the long signal in smaller chunks

% Define the segment length in seconds (e.g., 30 seconds is a good starting point)
segment_duration_sec = 1;
segment_length = fs * segment_duration_sec;

% Create an array to store the estimated signal
gg_estimated_signal_full = zeros(size(chin));

% Loop through the signal in chunks
for i = 1:segment_length:length(chin)
    
    % Define the start and end indices of the current segment
    end_idx = i + segment_length - 1;
    if end_idx > length(chin)
        end_idx = length(chin);
    end
    current_chin_segment = chin(i:end_idx);

    % Use a consistent nfft for the current segment
    nfft_current = 2^nextpow2(length(current_chin_segment)); 

    % Get the frequency domain representation of the current chin signal
    fft_chin = fft(current_chin_segment, nfft_current); 

    % Interpolate the transfer function to match the FFT length
    % 'f' and 'H_est' are the variables you calculated earlier
    H_est_interp = interp1(f, H_est, (0:nfft_current-1)*(fs/nfft_current), 'linear', 'extrap');

    % Perform the frequency-domain multiplication
    H_gg_estimated = fft_chin' .* H_est_interp;

    % Convert the estimated GG EMG frequency response back to the time domain
    gg_current_estimated = ifft(H_gg_estimated);
    
    % Store the result, trimming back to the original length
    gg_estimated_signal_full(i:end_idx) = real(gg_current_estimated(1:length(current_chin_segment)));
end

%% Step 6: Plot the results to compare with the original GG EMG signal

% % Plot the original and estimated signals (you can zoom in to see the details)
% figure;
% subplot(2,1,1);
% plot(gg, 'b-');
% title('Original Genioglossus (GG) EMG Signal');
% xlabel('Samples');
% ylabel('Amplitude');
% grid on;
% 
% subplot(2,1,2);
% plot(gg_estimated_signal_full, 'r-');
% title('Estimated Genioglossus (GG) EMG Signal');
% xlabel('Samples');
% ylabel('Amplitude');
% grid on;

% Assuming your sampling frequency 'fs' is 2000 Hz.
fs = 2000;

% Create a time vector 't' in seconds
% It will have the same number of elements as your signal
t = (0:length(gg)-1) / fs;

figure;
plot(t, gg, 'b-', 'LineWidth', 1);
hold on; 

plot(t, gg_estimated_signal_full, 'r--', 'LineWidth', 1);

legend('Original GG EMG', 'Estimated GG EMG');

title('Comparison of Original and Estimated Genioglossus (GG) EMG Signals Over Time','FontSize',10);
xlabel('Time (s)');
ylabel('Amplitude');
%grid on;
hold off;
