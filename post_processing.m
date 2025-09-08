function [emg_final, noise_segment, SNR_dB, f, P_clean] = preprocess_emg_complete_adv(raw_emg, fs, plotEnabled, savePlots)
% preprocess_emg Preprocess an EMG signal with filtering, ECG clearance, and denoising.
%
%   [emg_final, noise_segment, SNR_dB, f, P_clean] = preprocess_emg(raw_emg, fs, plotEnabled, savePlots)
%
%   INPUTS:
%       raw_emg    - Raw EMG signal vector.
%       fs         - Sampling frequency in Hz.
%       plotEnabled- (Optional) Boolean flag: true to display plots. Default false.
%       savePlots  - (Optional) Boolean flag: true to save plots as PNG files. Requires plotEnabled true.
%                    Default false.
%
%   OUTPUTS:
%       emg_final     - Final cleaned EMG signal after denoising.
%       noise_segment - Noise computed as the difference between pre-denoised and denoised signals.
%       SNR_dB        - Signal-to-noise ratio in dB after denoising.
%       f             - Frequency vector for FFT.
%       P_clean       - Single-sided amplitude spectrum of the cleaned signal.
%Processing includes:
%    1. Bandpass filtering (10–80 Hz) or 20-80 Hz for the activity and 50 Hz notch filtering.
%    2. ECG artifact removal via template subtraction:
%       - R peak detection (using pan_tompkin_revised or alternative).
%       - Extract segments around detected R-peaks.
%       - Form an average ECG template.
%       - Subtract the template from each corresponding segment.
%    3. Wavelet denoising (using wdenoise).
%    4. FFT computation of the final signal.

if nargin < 3
    plotEnabled = false;
end
if nargin < 4
    savePlots = false;
end

%% Clean raw_emg for NaN and Inf values before filtering
if any(isnan(raw_emg)) || any(isinf(raw_emg))
    warning('Signal contains NaN or Inf values. Cleaning...');
    raw_emg_clean = raw_emg;

    % Replace Infs with NaN for uniform handling
    raw_emg_clean(isinf(raw_emg_clean)) = NaN;

    % Find NaNs
    nanIdx = isnan(raw_emg_clean);

    if any(nanIdx)
        % Interpolate linearly to fill NaNs
        raw_emg_clean(nanIdx) = interp1(find(~nanIdx), raw_emg_clean(~nanIdx), find(nanIdx), 'linear', 'extrap');
    end

    raw_emg = raw_emg_clean;
end

%% Step 1: Filtering
% Bandpass Filter (20-80 Hz used here for active muscle)
bp_low = 20;
bp_high = 80;
[b_bp, a_bp] = butter(4, [bp_low, bp_high] / (fs/2), 'bandpass');
emg_bp = filtfilt(b_bp, a_bp, raw_emg);

% Notch Filter at 50 Hz
f0 = 50;
Q = 10;
bw = f0 / Q;
[b_notch, a_notch] = iirnotch(f0 / (fs/2), bw / (fs/2));
emg_notched = filtfilt(b_notch, a_notch, emg_bp);

%% Step 2: ECG Artifact Removal via Template Subtraction with Amplitude Scaling
diff_emg = diff(emg_notched);
squared_emg = diff_emg .^ 2;

windowSize = round(0.150 * fs);
integrationWindow = ones(1, windowSize) / windowSize;
integrated_emg = conv(squared_emg, integrationWindow, 'same');

try
    [~, qrs_i_raw] = pan_tompkin_revised(emg_notched, fs, 0);
catch
    warning('pan_tompkin_revised not available. Using findpeaks instead.');
    minPeakDistance = round(0.3 * fs);
    minPeakHeight = 0.5 * max(integrated_emg);
    [~, qrs_i_raw] = findpeaks(integrated_emg, 'MinPeakDistance', minPeakDistance, 'MinPeakHeight', minPeakHeight);
end

disp(['Detected R-peaks: ', num2str(length(qrs_i_raw))]);
if plotEnabled
    t = (1:length(emg_notched)) / fs;
    figure;
    plot(t, emg_notched, 'b-');
    hold on;
    plot(qrs_i_raw/fs, emg_notched(qrs_i_raw), 'ro');
    xlabel('Time (s)'); ylabel('Amplitude');
    title('Detected R-Peaks in Preprocessed EMG');
    if savePlots
        saveas(gcf, 'Detected_RPeaks.png');
    end
end

winDuration = 0.2; % seconds
winLen = round(winDuration * fs);
halfWin = round(winLen/2);

ecgSegments = [];
for k = 1:length(qrs_i_raw)
    idxStart = qrs_i_raw(k) - halfWin;
    idxEnd = qrs_i_raw(k) + halfWin - 1;
    if idxStart < 1 || idxEnd > length(emg_notched)
        continue;
    end
    segment = emg_notched(idxStart:idxEnd);
    ecgSegments = [ecgSegments; segment(:)'];
end

disp('Size of ECG segments matrix:');
disp(size(ecgSegments));

if isempty(ecgSegments)
    error('No ECG segments detected. Adjust detection parameters or winDuration.');
end

ecgTemplate = mean(ecgSegments, 1);

if plotEnabled
    t_template = (1:length(ecgTemplate)) / fs;
    fig2 = figure;
    plot(t_template, ecgTemplate);
    xlabel('Time (s)'); ylabel('Amplitude');
    title('Estimated ECG Template');
    if savePlots
        saveas(fig2, 'ECG_Template.png');
    end
end

emg_ecg_removed = emg_notched;

if mod(winLen, 2) == 1
    winLen = winLen + 1;
end
halfWin = winLen / 2;

for k = 1:length(qrs_i_raw)
    idxStart = qrs_i_raw(k) - halfWin;
    idxEnd = qrs_i_raw(k) + halfWin - 1;
    if idxStart < 1 || idxEnd > length(emg_notched)
        continue;
    end

    segment = emg_ecg_removed(idxStart:idxEnd);
    segment = segment(:)';

    if length(segment) ~= length(ecgTemplate)
        xq = linspace(1, length(ecgTemplate), length(segment));
        ecgTemplate_adj = interp1(1:length(ecgTemplate), ecgTemplate, xq, 'linear');
    else
        ecgTemplate_adj = ecgTemplate;
    end

    a = (segment * ecgTemplate_adj') / (ecgTemplate_adj * ecgTemplate_adj');
    corrected_segment = segment - a * ecgTemplate_adj;

    emg_ecg_removed(idxStart:idxEnd) = corrected_segment;
end

if plotEnabled
    t = (1:length(emg_notched)) / fs;
    fig3 = figure;
    plot(t, emg_notched, 'b-', t, emg_ecg_removed, 'r-');
    xlabel('Time (s)'); ylabel('Amplitude');
    legend('Notched EMG', 'ECG Removed EMG');
    title('EMG Before and After ECG Artifact Removal');
    if savePlots
        saveas(fig3, 'ECG_Removal_Comparison.png');
    end
end

%% Step 3: Wavelet Denoising
emg_final = wdenoise(emg_ecg_removed, 3, 'Wavelet', 'db4', ...
    'DenoisingMethod', 'Bayes', 'ThresholdRule', 'Hard', 'NoiseEstimate', 'LevelDependent');

noise_segment = emg_final - emg_ecg_removed;

if plotEnabled
    fig4 = figure;
    plot(t, emg_ecg_removed, 'b-', t, emg_final, 'r-');
    xlabel('Time (s)'); ylabel('Amplitude');
    legend('ECG Removed EMG', 'Denoised EMG');
    title('Comparison Before and After Denoising');
    if savePlots
        saveas(fig4, 'Wavelet_Denoising_Comparison.png');
    end
end

%% Compute SNR (in dB)
SNR_dB = 10 * log10(var(emg_final) / var(noise_segment));
disp(['SNR after denoising: ', num2str(SNR_dB), ' dB']);

%% Step 4: FFT of Final Signal
L = length(emg_final);
Y_clean = fft(emg_final);
P_clean = abs(Y_clean / L);
f = fs * (0:(L/2)) / L;

if plotEnabled
    fig5 = figure;
    plot(f, P_clean(1:floor(L/2) + 1));
    xlabel('Frequency (Hz)');
    ylabel('|P1(f)|');
    title('Frequency Spectrum After Filtering');
    xlim([0 500]);
    if savePlots
        saveas(fig5, 'Frequency_Spectrum.png');
    end
end
end