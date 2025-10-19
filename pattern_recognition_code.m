%% ------------------------------------------------------------------------
% Plot Multi‑Modal PSG Signals with Sleep‑Stage Shading over any interval
% -------------------------------------------------------------------------

clc; clear; close all;

%% 0) User parameters — adjust as needed
edf_dir = 'D:\Adil Research work\OneDrive_2025-03-05\Sleep Apnea Prediction\Sleep Apnea Prediction\Pilot Study\edfs\';  % Folder containing EDF files
xml_dir = 'D:\Adil Research work\OneDrive_2025-03-05\Sleep Apnea Prediction\Sleep Apnea Prediction\Pilot Study\annotations-events-nsrr\';  % Folder containing XML annotation files
output_dir = 'D:\Adil Research work\OneDrive_2025-03-05\Sleep Apnea Prediction\Sleep Apnea Prediction\Pilot Study\';  % Folder to save processed MAT files

% List all EDF files in the directory
edf_files = dir(fullfile(edf_dir, '*.edf'));

subject  = 'mesa-sleep-2750';         % e.g. '1001'
t_start  = 19508;   %6547 , 20593         % start time in seconds
t_end    = 19704;    %6605 , 20630        %   end time in seconds

for i = 127:length(edf_files)
    % Get current EDF file and corresponding XML file
    edf_file = fullfile(edf_dir, edf_files(i).name);
    [~, baseName, ~] = fileparts(edf_files(i).name);
    xml_file = fullfile(xml_dir, [baseName, '-nsrr.xml']);
    
    fprintf('Processing subject: %s\n', baseName);
    fprintf('EDF file: %s\n', edf_file);
    fprintf('XML file: %s\n', xml_file);
    
    %% Step 1: Load EDF File and Extract Chin EMG Signal
    try
        [headers, data] = edfread(edf_file);
    catch ME
        warning('Error reading EDF file %s: %s. Skipping this subject.', edf_file, ME.message);
        continue;
    end

% find channel indices (modify to match your EDF labels)
iEMG   = headers.EMG;
iNasal = headers.Pres;
iFlow  = headers.Flow;
iThor  = headers.Thor;
iAbd   = headers.Abdo;
iSpO2  = headers.SpO2;
iTherm = headers.Therm;
% extract raw signals (column vectors)
fs_emg   = 256;   emg_raw1   = cell2mat(iEMG)';
fs_nasal = 32; nasal_raw = cell2mat(iNasal)';
fs_flow  = 32;  flow_raw  = cell2mat(iFlow)';
fs_thor  = 32;  thor_raw  =  cell2mat(iThor)';
fs_abd   = 32;   abd_raw   = cell2mat(iAbd)';
fs_spo2  = 1;  spo2_raw  =  iSpO2';
fs_therm = 32;   therm_raw = cell2mat(iTherm)';
[emg_ecg_rmoved] = preprocess_emg_complete_adv(emg_raw1, fs_emg, false, false);
emg_raw = emg_ecg_rmoved;
%emg_raw = normalize(emg_raw);
%% 2) Build time axes for each channel
t_emg   = (0:numel(emg_raw)-1)/fs_emg;
t_nasal = (0:numel(nasal_raw)-1)/fs_nasal;
t_flow  = (0:numel(flow_raw)-1)/fs_flow;
t_thor  = (0:numel(thor_raw)-1)/fs_thor;
t_abd   = (0:numel(abd_raw)-1)/fs_abd;
t_spo2  = (0:numel(spo2_raw)-1)/fs_spo2;
t_therm = (0:numel(therm_raw)-1)/fs_therm;
%% 3) Crop all channels to [t_start, t_end]
mask_emg   = t_emg   >= t_start & t_emg   <= t_end;
mask_nasal = t_nasal >= t_start & t_nasal <= t_end;
mask_flow  = t_flow  >= t_start & t_flow  <= t_end;
mask_thor  = t_thor  >= t_start & t_thor  <= t_end;
mask_abd   = t_abd   >= t_start & t_abd   <= t_end;
mask_spo2  = t_spo2  >= t_start & t_spo2  <= t_end;
mask_therm = t_therm  >= t_start & t_therm  <= t_end;

emg_raw   = emg_raw(mask_emg);    t_emg   = t_emg(mask_emg); emg_raw1 = emg_raw1(mask_emg);
nasal_raw = nasal_raw(mask_nasal);t_nasal = t_nasal(mask_nasal);
flow_raw  = flow_raw(mask_flow);  t_flow  = t_flow(mask_flow);
thor_raw  = thor_raw(mask_thor);  t_thor  = t_thor(mask_thor);
abd_raw   = abd_raw(mask_abd);    t_abd   = t_abd(mask_abd);
spo2_raw  = spo2_raw(mask_spo2);  t_spo2  = t_spo2(mask_spo2);
therm_raw = therm_raw(mask_therm); t_therm = t_therm(mask_therm);
%% 4) EMG preprocessing: bandpass 10–70 Hz + notch @60 Hz
[b_bp,a_bp]      = butter(4, [10 70]/(fs_emg/2), 'bandpass');
[b_notch,a_notch]= iirnotch(60/(fs_emg/2), (60/(fs_emg/2))/35);
emg_filt = filtfilt(b_bp, a_bp, emg_raw);
emg_filt = filtfilt(b_notch, a_notch, emg_filt);
% emg_filt_OSA1 = emg_filt;
% emg_filt = emg_normal_filt;
% Compute EMG envelope (<2 Hz) to compare with slow respiratory signals
env = abs(hilbert(emg_raw));
env_lp = lowpass(env, 2, fs_emg);

%% 5) Resample other signals to EMG rate (256 Hz)
[p_nasal,q_nasal] = rat(fs_emg/fs_nasal);
[p_flow,  q_flow ] = rat(fs_emg/fs_flow);
[p_thor,  q_thor ] = rat(fs_emg/fs_thor);
[p_abd,   q_abd  ] = rat(fs_emg/fs_abd);
[p_spo2,  q_spo2 ] = rat(fs_emg/fs_spo2);
[p_therm, q_therm] = rat(fs_emg/fs_therm);

nasal_rs = resample(nasal_raw, p_nasal, q_nasal);
flow_rs  = resample(flow_raw,  p_flow,  q_flow);
thor_rs  = resample(thor_raw,  p_thor,  q_thor);
abd_rs   = resample(abd_raw,   p_abd,   q_abd);
spo2_rs  = resample(spo2_raw,  p_spo2,  q_spo2);
therm_rs = resample(therm_raw,  p_therm,  q_therm);

% Trim/resync to EMG length
N = numel(env_lp);
nasal_rs = nasal_rs(1:N);
flow_rs  = flow_rs(1:N);
thor_rs  = thor_rs(1:N);
abd_rs   = abd_rs(1:N);
spo2_rs  = spo2_rs(1:N);
therm_rs = therm_rs(1:N);

t_common = (0:N-1)/fs_emg + t_start;  % common time axis

%% 6) Parse sleep‑stage annotations from XML
    xml_doc = xmlread(xml_file);
    event_nodes = xml_doc.getElementsByTagName('ScoredEvent');
    annotations = struct('EventType', [], 'EventConcept', [], 'Start', [], 'Duration', []);
    
    % Preallocate arrays for sleep stage and respiratory event annotations.
    sleep_stage_times = [];
    sleep_stage_durations = [];
    sleep_stage_labels = [];  % Numeric codes (e.g., Wake=0, Stage1=1, etc.)
    
    for j = 0:event_nodes.getLength-1
        event = event_nodes.item(j);
        event_type = char(event.getElementsByTagName('EventType').item(0).getTextContent);
        eventConcept = char(event.getElementsByTagName('EventConcept').item(0).getTextContent);
        start_time = str2double(event.getElementsByTagName('Start').item(0).getTextContent);
        duration = str2double(event.getElementsByTagName('Duration').item(0).getTextContent);
        annotations(j+1).EventType = event_type;
        annotations(j+1).EventConcept = eventConcept;
        annotations(j+1).Start = start_time;
        annotations(j+1).Duration = duration;
        
        % --- Sleep Stage Annotations ---
        if contains(event_type, 'Stages|Stages', 'IgnoreCase', true)
            sleep_stage_times(end+1) = start_time;
            sleep_stage_durations(end+1) = duration;
            if contains(eventConcept, 'Wake|0', 'IgnoreCase', true)
                sleep_stage_labels(end+1) = 0;
            elseif contains(eventConcept, 'Stage 1 sleep|1', 'IgnoreCase', true)
                sleep_stage_labels(end+1) = 1;
            elseif contains(eventConcept, 'Stage 2 sleep|2', 'IgnoreCase', true)
                sleep_stage_labels(end+1) = 2;
            elseif contains(eventConcept, 'Stage 3 sleep|3', 'IgnoreCase', true)
                sleep_stage_labels(end+1) = 3;
            elseif contains(eventConcept, 'REM sleep|5', 'IgnoreCase', true)
                sleep_stage_labels(end+1) = 4;
            else
                sleep_stage_labels(end+1) = NaN;
            end
        end
    end
%% 7) Plot signals with sleep‑stage shading
% build vector of stage at each common time point
stage_vec = nan(N,1);
for k = 1:numel(sleep_stage_times)
    idx = t_common >= sleep_stage_times(k) & ...
          t_common <  sleep_stage_times(k)+sleep_stage_durations(k);
    stage_vec(idx) = sleep_stage_labels(k);
end

%% 7) Plot in 2 subplots

figure;
% 7a) Hypnogram
subplot(9,1,1);
stairs(t_common, stage_vec, 'LineWidth',1.5);
ylim([-0.5 4.5]);
yticks(0:4);
yticklabels({'Wake','N1','N2','N3','REM'});
xlim([t_start t_end]);
xtickformat('%d')
title(sprintf('Subject %s: Sleep Stages [%d–%d s] Raw, Filtered EMG, Envelope & Respiratory Signals',subject,t_start,t_end));
grid on;

% 7d) Raw EMG
subplot(9,1,2);
plot(t_common, emg_raw1,'Color', [0.9290 0.6940 0.1250]);
xlim([t_start t_end]); 
ylabel('R-Chin(mV)');
grid on;

% 7b) ECG_Removed Chin EMG
subplot(9,1,3);
plot(t_common, emg_raw,'Color', [0.8500 0.3250 0.0980]);
xlim([t_start t_end]); ylabel('Chin(mV)-ECG');
grid on;

% 7b) Filtered EMG
subplot(9,1,4);
plot(t_common, emg_filt,'r');
xlim([t_start t_end]);
ylabel('Chin(mV)');
grid on;

% 7c) EMG Envelope
subplot(9,1,5);
plot(t_common, env_lp,'Color',[0.4940 0.1840 0.5560]);
xlim([t_start t_end]); 
ylabel('Chin(<2 Hz)');
grid on;
% 
% % 7d) Nasal pressure
% subplot(9,1,5);
% plot(t_common, nasal_rs,'Color', [0.9290 0.6940 0.1250]);
% xlim([t_start t_end]); 
% ylabel('Nasal');
% grid on;

% 7e) Airflow
subplot(9,1,6);
plot(t_common, flow_rs,'Color', [0.4660 0.6740 0.1880]);
xlim([t_start t_end]);
ylabel('Flow(cmH2O)');
grid on;

% 7f) Chest effort
subplot(9,1,7);
plot(t_common, thor_rs,'Color',[0.3010 0.7450 0.9330]);
xlim([t_start t_end]); 
ylabel('Chest(mV)');
grid on;

% 7g) Abdominal effort
subplot(9,1,8);
plot(t_common, abd_rs,'m');
xlim([t_start t_end]); 
ylabel('Abd(mV)');
grid on;

% 7h) SpO2
subplot(9,1,9);
plot(t_common, spo2_rs,'Color', [0.6350 0.0780 0.1840]);
xlim([t_start t_end]); 
ylabel('SpO₂(%)');
xlabel('Time (s)');
grid on;
%% Apply formatting to all x-axes in the current figure to remove scientific notation
all_axes_fig2 = findall(gcf, 'Type', 'axes');
for ax = all_axes_fig2' % Iterate through each axes object
    ax.XAxis.Exponent = 0; % Remove the common exponent (e.g., x10^4)
    xtickformat(ax, '%d'); % Format tick labels as integers
end
%legend('EMG (10–70Hz)','EMG env','Nasal','Flow','Chest','Abd','SpO₂','Location','eastoutside')

end
%%
% DWT Analysis of Chin EMG vs. Airflow (5 levels) with Subplots
% 0) Assumes you already have:
%   emg_filt   (Nx1), flow_rs   (Nx1), t_common (Nx1), fs_emg

% 1) Choose wavelet and levels
wavename = 'db4';
maxL     = wmaxlev(length(emg_raw), wavename);
nLevel   = min(7, maxL);   % e.g. 5 levels

%% 2) Perform DWT on EMG
[C_emg,L_emg] = wavedec(emg_raw, nLevel, wavename);
for j = 1:nLevel
    D_emg{j} = wrcoef('d', C_emg, L_emg, wavename, j);
end
A_emg = wrcoef('a', C_emg, L_emg, wavename, nLevel);

%% 3) Perform DWT on Flow
[C_flow,L_flow] = wavedec(flow_rs, nLevel, wavename);
for j = 1:nLevel
    D_flow{j} = wrcoef('d', C_flow, L_flow, wavename, j);
end
A_flow = wrcoef('a', C_flow, L_flow, wavename, nLevel);

%% 4) Plot original, details, and approximation in subplots
%title('Chin and Airflow signal during Normal Breathing Event');
figure;
% Row 1: original signals
subplot(nLevel+2,2,1);
plot(t_common, emg_raw, 'k');
title('RAW EMG: Original');
ylabel('Amp');
xlim([t_common(1) t_common(end)]); grid on;

subplot(nLevel+2,2,2);
plot(t_common, flow_rs, 'r');
title('Flow: Original');
ylabel('Amp');
xlim([t_common(1) t_common(end)]); grid on;

% Rows 2..nLevel+1: detail coefficients D1..Dn
for j = 1:nLevel
    row = j+1;
    % compute frequency band
    f_hi = fs_emg/2^(j);
    f_lo = fs_emg/2^(j+1);
    
    subplot(nLevel+2,2,2*(row-1)+1);
    plot(t_common, D_emg{j}, 'b');
    title(sprintf('EMG: D%d (%.1f–%.1f Hz)', j, f_lo, f_hi));
    ylabel('Amp');
    xlim([t_common(1) t_common(end)]); grid on;
    
    subplot(nLevel+2,2,2*(row-1)+2);
    plot(t_common, D_flow{j}, 'm');
    title(sprintf('Flow: D%d (%.1f–%.1f Hz)', j, f_lo, f_hi));
    ylabel('Amp');
    xlim([t_common(1) t_common(end)]); grid on;
end

% Last row: approximation A_n
row = nLevel+2;
subplot(nLevel+2,2,2*(row-1)+1);
plot(t_common, A_emg, 'b');
title(sprintf('EMG: A%d (0–%.1f Hz)', nLevel, fs_emg/2^nLevel));
ylabel('Amp');
xlabel('Time (s)');
xlim([t_common(1) t_common(end)]); grid on;

subplot(nLevel+2,2,2*(row-1)+2);
plot(t_common, A_flow, 'm');
title(sprintf('Flow: A%d (0–%.1f Hz)', nLevel, fs_emg/2^nLevel));
ylabel('Amp');
xlabel('Time (s)');
xlim([t_common(1) t_common(end)]); grid on;

%% Apply formatting to all x-axes in the current figure to remove scientific notation
all_axes_fig2 = findall(gcf, 'Type', 'axes');
for ax = all_axes_fig2' % Iterate through each axes object
    ax.XAxis.Exponent = 0; % Remove the common exponent (e.g., x10^4)
    xtickformat(ax, '%d'); % Format tick labels as integers
end

%% Helper function
function [emg_ecg_removed] = preprocess_emg_complete_adv(raw_emg, fs, plotEnabled, savePlots)
%function [emg_final, noise_segment, SNR_dB, f, P_clean] = preprocess_emg_complete_adv(raw_emg, fs, plotEnabled, savePlots)
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
% % Bandpass Filter (20-90 Hz used here for active muscle)
% bp_low = 10;
% bp_high = 70;
% [b_bp, a_bp] = butter(4, [bp_low, bp_high] / (fs/2), 'bandpass');
% emg_bp = filtfilt(b_bp, a_bp, raw_emg);
% 
% % Notch Filter at 50 Hz
% f0 = 60;
% Q = 10;
% bw = f0 / Q;
% [b_notch, a_notch] = iirnotch(f0 / (fs/2), bw / (fs/2));
% emg_notched = filtfilt(b_notch, a_notch, emg_bp);
emg_notched = raw_emg;
%% Step 2: ECG Artifact Removal via Template Subtraction with Amplitude Scaling
diff_emg = diff(emg_notched);
squared_emg = diff_emg .^ 2;

windowSize = round(0.150 * fs);
integrationWindow = ones(1, windowSize) / windowSize;
integrated_emg = conv(squared_emg, integrationWindow, 'same');

% try
%     [~, qrs_i_raw] = pan_tompkin_revised(emg_notched, fs, 0);
% catch
%warning('pan_tompkin_revised not available. Using findpeaks instead.');
minPeakDistance = round(0.3 * fs);
minPeakHeight = 0.5 * max(integrated_emg);
[~, qrs_i_raw] = findpeaks(integrated_emg, 'MinPeakDistance', minPeakDistance, 'MinPeakHeight', minPeakHeight);
% end

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
% emg_final = wdenoise(emg_ecg_removed, 3, 'Wavelet', 'db4', ...
%     'DenoisingMethod', 'Bayes', 'ThresholdRule', 'Hard', 'NoiseEstimate', 'LevelDependent');
% 
% noise_segment = emg_final - emg_ecg_removed;
% 
% if plotEnabled
%     fig4 = figure;
%     plot(t, emg_ecg_removed, 'b-', t, emg_final, 'r-');
%     xlabel('Time (s)'); ylabel('Amplitude');
%     legend('ECG Removed EMG', 'Denoised EMG');
%     title('Comparison Before and After Denoising');
%     if savePlots
%         saveas(fig4, 'Wavelet_Denoising_Comparison.png');
%     end
% end

%% Compute SNR (in dB)
% SNR_dB = 10 * log10(var(emg_final) / var(noise_segment));
% %disp(['SNR after denoising: ', num2str(SNR_dB), ' dB']);
% 
% % Step 4: FFT of Final Signal
% L = length(emg_final);
% Y_clean = fft(emg_final);
% P_clean = abs(Y_clean / L);
% f = fs * (0:(L/2)) / L;
% 
% if plotEnabled
%     fig5 = figure;
%     plot(f, P_clean(1:floor(L/2) + 1));
%     xlabel('Frequency (Hz)');
%     ylabel('|P1(f)|');
%     title('Frequency Spectrum After Filtering');
%     xlim([0 90]);
%     if savePlots
%         saveas(fig5, 'Frequency_Spectrum.png');
%     end
% end


end
