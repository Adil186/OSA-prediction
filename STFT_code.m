% --- Inputs (set these) ---
fs   = 2000;          % sampling rate (Hz)

% --- Match lengths & ensure column vectors ---
L    = min(length(gg), length(chin));
gg   = gg(1:L);
chin = chin(1:L);

% --- STFT / spectrogram params ---
winSec   = 0.512;                     % ~0.5 s -> 1024 samples at 2 kHz
wlen     = round(winSec * fs);        % window length (samples)
wlen     = max(256, min(wlen, L));    % guard for very short signals
win      = hamming(wlen, 'periodic'); % Hamming window
noverlap = round(0.5 * wlen);         % 50% overlap
if noverlap >= wlen, noverlap = max(0, wlen-1); end
nfft     = 2048;                      % ~0.98 Hz bin spacing when fs=2 kHz
fmax     = 100;                       % display up to 100 Hz (use 80 if preferred)

% --- Compute spectrograms (power -> dB) ---
[Sgg,Fg,Tg,Pgg] = spectrogram(gg,   win, noverlap, nfft, fs, 'yaxis');
[Sch,Fc,Tc,Pch] = spectrogram(chin, win, noverlap, nfft, fs, 'yaxis');
Sgg_dB = 10*log10(Pgg + eps);
Sch_dB = 10*log10(Pch + eps);

% --- Shared color scale for fair comparison (robust percentiles) ---
allDB = [Sgg_dB(:); Sch_dB(:)];
clims = [prctile(allDB,5), prctile(allDB,95)];

% --- Plot: GG ---
figure('Name','GG EMG Spectrogram (Rest)','Color','w');
imagesc(Tg, Fg, Sgg_dB); axis xy;
ylim([0 fmax]); xlim([Tg(1) Tg(end)]);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('GG EMG Spectrogram — Rest (whole signal)');
c = colorbar; ylabel(c,'Power (dB)');
clim(clims); 

grid on;

% --- Plot: Chin ---
figure('Name','Chin EMG Spectrogram (Rest)','Color','w');
imagesc(Tc, Fc, Sch_dB); axis xy;
ylim([0 fmax]); xlim([Tc(1) Tc(end)]);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Chin EMG Spectrogram — Rest (whole signal)');
c = colorbar; ylabel(c,'Power (dB)');
clim(clims); 

grid on;