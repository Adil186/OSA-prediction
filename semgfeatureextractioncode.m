clc; clear; close all;

%% Directories and Parameters
input_dir = 'D:\Adil Research work\OneDrive_2025-03-05\Sleep Apnea Prediction\Sleep Apnea Prediction\Pilot Study\Respiratory_Event_relation_between_ChinandEMG\ProcessedData\';
output_dir = 'D:\Adil Research work\OneDrive_2025-03-05\Sleep Apnea Prediction\Sleep Apnea Prediction\Pilot Study\Respiratory_Event_relation_between_ChinandEMG\Features\Features multiple lead time\preOSA_0_30 vs Normal\';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
fs = 256;
%% Feature Names
jfemgAmpFeatures = {'Enhanced Mean Absolute Value','Average Energy','Integrated EMG','Mean Absolute Value'};
% Pattern features from normalized signal:
jfemgPatternFeatures = {'Enhanced Wavelength','New Zero Crossing','Asolute Value of Summation of exp root','Absolute Value of Summation of Square Root','Mean Value of Square Root', ...
    'Log Teager Kaiser Energy Operator','Log Coefficient of Variation','Cardinality','Log Difference Absolute Standard Deviation', ...
    'Log Difference Absolute Mean Value','Difference Variance Value','V-Order','Temporal Moment','Difference Absolute Mean Value', ...
    'Autoregression Model T1','Autoregression Model T2','Autoregression Model T3','Autoregression Model T4','Mean Absolute Deviation', ...
    'Interquartile Range','Skewness','Kurtosis','Coefficient of Variation','Standard Deviation','Variance','Slope Sign Change', ...
    'Zero Crossing','Waveform Length','Average Amplitude Change','Difference Absolute Standard Deviation Value','Log Detector', ...
    'Modified Mean Absolute Value','Modified Mean Absolute Value 2','Myopulse Percentage Rate','Simple Square Integral', ...
    'Variance of EMG','Willison Amplitude','Maximum Fractal Length'};

timeFeatureNames = {'ShapeFactor','SNR','THD','SINAD','CrestFactor','ClearanceFactor','ImpulseFactor','PeakValue'};
freqFeatureNames = {'MeanFrequency','MedianFrequency','BandPower','OccupiedBandwidth','PowerBandwidth','PeakAmplitude','PeakLocation'};
featureNamesWithRaw = [{'Mean','RMS','Energy'}, jfemgAmpFeatures, jfemgPatternFeatures, timeFeatureNames, freqFeatureNames];

sFEt = signalTimeFeatureExtractor(SampleRate=fs, ShapeFactor=true, SNR=true, THD=true, SINAD=true, CrestFactor=true, ClearanceFactor=true, ImpulseFactor=true, PeakValue=true);
sFEf = signalFrequencyFeatureExtractor(SampleRate=fs, MeanFrequency=true, MedianFrequency=true, BandPower=true, OccupiedBandwidth=true, PowerBandwidth=true, PeakAmplitude=true, PeakLocation=true);

subjectFiles = dir(fullfile(input_dir, 'mesa-sleep-*_extracted_events.mat'));
combinedTable = table();

% Specify your preOSA lead time field names as used in your extraction code
lead_fields = {'preOSA_0_30','preOSA_30_60','preOSA_60_90','preOSA_90_120','preOSA_120_150'};

for subjIdx = 1:length(subjectFiles)
    load(fullfile(input_dir, subjectFiles(subjIdx).name), 'subject_results');
    fprintf('Extracting features for subject: %s\n', subjectFiles(subjIdx).name);
    % Demographics
    subjectID = subject_results.mesaid;
    nsrr_age = subject_results.nsrr_age;
    nsrr_bmi = subject_results.nsrr_bmi;
    nsrr_sex = subject_results.nsrr_sex;
    nsrr_race = subject_results.nsrr_race;
    nsrr_ahi_hp3r_aasm15 = subject_results.nsrr_ahi_hp3r_aasm15;
    wrksched5   = subject_results.wrksched5;
    smkstat5    = subject_results.smkstat5;
    slp_eff5    = subject_results.slp_eff5;
    types5      = subject_results.types5;
    epslpscl5c  = subject_results.epslpscl5c;
    insmnia5    = subject_results.insmnia5;
    rstlesslgs5 = subject_results.rstlesslgs5;
    slpapnea5   = subject_results.slpapnea5;
    cpap5       = subject_results.cpap5;
    dntaldv5    = subject_results.dntaldv5;
    uvula5      = subject_results.uvula5;
    site5c      = subject_results.site5c;
    overall5    = subject_results.overall5;
    quchin5     = subject_results.quchin5;

    %% --------- OSA events (All preOSA lead times) ----------
    for k = 1:length(subject_results.OSA_events)
        OSA_ev = subject_results.OSA_events(k);

        for lf = 1:length(lead_fields)
            lf_name = lead_fields{lf};
            if isfield(OSA_ev, lf_name)
                pre = OSA_ev.(lf_name);
                if ~isfield(pre, 'emg_ecg_rmoved'), continue; end
                sig_raw = pre.emg_ecg_rmoved(:);
                if isempty(sig_raw), continue; end

                % Feature extraction
                mean_raw = mean(sig_raw);
                rms_raw = rms(sig_raw);
                energy_raw = sum(sig_raw.^2);
                emav_raw = abs(jfemg('emav', sig_raw));
                ae_raw   = abs(jfemg('ae', sig_raw));
                iemg_raw = abs(jfemg('iemg', sig_raw));
                mav_raw  = abs(jfemg('mav', sig_raw));
                jfemgAmpVec = [emav_raw, ae_raw, iemg_raw, mav_raw];
                
                % Pattern features (normalized)
                sig = normalized(sig_raw);
                f1 = abs(jfemg('ewl', sig));
                f2 = abs(jfemg('fzc', sig));
                f3 = abs(jfemg('asm', sig));
                f4 = abs(jfemg('ass', sig));
                f5 = abs(jfemg('msr', sig));
                f6 = abs(jfemg('ltkeo', sig));
                f7 = abs(jfemg('lcov', sig));
                f8 = abs(jfemg('card', sig));
                f9 = abs(jfemg('ldasdv', sig));
                f10 = abs(jfemg('ldamv', sig));
                f11 = abs(jfemg('dvarv', sig));
                f12 = abs(jfemg('vo', sig, struct('order',2)));
                f13 = abs(jfemg('tm', sig, struct('order',3)));
                f14 = abs(jfemg('damv', sig));
                ar_vals = abs(jfemg('ar', sig, struct('order',4))); if iscolumn(ar_vals), ar_vals = ar_vals'; end
                f15 = ar_vals; % T1,T2,T3,T4
                f16 = abs(jfemg('mad', sig));
                f17 = abs(jfemg('iqr', sig));
                f18 = abs(jfemg('skew', sig));
                f19 = abs(jfemg('kurt', sig));
                f20 = abs(jfemg('cov', sig));
                f21 = abs(jfemg('sd', sig));
                f22 = abs(jfemg('var', sig));
                f23 = abs(jfemg('ssc', sig, struct('thres',0.01)));
                f24 = abs(jfemg('zc', sig, struct('thres',0.01)));
                f25 = abs(jfemg('wl', sig));
                f26 = abs(jfemg('aac', sig));
                f27 = abs(jfemg('dasdv', sig));
                f28 = abs(jfemg('ld', sig));
                f29 = abs(jfemg('mmav', sig));
                f30 = abs(jfemg('mmav2', sig));
                f31 = abs(jfemg('myop', sig, struct('thres',0.016)));
                f32 = abs(jfemg('ssi', sig));
                f33 = abs(jfemg('vare', sig));
                f34 = abs(jfemg('wa', sig, struct('thres',0.01)));
                f35 = abs(jfemg('mfl', sig));
                jfemgPatternVec = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, ...
                    f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35];

                time_feats = extract(sFEt, sig); time_feats = time_feats(:)';
                freq_feats = extract(sFEf, sig); freq_feats = freq_feats(:)';
                all_feats = [mean_raw, rms_raw, energy_raw, jfemgAmpVec, jfemgPatternVec, time_feats, freq_feats];

                annot_cells = {};
                if isfield(pre,'annotations')
                    for j = 1:length(pre.annotations)
                        annot_cells{end+1} = pre.annotations(j).EventConcept;
                    end
                end

                rowTable = table(...
                    subjectID, string(lf_name), pre.majority_sleep_stage, string(pre.majority_pos_label), {annot_cells}, ...
                    nsrr_age, nsrr_bmi, nsrr_sex, nsrr_race, nsrr_ahi_hp3r_aasm15, ...
                    wrksched5, smkstat5, slp_eff5, types5, epslpscl5c, insmnia5, rstlesslgs5, ...
                    slpapnea5, cpap5, dntaldv5, uvula5, site5c, overall5, quchin5, ...
                    'VariableNames', {'SubjectID','EventType','MajoritySleepStage','MajorityPosLabel','OverlappingAnnotations', ...
                    'nsrr_age','nsrr_bmi','nsrr_sex','nsrr_race','nsrr_ahi_hp3r_aasm15', ...
                    'wrksched5','smkstat5','slp_eff5','types5','epslpscl5c','insmnia5','rstlesslgs5','slpapnea5','cpap5','dntaldv5','uvula5','site5c','overall5','quchin5'});
                featTable = array2table(all_feats, 'VariableNames', featureNamesWithRaw);
                rowTable = [rowTable, featTable];
                combinedTable = [combinedTable; rowTable];
            end
        end
    end

    %% --------- Normal events (Normal window and preNormal window) ----------
    for k = 1:length(subject_results.Normal_events)
        ev = subject_results.Normal_events(k);
        % (A) Normal event window
        sig_raw = ev.emg_ecg_rmoved(:);
        if ~isempty(sig_raw)
            mean_raw = mean(sig_raw);
            rms_raw = rms(sig_raw);
            energy_raw = sum(sig_raw.^2);
            emav_raw = abs(jfemg('emav', sig_raw));
            ae_raw   = abs(jfemg('ae', sig_raw));
            iemg_raw = abs(jfemg('iemg', sig_raw));
            mav_raw  = abs(jfemg('mav', sig_raw));
            jfemgAmpVec = [emav_raw, ae_raw, iemg_raw, mav_raw];
              
            % Pattern features (normalized)
            sig = normalize(sig_raw);
            f1 = abs(jfemg('ewl', sig));
            f2 = abs(jfemg('fzc', sig));
            f3 = abs(jfemg('asm', sig));
            f4 = abs(jfemg('ass', sig));
            f5 = abs(jfemg('msr', sig));
            f6 = abs(jfemg('ltkeo', sig));
            f7 = abs(jfemg('lcov', sig));
            f8 = abs(jfemg('card', sig));
            f9 = abs(jfemg('ldasdv', sig));
            f10 = abs(jfemg('ldamv', sig));
            f11 = abs(jfemg('dvarv', sig));
            f12 = abs(jfemg('vo', sig, struct('order',2)));
            f13 = abs(jfemg('tm', sig, struct('order',3)));
            f14 = abs(jfemg('damv', sig));
            ar_vals = abs(jfemg('ar', sig, struct('order',4))); if iscolumn(ar_vals), ar_vals = ar_vals'; end
            f15 = ar_vals;
            f16 = abs(jfemg('mad', sig));
            f17 = abs(jfemg('iqr', sig));
            f18 = abs(jfemg('skew', sig));
            f19 = abs(jfemg('kurt', sig));
            f20 = abs(jfemg('cov', sig));
            f21 = abs(jfemg('sd', sig));
            f22 = abs(jfemg('var', sig));
            f23 = abs(jfemg('ssc', sig, struct('thres',0.01)));
            f24 = abs(jfemg('zc', sig, struct('thres',0.01)));
            f25 = abs(jfemg('wl', sig));
            f26 = abs(jfemg('aac', sig));
            f27 = abs(jfemg('dasdv', sig));
            f28 = abs(jfemg('ld', sig));
            f29 = abs(jfemg('mmav', sig));
            f30 = abs(jfemg('mmav2', sig));
            f31 = abs(jfemg('myop', sig, struct('thres',0.016)));
            f32 = abs(jfemg('ssi', sig));
            f33 = abs(jfemg('vare', sig));
            f34 = abs(jfemg('wa', sig, struct('thres',0.01)));
            f35 = abs(jfemg('mfl', sig));
            jfemgPatternVec = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, ...
                f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35];
            time_feats = extract(sFEt, sig); time_feats = time_feats(:)';
            freq_feats = extract(sFEf, sig); freq_feats = freq_feats(:)';
            all_feats = [mean_raw, rms_raw, energy_raw, jfemgAmpVec, jfemgPatternVec, time_feats, freq_feats];
            annot_cells = {};
            if isfield(ev, 'annotations')
                for j = 1:length(ev.annotations)
                    annot_cells{end+1} = ev.annotations(j).EventConcept;
                end
            end
            rowTable = table(...
                subjectID, {'Normal'}, ev.majority_sleep_stage, string(ev.majority_pos_label), {annot_cells}, ...
                nsrr_age, nsrr_bmi, nsrr_sex, nsrr_race, nsrr_ahi_hp3r_aasm15, ...
                wrksched5, smkstat5, slp_eff5, types5, epslpscl5c, insmnia5, rstlesslgs5, ...
                slpapnea5, cpap5, dntaldv5, uvula5, site5c, overall5, quchin5, ...
                'VariableNames', {'SubjectID','EventType','MajoritySleepStage','MajorityPosLabel','OverlappingAnnotations', ...
                'nsrr_age','nsrr_bmi','nsrr_sex','nsrr_race','nsrr_ahi_hp3r_aasm15', ...
                'wrksched5','smkstat5','slp_eff5','types5','epslpscl5c','insmnia5','rstlesslgs5','slpapnea5','cpap5','dntaldv5','uvula5','site5c','overall5','quchin5'});
            featTable = array2table(all_feats, 'VariableNames', featureNamesWithRaw);
            rowTable = [rowTable, featTable];
            combinedTable = [combinedTable; rowTable];
        end
        % (B) preNormal window (0–30s before normal event)
        if isfield(ev, 'pre_event')
            pre = ev.pre_event;
            sig_raw = pre.emg_ecg_rmoved(:);
            if ~isempty(sig_raw)
                mean_raw = mean(sig_raw);
                rms_raw = rms(sig_raw);
                energy_raw = sum(sig_raw.^2);
                emav_raw = abs(jfemg('emav', sig_raw));
                ae_raw   = abs(jfemg('ae', sig_raw));
                iemg_raw = abs(jfemg('iemg', sig_raw));
                mav_raw  = abs(jfemg('mav', sig_raw));
                jfemgAmpVec = [emav_raw, ae_raw, iemg_raw, mav_raw];

                % Pattern features (normalized)
                sig = normalized(sig_raw);
                f1 = abs(jfemg('ewl', sig));
                f2 = abs(jfemg('fzc', sig));
                f3 = abs(jfemg('asm', sig));
                f4 = abs(jfemg('ass', sig));
                f5 = abs(jfemg('msr', sig));
                f6 = abs(jfemg('ltkeo', sig));
                f7 = abs(jfemg('lcov', sig));
                f8 = abs(jfemg('card', sig));
                f9 = abs(jfemg('ldasdv', sig));
                f10 = abs(jfemg('ldamv', sig));
                f11 = abs(jfemg('dvarv', sig));
                f12 = abs(jfemg('vo', sig, struct('order',2)));
                f13 = abs(jfemg('tm', sig, struct('order',3)));
                f14 = abs(jfemg('damv', sig));
                ar_vals = abs(jfemg('ar', sig, struct('order',4))); if iscolumn(ar_vals), ar_vals = ar_vals'; end
                f15 = ar_vals;
                f16 = abs(jfemg('mad', sig));
                f17 = abs(jfemg('iqr', sig));
                f18 = abs(jfemg('skew', sig));
                f19 = abs(jfemg('kurt', sig));
                f20 = abs(jfemg('cov', sig));
                f21 = abs(jfemg('sd', sig));
                f22 = abs(jfemg('var', sig));
                f23 = abs(jfemg('ssc', sig, struct('thres',0.01)));
                f24 = abs(jfemg('zc', sig, struct('thres',0.01)));
                f25 = abs(jfemg('wl', sig));
                f26 = abs(jfemg('aac', sig));
                f27 = abs(jfemg('dasdv', sig));
                f28 = abs(jfemg('ld', sig));
                f29 = abs(jfemg('mmav', sig));
                f30 = abs(jfemg('mmav2', sig));
                f31 = abs(jfemg('myop', sig, struct('thres',0.016)));
                f32 = abs(jfemg('ssi', sig));
                f33 = abs(jfemg('vare', sig));
                f34 = abs(jfemg('wa', sig, struct('thres',0.01)));
                f35 = abs(jfemg('mfl', sig));
                jfemgPatternVec = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, ...
                    f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35];
                time_feats = extract(sFEt, sig); time_feats = time_feats(:)';
                freq_feats = extract(sFEf, sig); freq_feats = freq_feats(:)';
                all_feats = [mean_raw, rms_raw, energy_raw, jfemgAmpVec, jfemgPatternVec, time_feats, freq_feats];
                annot_cells = {};
                if isfield(pre, 'annotations')
                    for j = 1:length(pre.annotations)
                        annot_cells{end+1} = pre.annotations(j).EventConcept;
                    end
                end
                rowTable = table(...
                    subjectID, {'preNormal'}, pre.majority_sleep_stage, string(pre.majority_pos_label), {annot_cells}, ...
                    nsrr_age, nsrr_bmi, nsrr_sex, nsrr_race, nsrr_ahi_hp3r_aasm15, ...
                    wrksched5, smkstat5, slp_eff5, types5, epslpscl5c, insmnia5, rstlesslgs5, ...
                    slpapnea5, cpap5, dntaldv5, uvula5, site5c, overall5, quchin5, ...
                    'VariableNames', {'SubjectID','EventType','MajoritySleepStage','MajorityPosLabel','OverlappingAnnotations', ...
                    'nsrr_age','nsrr_bmi','nsrr_sex','nsrr_race','nsrr_ahi_hp3r_aasm15', ...
                    'wrksched5','smkstat5','slp_eff5','types5','epslpscl5c','insmnia5','rstlesslgs5','slpapnea5','cpap5','dntaldv5','uvula5','site5c','overall5','quchin5'});
                featTable = array2table(all_feats, 'VariableNames', featureNamesWithRaw);
                rowTable = [rowTable, featTable];
                combinedTable = [combinedTable; rowTable];
            end
        end
    end
end

% Save
writetable(combinedTable, fullfile(output_dir, 'AllLeadTimes_preOSA_preNormal_Normal_emgecgrmoved_Features_removedTHD.xlsx'));
save(fullfile(output_dir, 'AllLeadTimes_preOSA_preNormal_Normal_emgecgrmoved_Features_removedTHD.mat'), 'combinedTable');
fprintf('Saved combined feature table for %d segments.\n', height(combinedTable));
%%

% badRows = isinf(combinedTable.("Coefficient of Variation")) | isnan(combinedTable.("Coefficient of Variation"));
badRows = isinf(combinedTable.THD);
% Remove these rows
combinedTable(badRows, :) = [];

fprintf('Removed %d rows with Inf/NaN in THD.\n', sum(badRows));
%%
pairs = {
    "preOSA_0_30",  "Normal",     "preOSA_0_30_vs_Normal"
    "preOSA_30_60", "Normal",     "preOSA_30_60_vs_Normal"
    "preOSA_30_60", "preNormal",  "preOSA_30_60_vs_preNormal"
    "preOSA_60_90", "Normal",     "preOSA_60_90_vs_Normal"
    "preOSA_60_90", "preNormal",  "preOSA_60_90_vs_preNormal"
    "preOSA_90_120","Normal",     "preOSA_90_120_vs_Normal"
    "preOSA_90_120","preNormal",  "preOSA_90_120_vs_preNormal"
    "preOSA_120_150","Normal",    "preOSA_120_150_vs_Normal"
    "preOSA_120_150","preNormal", "preOSA_120_150_vs_preNormal"
    };

for i = 1:size(pairs,1)
    g1 = pairs{i,1};
    g2 = pairs{i,2};
    fname = pairs{i,3};
    idx_g1 = combinedTable.EventType == g1;
    idx_g2 = combinedTable.EventType == g2;
    tbl_g1 = combinedTable(idx_g1, :);
    tbl_g2 = combinedTable(idx_g2, :);

    pairTable = [tbl_g1; tbl_g2];
    % pairTable = pairTable(randperm(height(pairTable)), :);

    matpath = fullfile(output_dir, [char(fname) '.mat']);
    xlsxpath = fullfile(output_dir, [char(fname) '.xlsx']);
    save(matpath, 'pairTable');
    writetable(pairTable, xlsxpath);
    fprintf('Saved pair: %s (%d rows)\n', fname, height(pairTable));
end
