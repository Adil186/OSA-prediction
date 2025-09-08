clc; clear; close all;

%% 0) Setup and file list
edf_dir   = 'D:\Adil Research work\mesa\edfs\';
xml_dir   = 'D:\Adil Research work\mesa\annotations-events-nsrr\';
output_dir= 'D:\Adil Research work\OneDrive_2025-03-05\Sleep Apnea Prediction\Sleep Apnea Prediction\Pilot Study\Respiratory_Event_relation_between_ChinandEMG\ProcessedData - Original without change_multiple_leadtime';
demog_file= 'D:\Adil Research work\OneDrive_2025-03-05\Sleep Apnea Prediction\Sleep Apnea Prediction\Sleep Apnea Prediction\datasets\mesa-sleep-harmonized-dataset-0.7.0.csv';

edf_files = dir(fullfile(edf_dir, '*.edf'));
demog_tbl = readtable(demog_file);

for subj_idx = 1755:length(edf_files)
    %% Load signals
    edf_path = fullfile(edf_dir, edf_files(subj_idx).name);
    [~, baseName, ~] = fileparts(edf_files(subj_idx).name);
    xml_path = fullfile(xml_dir, [baseName, '-nsrr.xml']);
    fprintf('Processing subject: %s\n', baseName);

    try
        [headers, data] = edfread(edf_path);
    catch ME
        warning('Could not read EDF: %s'); continue;
    end

    % Channels
    iEMG   = headers.EMG;
    iNasal = headers.Pres;
    iFlow  = headers.Flow;
    iThor  = headers.Thor;
    iAbd   = headers.Abdo;
    iSpO2  = headers.SpO2;
    fields = fieldnames(headers);
    therm_names = {'therm', 'Therm', 'AUXac', 'Aux_AC'};
    iTherm = [];
    for ii = 1:length(therm_names)
        idx_therm = find(strcmpi(therm_names{ii}, fields), 1);
        if ~isempty(idx_therm)
            iTherm = headers.(fields{idx_therm});
            break;
        end
    end
    if isempty(iTherm)
        warning('No thermistor/AUXac channel found for subject %s.', baseName);
        therm_raw = [];
    else
        therm_raw = cell2mat(iTherm)';
    end
    fs_emg = 256;

    % --- Signals (all full-night, pre-allocate to EMG length if needed) ---
    emg_raw   = cell2mat(iEMG)';
    nasal_raw = cell2mat(iNasal)';
    flow_raw  = cell2mat(iFlow)';
    thor_raw  = cell2mat(iThor)';
    abd_raw   = cell2mat(iAbd)';
    spo2_raw  = iSpO2';
    %therm_raw = cell2mat(iTherm)';

    % ECG removal (your function)
    emg_ecg_rmoved = preprocess_emg_complete_adv(emg_raw, fs_emg, false, false);

    % Filtering: bandpass 10-70Hz, notch 60Hz
    [b_bp,a_bp]=butter(4,[10 70]/(fs_emg/2),'bandpass');
    [b_notch,a_notch]=iirnotch(60/(fs_emg/2),(60/(fs_emg/2))/35);
    emg_filt = filtfilt(b_bp, a_bp, emg_ecg_rmoved);
    emg_filt = filtfilt(b_notch, a_notch, emg_filt);

    % Envelope
    env = abs(hilbert(emg_ecg_rmoved));
    env_lp = lowpass(env, 2, fs_emg);

    % DWT (0–1 Hz, nLevel=8 for fs_emg=256)
    wavename = 'db4';
    nLevel = 8;
    [C_emg, L_emg] = wavedec(emg_ecg_rmoved, nLevel, wavename);
    A_emg = wrcoef('a', C_emg, L_emg, wavename, nLevel);

    % Resample all signals to fs_emg
    [p_nasal,q_nasal] = rat(fs_emg/32);
    [p_flow, q_flow ] = rat(fs_emg/32);
    [p_thor, q_thor ] = rat(fs_emg/32);
    [p_abd,  q_abd  ] = rat(fs_emg/32);
    [p_spo2, q_spo2 ] = rat(fs_emg/1);
    [p_therm,q_therm] = rat(fs_emg/32);

    nasal_rs = resample(nasal_raw, p_nasal, q_nasal);
    flow_rs  = resample(flow_raw,  p_flow,  q_flow);
    thor_rs  = resample(thor_raw,  p_thor,  q_thor);
    abd_rs   = resample(abd_raw,   p_abd,   q_abd);
    spo2_rs  = resample(spo2_raw,  p_spo2,  q_spo2);
    therm_rs = resample(therm_raw, p_therm, q_therm);

    N = min([length(emg_raw), length(nasal_rs), length(flow_rs), length(thor_rs), length(abd_rs), length(spo2_rs), length(therm_rs)]);
    emg_raw      = emg_raw(1:N); emg_ecg_rmoved = emg_ecg_rmoved(1:N); emg_filt = emg_filt(1:N);
    env_lp       = env_lp(1:N);  A_emg = A_emg(1:N);
    nasal_rs     = nasal_rs(1:N); flow_rs = flow_rs(1:N); thor_rs = thor_rs(1:N);
    abd_rs       = abd_rs(1:N);  spo2_rs = spo2_rs(1:N); therm_rs = therm_rs(1:N);
    t_common     = (0:N-1)/fs_emg;

    %% --- Parse annotations & sleep stages from XML ---
    try
        xml_doc = xmlread(xml_path);
    catch
        warning('Could not read XML for subject %s', baseName);
        continue;
    end
    event_nodes = xml_doc.getElementsByTagName('ScoredEvent');
    annotations = struct('EventType', {}, 'EventConcept', {}, 'Start', {}, 'Duration', {});
    sleep_stages = struct('Start', {}, 'Duration', {}, 'Label', {});
    resp_events = struct('Start', {}, 'Duration', {}, 'Label', {});
    event_annots = struct('EventConcept', {}, 'Start', {}, 'Duration', {});

    for j = 0:event_nodes.getLength-1
        event = event_nodes.item(j);
        event_type = char(event.getElementsByTagName('EventType').item(0).getTextContent);
        event_concept = char(event.getElementsByTagName('EventConcept').item(0).getTextContent);
        start_time = str2double(event.getElementsByTagName('Start').item(0).getTextContent);
        duration = str2double(event.getElementsByTagName('Duration').item(0).getTextContent);
        annotations(end+1) = struct('EventType', event_type, 'EventConcept', event_concept, 'Start', start_time, 'Duration', duration);
        event_annots(end+1) = struct('EventConcept', event_concept, 'Start', start_time, 'Duration', duration);

        % Sleep stages
        if contains(event_type, 'Stages', 'IgnoreCase', true)
            label = NaN;
            if contains(event_concept, 'Wake|0', 'IgnoreCase', true)
                label = 0;
            elseif contains(event_concept, 'Stage 1 sleep|1', 'IgnoreCase', true)
                label = 1;
            elseif contains(event_concept, 'Stage 2 sleep|2', 'IgnoreCase', true)
                label = 2;
            elseif contains(event_concept, 'Stage 3 sleep|3', 'IgnoreCase', true)
                label = 3;
            elseif contains(event_concept, 'REM sleep|5', 'IgnoreCase', true)
                label = 5;
            end
            sleep_stages(end+1) = struct('Start', start_time, 'Duration', duration, 'Label', label);
        end
        % OSA events
        if contains(event_concept, 'Obstructive apnea|Obstructive Apnea', 'IgnoreCase', true)
            resp_events(end+1) = struct('Start', start_time, 'Duration', duration, 'Label', 'OSA');
        end
    end

    signal_length = length(emg_raw);
    fs = fs_emg;

    %% --- OSA Event Extraction (with multi-lead preOSA segments) ---
results_OSA = struct([]);
leadTimes = [0 30; 30 60; 60 90; 90 120; 120 150]; % seconds before event onset
leadLabels = {'preOSA_0_30', 'preOSA_30_60', 'preOSA_60_90', 'preOSA_90_120', 'preOSA_120_150'};

for k = 1:length(resp_events)
    event_start = resp_events(k).Start;
    event_dur   = resp_events(k).Duration;
    event_end   = event_start + event_dur;
    idx_start = max(1, round(event_start * fs) + 1);
    idx_end   = min(length(emg_raw), round(event_end * fs));
    seg_idx = idx_start:idx_end;
    t_seg = t_common(seg_idx);

    % OSA event block (as before)
    results_OSA(k).event_type   = 'OSA';
    results_OSA(k).event_annotation = 'OSA';
    results_OSA(k).event_start  = event_start;
    results_OSA(k).event_end    = event_end;
    results_OSA(k).duration     = (idx_end - idx_start + 1) / fs;
    results_OSA(k).time         = t_seg;
    results_OSA(k).emg_raw      = emg_raw(seg_idx);
    results_OSA(k).emg_ecg_rmoved  = emg_ecg_rmoved(seg_idx);
    results_OSA(k).emg_filt     = emg_filt(seg_idx);
    results_OSA(k).env_lp       = env_lp(seg_idx);
    results_OSA(k).emg_dwt_0_1Hz = A_emg(seg_idx);
    results_OSA(k).nasal        = nasal_rs(seg_idx);
    results_OSA(k).flow         = flow_rs(seg_idx);
    results_OSA(k).thor         = thor_rs(seg_idx);
    results_OSA(k).abd          = abd_rs(seg_idx);
    results_OSA(k).spo2         = spo2_rs(seg_idx);
    results_OSA(k).therm        = therm_rs(seg_idx);
    results_OSA(k).annotations  = get_overlapping_annotations(annotations, event_start, event_end);
    results_OSA(k).sleep_stages = get_overlapping_sleepstages(sleep_stages, event_start, event_end);
    results_OSA(k).majority_sleep_stage = majority_stage_interval(results_OSA(k).sleep_stages, event_start, event_end);

    % --- Multi-lead preOSA segments
    for l = 1:size(leadTimes,1)
        offset_sec = leadTimes(l,1);
        window_sec = leadTimes(l,2) - leadTimes(l,1);

        % Start and end time for this preOSA window
        win_end_time = event_start - offset_sec;      % e.g. event_start - 0s, -30s, -60s...
        win_start_time = win_end_time - window_sec;   % e.g. (event_start-30) to (event_start-0)

        % Indexing
        idx_win_start = max(1, round(win_start_time * fs) + 1);
        idx_win_end   = max(1, round(win_end_time * fs)); % avoid negatives at start of record

        if idx_win_start < idx_win_end && idx_win_start > 0 && idx_win_end <= length(emg_raw) && (win_start_time >= 0)
            seg_pre = idx_win_start:idx_win_end;
            results_OSA(k).(leadLabels{l}).event_annotation = leadLabels{l};
            results_OSA(k).(leadLabels{l}).time             = t_common(seg_pre);
            results_OSA(k).(leadLabels{l}).emg_raw          = emg_raw(seg_pre);
            results_OSA(k).(leadLabels{l}).emg_ecg_rmoved   = emg_ecg_rmoved(seg_pre);
            results_OSA(k).(leadLabels{l}).emg_filt         = emg_filt(seg_pre);
            results_OSA(k).(leadLabels{l}).env_lp           = env_lp(seg_pre);
            results_OSA(k).(leadLabels{l}).emg_dwt_0_1Hz    = A_emg(seg_pre);
            results_OSA(k).(leadLabels{l}).nasal            = nasal_rs(seg_pre);
            results_OSA(k).(leadLabels{l}).flow             = flow_rs(seg_pre);
            results_OSA(k).(leadLabels{l}).thor             = thor_rs(seg_pre);
            results_OSA(k).(leadLabels{l}).abd              = abd_rs(seg_pre);
            results_OSA(k).(leadLabels{l}).spo2             = spo2_rs(seg_pre);
            results_OSA(k).(leadLabels{l}).therm            = therm_rs(seg_pre);
            results_OSA(k).(leadLabels{l}).annotations      = get_overlapping_annotations(annotations, win_start_time, win_end_time);
            results_OSA(k).(leadLabels{l}).sleep_stages     = get_overlapping_sleepstages(sleep_stages, win_start_time, win_end_time);
            results_OSA(k).(leadLabels{l}).majority_sleep_stage = majority_stage_interval(results_OSA(k).(leadLabels{l}).sleep_stages, win_start_time, win_end_time);
        else
            % If not enough data (e.g., at beginning of record), skip
            results_OSA(k).(leadLabels{l}) = [];
        end
    end
end

    % results_OSA = struct([]);
    % for k = 1:length(resp_events)
    %     event_start = resp_events(k).Start;
    %     event_dur   = resp_events(k).Duration;
    %     event_end   = event_start + event_dur;
    %     idx_start = round(event_start * fs) + 1;
    %     idx_end   = round(event_end * fs);
    %     idx_pre_start = max(1, idx_start - fs*30);
    %     idx_pre_end = idx_start - 1;
    %     seg_idx = idx_start:idx_end;
    %     pre_idx = idx_pre_start:idx_pre_end;
    % 
    %     % -- OSA --
    %     results_OSA(k).event_type   = 'OSA';
    %     results_OSA(k).event_annotation = 'OSA';
    %     results_OSA(k).event_start  = event_start;
    %     results_OSA(k).event_end    = event_end;
    %     results_OSA(k).duration     = (idx_end - idx_start + 1) / fs;
    %     results_OSA(k).time         = t_common(seg_idx);
    % 
    %     results_OSA(k).emg_raw         = emg_raw(seg_idx);
    %     results_OSA(k).emg_ecg_rmoved  = emg_ecg_rmoved(seg_idx);
    %     results_OSA(k).emg_filt        = emg_filt(seg_idx);
    %     results_OSA(k).env_lp          = env_lp(seg_idx);
    %     results_OSA(k).emg_dwt_0_1Hz   = A_emg(seg_idx);
    % 
    %     results_OSA(k).nasal   = nasal_rs(seg_idx);
    %     results_OSA(k).flow    = flow_rs(seg_idx);
    %     results_OSA(k).thor    = thor_rs(seg_idx);
    %     results_OSA(k).abd     = abd_rs(seg_idx);
    %     results_OSA(k).spo2    = spo2_rs(seg_idx);
    %     results_OSA(k).therm   = therm_rs(seg_idx);
    % 
    %     results_OSA(k).annotations  = get_overlapping_annotations(annotations, event_start, event_end);
    %     results_OSA(k).sleep_stages = get_overlapping_sleepstages(sleep_stages, event_start, event_end);
    %     results_OSA(k).majority_sleep_stage = majority_stage_interval(results_OSA(k).sleep_stages, event_start, event_end);
    % 
    %     % -- Pre-OSA --
    %     results_OSA(k).pre_event.event_annotation = 'preOSA';
    %     results_OSA(k).pre_event.time         = t_common(pre_idx);
    %     results_OSA(k).pre_event.emg_raw      = emg_raw(pre_idx);
    %     results_OSA(k).pre_event.emg_ecg_rmoved = emg_ecg_rmoved(pre_idx);
    %     results_OSA(k).pre_event.emg_filt     = emg_filt(pre_idx);
    %     results_OSA(k).pre_event.env_lp       = env_lp(pre_idx);
    %     results_OSA(k).pre_event.emg_dwt_0_1Hz= A_emg(pre_idx);
    %     results_OSA(k).pre_event.nasal        = nasal_rs(pre_idx);
    %     results_OSA(k).pre_event.flow         = flow_rs(pre_idx);
    %     results_OSA(k).pre_event.thor         = thor_rs(pre_idx);
    %     results_OSA(k).pre_event.abd          = abd_rs(pre_idx);
    %     results_OSA(k).pre_event.spo2         = spo2_rs(pre_idx);
    %     results_OSA(k).pre_event.therm        = therm_rs(pre_idx);
    %     results_OSA(k).pre_event.annotations  = get_overlapping_annotations(annotations, (idx_pre_start-1)/fs, (idx_pre_end-1)/fs);
    %     results_OSA(k).pre_event.sleep_stages = get_overlapping_sleepstages(sleep_stages, (idx_pre_start-1)/fs, (idx_pre_end-1)/fs);
    %     results_OSA(k).pre_event.majority_sleep_stage = majority_stage_interval(results_OSA(k).pre_event.sleep_stages, (idx_pre_start-1)/fs, (idx_pre_end-1)/fs);
    % end

        %% --- Find gaps between events for Normal breathing (must have your own gap detection code here) ---
    % Let's assume you have gap_starts and gap_ends (in seconds) from a prior step
    % and event_annots + idx for context checks.
    % If you need a gap detection template, ask!

    % Example: gap_starts, gap_ends, idx should be provided
    % [gap_starts, gap_ends, idx] = your_gap_detection_function(event_annots, ...);
    % --- Find gaps between events (for Normal extraction) ---
    % Build sorted list of all respiratory event end/start times
   % sorted_starts and sorted_ends: vectors of event start/end times (seconds)

   % Step 1: Build event_annots list (real events only)
  % Step 1: Build event_annots list (real events only)
  signal_length = length(emg_raw);
  signal_duration = signal_length / fs;
  min_pre_duration = 30; % seconds for preNormal
  if isempty(results_OSA)
      warning('No OSA events for subject %s, skipping normal extraction.', baseName);
      continue
  end
  min_duration = min([results_OSA.duration]);
  event_annots = [];
  for i = 1:length(annotations)
      type = lower(annotations(i).EventType);
      if ~contains(type, 'stage') && annotations(i).Duration < 0.9*signal_duration
          event_annots = [event_annots, annotations(i)];
      end
  end

  % Step 2: Find gaps between events
  event_starts = [event_annots.Start];
  event_ends = [event_annots.Start] + [event_annots.Duration];
  [sorted_starts, idx] = sort(event_starts);
  sorted_ends = event_ends(idx);

  gap_starts = [];
  gap_ends = [];
  idx = []; % index of event *after* the gap (for context)

  % Initial gap before the first event
  if isempty(sorted_starts) || sorted_starts(1) > 0
      gap_starts(end+1) = 0;
      gap_ends(end+1) = sorted_starts(1);
      idx(end+1) = 1; % gap is before first event
  end

  % Gaps between events
  for i = 1:length(sorted_starts)-1
      if sorted_ends(i) < sorted_starts(i+1)
          gap_starts(end+1) = sorted_ends(i);
          gap_ends(end+1)   = sorted_starts(i+1);
          idx(end+1) = i+1; % gap is between event i and event i+1
      end
  end

  % Final gap after the last event
  if isempty(sorted_ends) || sorted_ends(end) < signal_duration
      gap_starts(end+1) = sorted_ends(end);
      gap_ends(end+1)   = signal_duration;
      idx(end+1) = length(sorted_starts)+1; % after last event
  end
    %% --- Normal (diverse-by-stage) Extraction (with all signals/versions) ---
    % Required: gap_starts, gap_ends (in seconds), idx, event_annots as above
    bad_types = {'hypopnea', 'obstructive apnea', 'central apnea', 'mixed apnea'};
    normals_needed = length(results_OSA); % match # of OSA events
    min_duration = 30; buffer_sec = 10; stride = min_duration; min_pre_duration = 30;

    stage_labels = {'Wake', 'N1', 'N2', 'N3', 'REM'};
    for s = 1:length(stage_labels)
        stage_buckets.(stage_labels{s}) = [];
    end

    for i = 1:length(gap_starts)
    % ---- Robust context assignment ----
    prev_type = '';
    if i > 1 && idx(i-1) > 0 && idx(i-1) <= length(event_annots)
        prev_type = lower(event_annots(idx(i-1)).EventConcept);
    end
    next_type = '';
    if idx(i) > 0 && idx(i) <= length(event_annots)
        next_type = lower(event_annots(idx(i)).EventConcept);
    end
    if any(contains(prev_type, bad_types, 'IgnoreCase', true)) && ...
            any(contains(next_type, bad_types, 'IgnoreCase', true))
        continue;
    end
        gap_len = gap_ends(i) - gap_starts(i);
        if gap_len >= min_duration + 2*buffer_sec
            t1 = gap_starts(i) + buffer_sec;
            t2_max = gap_ends(i) - buffer_sec - min_duration;
            while (t1 <= t2_max)
                t2 = t1 + min_duration;
                if t1 < min_pre_duration
                    t1 = t1 + stride; continue;
                end
                idx_start = max(1, round(t1 * fs) + 1);
                idx_end = min(signal_length, round(t2 * fs));
                idx_pre_start = max(1, idx_start - fs*min_pre_duration);
                idx_pre_end = idx_start - 1;
                seg_idx = idx_start:idx_end;
                pre_idx = idx_pre_start:idx_pre_end;

                normal_stages = get_overlapping_sleepstages(sleep_stages, t1, t2);
                normal_maj_stage = majority_stage_interval(normal_stages, t1, t2);
                stage_name = '';
                switch normal_maj_stage
                    case 0, stage_name = 'Wake';
                    case 1, stage_name = 'N1';
                    case 2, stage_name = 'N2';
                    case 3, stage_name = 'N3';
                    case 5, stage_name = 'REM';
                end
                if isfield(stage_buckets, stage_name) && ~isempty(stage_name)
                    this_struct.event_type = 'Normal';
                    this_struct.event_annotation = 'Normal';
                    this_struct.event_start = t1;
                    this_struct.event_end = t2;
                    this_struct.duration = (idx_end - idx_start + 1) / fs;
                    this_struct.time = t_common(seg_idx);
                    this_struct.emg_raw         = emg_raw(seg_idx);
                    this_struct.emg_ecg_rmoved  = emg_ecg_rmoved(seg_idx);
                    this_struct.emg_filt        = emg_filt(seg_idx);
                    this_struct.env_lp          = env_lp(seg_idx);
                    this_struct.emg_dwt_0_1Hz   = A_emg(seg_idx);
                    this_struct.nasal   = nasal_rs(seg_idx);
                    this_struct.flow    = flow_rs(seg_idx);
                    this_struct.thor    = thor_rs(seg_idx);
                    this_struct.abd     = abd_rs(seg_idx);
                    this_struct.spo2    = spo2_rs(seg_idx);
                    this_struct.therm   = therm_rs(seg_idx);
                    this_struct.annotations  = get_overlapping_annotations(annotations, t1, t2);
                    this_struct.sleep_stages = normal_stages;
                    this_struct.majority_sleep_stage = normal_maj_stage;
                    this_struct.pre_event.event_annotation = 'preNormal';
                    this_struct.pre_event.time         = t_common(pre_idx);
                    this_struct.pre_event.emg_raw      = emg_raw(pre_idx);
                    this_struct.pre_event.emg_ecg_rmoved = emg_ecg_rmoved(pre_idx);
                    this_struct.pre_event.emg_filt     = emg_filt(pre_idx);
                    this_struct.pre_event.env_lp       = env_lp(pre_idx);
                    this_struct.pre_event.emg_dwt_0_1Hz= A_emg(pre_idx);
                    this_struct.pre_event.nasal        = nasal_rs(pre_idx);
                    this_struct.pre_event.flow         = flow_rs(pre_idx);
                    this_struct.pre_event.thor         = thor_rs(pre_idx);
                    this_struct.pre_event.abd          = abd_rs(pre_idx);
                    this_struct.pre_event.spo2         = spo2_rs(pre_idx);
                    this_struct.pre_event.therm        = therm_rs(pre_idx);
                    this_struct.pre_event.annotations  = get_overlapping_annotations(annotations, (idx_pre_start-1)/fs, (idx_pre_end-1)/fs);
                    this_struct.pre_event.sleep_stages = get_overlapping_sleepstages(sleep_stages, (idx_pre_start-1)/fs, (idx_pre_end-1)/fs);
                    this_struct.pre_event.majority_sleep_stage = majority_stage_interval(this_struct.pre_event.sleep_stages, (idx_pre_start-1)/fs, (idx_pre_end-1)/fs);
                    stage_buckets.(stage_name) = [stage_buckets.(stage_name), this_struct];
                end
                t1 = t1 + stride;
            end
        end
    end

    % Interleave from buckets for diversity
    results_Normal = struct('event_type', {}, 'event_annotation', {}, 'event_start', {}, ...
        'event_end', {}, 'duration', {}, 'time', {}, 'emg_raw', {}, 'emg_ecg_rmoved', {}, 'emg_filt', {}, ...
        'env_lp', {}, 'emg_dwt_0_1Hz', {}, 'nasal', {}, 'flow', {}, 'thor', {}, 'abd', {}, ...
        'spo2', {}, 'therm', {}, 'annotations', {}, 'sleep_stages', {}, 'majority_sleep_stage', {}, 'pre_event', {});
    added = 0; idx2 = 1;
    while added < normals_needed
        found = false;
        for s = 1:length(stage_labels)
            stage = stage_labels{s};
            if idx2 <= length(stage_buckets.(stage))
                results_Normal(end+1) = stage_buckets.(stage)(idx2);
                added = added + 1;
                found = true;
                if added >= normals_needed
                    break
                end
            end
        end
        if ~found, break; end
        idx2 = idx2 + 1;
    end
    disp(['Total OSA: ', num2str(length(results_OSA)), '   Total Normal: ', num2str(length(results_Normal))]);

    %% --- Demographic & Save ---
    subject_id = str2double(regexp(baseName, '\d+$', 'match', 'once'));
    demog_row = demog_tbl(demog_tbl.mesaid == subject_id, :);
    if isempty(demog_row)
        warning('No demographic info found for subject %d', subject_id);
        demog_row = table(NaN, NaN, NaN, {''}, {''}, NaN, ...
            'VariableNames', {'mesaid','nsrr_age','nsrr_bmi','nsrr_sex','nsrr_race','nsrr_ahi_hp3r_aasm15'});
    end
    subject_results.mesaid = subject_id;
    subject_results.nsrr_age = demog_row.nsrr_age;
    subject_results.nsrr_bmi = demog_row.nsrr_bmi;
    subject_results.nsrr_sex = demog_row.nsrr_sex;
    subject_results.nsrr_race = demog_row.nsrr_race;
    subject_results.nsrr_ahi_hp3r_aasm15 = demog_row.nsrr_ahi_hp3r_aasm15;
    subject_results.OSA_events = results_OSA;
    subject_results.Normal_events = results_Normal;
    save(fullfile(output_dir, [baseName '_extracted_events.mat']), 'subject_results');
end

%%%PostprocesstoAddPosandDemographics_multiple_lead_time
clc; clear; close all;
output_dir = 'D:\Adil Research work\OneDrive_2025-03-05\Sleep Apnea Prediction\Sleep Apnea Prediction\Pilot Study\Respiratory_Event_relation_between_ChinandEMG\ProcessedData';
edf_dir    = 'D:\Adil Research work\mesa\edfs\'; % Your EDF folder
demog_file = 'D:\Adil Research work\OneDrive_2025-03-05\Sleep Apnea Prediction\Sleep Apnea Prediction\Sleep Apnea Prediction\datasets\mesa-sleep-dataset-0.7.0.csv'; % Your demographics CSV

mat_files = dir(fullfile(output_dir, '*_extracted_events.mat'));
demog_tbl = readtable(demog_file);

% Position code-to-label mapping
pos_labels = {'Right', 'Back', 'Left', 'Prone', 'Upright'};

for i = 1:length(mat_files)
    mat_path = fullfile(output_dir, mat_files(i).name);
    load(mat_path, 'subject_results'); % Loads struct
    
    tokens = regexp(mat_files(i).name, 'mesa-sleep-(\d+)_extracted_events.mat', 'tokens');
    if isempty(tokens)
        warning('Could not parse subject ID from file name: %s', mat_files(i).name);
        continue;
    end
    subject_id = str2double(tokens{1}{1});
    
    % Load EDF file for this subject
    edf_file = fullfile(edf_dir, sprintf('mesa-sleep-%04d.edf', subject_id));
    try
        [headers, data] = edfread(edf_file);
    catch
        warning('Could not read EDF for subject %d', subject_id);
        continue;
    end
    
    fields = fieldnames(headers);
    pos_names = {'pos', 'Pos', 'POS', 'Position', 'POSITION'};
    iPos = [];
    for ii = 1:length(pos_names)
        idx_pos = find(strcmpi(pos_names{ii}, fields), 1);
        if ~isempty(idx_pos)
            iPos = headers.(fields{idx_pos});
            break;
        end
    end
    if isempty(iPos)
        pos_raw = [];
        pos_rs = [];
        warning('No Pos channel for subject %d', subject_id);
    else
        pos_raw = cell2mat(iPos)';
        fs_pos = 32;
        fs_emg = 256;
        N_pos = length(pos_raw);
        duration_sec = N_pos / fs_pos;
        N_emg = round(duration_sec * fs_emg);
        t_pos = (0:N_pos-1) / fs_pos;
        t_emg = (0:N_emg-1) / fs_emg;
        pos_rs = interp1(t_pos, pos_raw, t_emg, 'nearest', 'extrap');
        % pos_rs is now at fs_emg, all values 0–4
    end

    % --- OSA events: Annotate main event and all preOSA windows ---
    for k = 1:length(subject_results.OSA_events)
        ev = subject_results.OSA_events(k);
        % Main event
        idx_start = max(1, round(ev.event_start * fs_emg) + 1);
        idx_end   = min(length(pos_rs), round(ev.event_end * fs_emg));
        if idx_end < idx_start
            pos_seg = [];
        else
            pos_seg = pos_rs(idx_start:idx_end);
        end
        if isempty(pos_seg) || all(isnan(pos_seg))
            maj_code = NaN; maj_label = '';
        else
            maj_code = mode(pos_seg(~isnan(pos_seg)));
            if isnan(maj_code) || maj_code < 0 || maj_code > 4
                maj_label = 'Unknown';
            else
                maj_label = pos_labels{maj_code + 1};
            end
        end
        subject_results.OSA_events(k).pos = pos_seg;
        subject_results.OSA_events(k).majority_pos = maj_code;
        subject_results.OSA_events(k).majority_pos_label = maj_label;

        % --- Annotate ALL preOSA lead time windows for this event ---
        event_fields = fieldnames(subject_results.OSA_events(k));
        for ff = 1:length(event_fields)
            fname = event_fields{ff};
            if startsWith(fname, 'preOSA') && isstruct(subject_results.OSA_events(k).(fname))
                this_preOSA = subject_results.OSA_events(k).(fname);
                if ~isfield(this_preOSA, 'time'), continue; end
                t_pre = this_preOSA.time;
                if isempty(t_pre)
                    pos_pre = [];
                else
                    idx_pre = max(1, round(t_pre(1)*fs_emg)+1):min(length(pos_rs), round(t_pre(end)*fs_emg));
                    if isempty(idx_pre) || idx_pre(1) > idx_pre(end)
                        pos_pre = [];
                    else
                        pos_pre = pos_rs(idx_pre);
                    end
                end
                if isempty(pos_pre) || all(isnan(pos_pre))
                    maj_code_pre = NaN; maj_label_pre = '';
                else
                    maj_code_pre = mode(pos_pre(~isnan(pos_pre)));
                    if isnan(maj_code_pre) || maj_code_pre < 0 || maj_code_pre > 4
                        maj_label_pre = 'Unknown';
                    else
                        maj_label_pre = pos_labels{maj_code_pre + 1};
                    end
                end
                subject_results.OSA_events(k).(fname).pos = pos_pre;
                subject_results.OSA_events(k).(fname).majority_pos = maj_code_pre;
                subject_results.OSA_events(k).(fname).majority_pos_label = maj_label_pre;
            end
        end
    end

    % --- Normal events (and preNormal): No changes needed! ---
    for k = 1:length(subject_results.Normal_events)
        ev = subject_results.Normal_events(k);
        idx_start = max(1, round(ev.event_start * fs_emg) + 1);
        idx_end   = min(length(pos_rs), round(ev.event_end * fs_emg));
        if idx_end < idx_start
            pos_seg = [];
        else
            pos_seg = pos_rs(idx_start:idx_end);
        end
        if isempty(pos_seg) || all(isnan(pos_seg))
            maj_code = NaN; maj_label = '';
        else
            maj_code = mode(pos_seg(~isnan(pos_seg)));
            if isnan(maj_code) || maj_code < 0 || maj_code > 4
                maj_label = 'Unknown';
            else
                maj_label = pos_labels{maj_code + 1};
            end
        end
        subject_results.Normal_events(k).pos = pos_seg;
        subject_results.Normal_events(k).majority_pos = maj_code;
        subject_results.Normal_events(k).majority_pos_label = maj_label;

        % preNormal
        t_pre_start = ev.event_start - 30;
        t_pre_end   = ev.event_start;
        idx_pre_start = max(1, round(t_pre_start * fs_emg) + 1);
        idx_pre_end   = min(length(pos_rs), round(t_pre_end * fs_emg));
        if idx_pre_end < idx_pre_start
            pos_pre = [];
        else
            pos_pre = pos_rs(idx_pre_start:idx_pre_end);
        end
        if isempty(pos_pre) || all(isnan(pos_pre))
            maj_code_pre = NaN; maj_label_pre = '';
        else
            maj_code_pre = mode(pos_pre(~isnan(pos_pre)));
            if isnan(maj_code_pre) || maj_code_pre < 0 || maj_code_pre > 4
                maj_label_pre = 'Unknown';
            else
                maj_label_pre = pos_labels{maj_code_pre + 1};
            end
        end
        subject_results.Normal_events(k).pre_event.pos = pos_pre;
        subject_results.Normal_events(k).pre_event.majority_pos = maj_code_pre;
        subject_results.Normal_events(k).pre_event.majority_pos_label = maj_label_pre;
    end

    % --- Add new demographics/clinical variables ---
    demog_row = demog_tbl(demog_tbl.mesaid == subject_id, :);
    if ~isempty(demog_row)
        subject_results.wrksched5   = demog_row.wrksched5;
        subject_results.smkstat5    = demog_row.smkstat5;
        subject_results.slp_eff5    = demog_row.slp_eff5;
        subject_results.types5      = demog_row.types5;
        subject_results.epslpscl5c  = demog_row.epslpscl5c;
        subject_results.insmnia5    = demog_row.insmnia5;
        subject_results.rstlesslgs5 = demog_row.rstlesslgs5;
        subject_results.slpapnea5   = demog_row.slpapnea5;
        subject_results.cpap5       = demog_row.cpap5;
        subject_results.dntaldv5    = demog_row.dntaldv5;
        subject_results.uvula5      = demog_row.uvula5;
        subject_results.site5c      = demog_row.site5c;
        subject_results.overall5    = demog_row.overall5;
        subject_results.quchin5     = demog_row.quchin5;
    else
        warning('No demographic info for subject %d', subject_id);
    end

    % --- Save updated struct ---
    save(mat_path, 'subject_results');
    fprintf('Updated and saved %s\n', mat_path);
end

%% Helper Function
function overlap = get_overlapping_annotations(annotations, t_start, t_end)
    overlap = [];
    for i = 1:length(annotations)
        a_start = annotations(i).Start;
        a_end = a_start + annotations(i).Duration;
        if max(a_start, t_start) < min(a_end, t_end)
            overlap = [overlap; annotations(i)];
        end
    end
end

function stages = get_overlapping_sleepstages(sleep_stages, t_start, t_end)
    stages = [];
    for i = 1:length(sleep_stages)
        s_start = sleep_stages(i).Start;
        s_end = s_start + sleep_stages(i).Duration;
        if max(s_start, t_start) < min(s_end, t_end)
            stages = [stages; sleep_stages(i)];
        end
    end
end

function maj_stage = majority_stage_interval(sleep_stages, t_start, t_end)
    % Returns the label with the longest coverage within [t_start, t_end]
    if isempty(sleep_stages)
        maj_stage = NaN;
        return;
    end
    unique_labels = unique([sleep_stages.Label]);
    max_cov = 0;
    maj_stage = NaN;
    for l = 1:length(unique_labels)
        label = unique_labels(l);
        cov = 0;
        for i = 1:length(sleep_stages)
            if sleep_stages(i).Label == label
                % Calculate overlap in the current interval
                s1 = max(sleep_stages(i).Start, t_start);
                s2 = min(sleep_stages(i).Start + sleep_stages(i).Duration, t_end);
                overlap = max(0, s2 - s1);
                cov = cov + overlap;
            end
        end
        if cov > max_cov
            max_cov = cov;
            maj_stage = label;
        end
    end
end