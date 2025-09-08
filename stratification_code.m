clc; clear; close all;
load('Test_Features_EMG_ECG_Removed_feature(from RAW)_removed_Inf_from_THD_features.mat', 'testTable');
T = testTable;
outdir = 'subgroup_tables';
if ~exist(outdir, 'dir'), mkdir(outdir); end
%% --- 1. Gender ---
groups = {'male', 'female'};
for i = 1:length(groups)
    idx = strcmpi(T.nsrr_sex, groups{i});
    subT = T(idx,:);
    save(fullfile(outdir, ['testTable_Gender_' groups{i} '.mat']), 'subT');
    writetable(subT, fullfile(outdir, ['testTable_Gender_' groups{i} '.csv']));
end

%% --- 2. Age ---
age = T.nsrr_age;
isMiddleAge = age >= 40 & age < 60;
isElderly   = age >= 60;
isYoung     = age < 40;
ageCats = {isYoung, 'Young'; isMiddleAge, 'MiddleAge'; isElderly, 'Elderly'};
for i = 1:size(ageCats,1)
    subT = T(ageCats{i,1}, :);
    save(fullfile(outdir, ['testTable_Age_' ageCats{i,2} '.mat']), 'subT');
    writetable(subT, fullfile(outdir, ['testTable_Age_' ageCats{i,2} '.csv']));
end

%% --- 3. Race ---
raceList = unique(T.nsrr_race);
for i = 1:length(raceList)
    idx = strcmpi(T.nsrr_race, raceList{i});
    safeName = regexprep(raceList{i}, '[^a-zA-Z0-9]', '');
    subT = T(idx, :);
    save(fullfile(outdir, ['testTable_Race_' safeName '.mat']), 'subT');
    writetable(subT, fullfile(outdir, ['testTable_Race_' safeName '.csv']));
end

%% --- 4. BMI Group ---
bmi = T.nsrr_bmi;
bmiCats = {bmi < 18.5,         'Underweight'; ...
           bmi >= 18.5 & bmi < 25,   'Normal'; ...
           bmi >= 25 & bmi < 30,     'Overweight'; ...
           bmi >= 30 & bmi < 35,     'Obesity1'; ...
           bmi >= 35 & bmi < 40,     'Obesity2'; ...
           bmi >= 40,                'Obesity3'};
for i = 1:size(bmiCats,1)
    subT = T(bmiCats{i,1}, :);
    save(fullfile(outdir, ['testTable_BMI_' bmiCats{i,2} '.mat']), 'subT');
    writetable(subT, fullfile(outdir, ['testTable_BMI_' bmiCats{i,2} '.csv']));
end

%% --- 5. Sleep Stage ---
stage = T.MajoritySleepStage;
sleepCats = {(stage==1 | stage==2), 'Light'; ...
             (stage==3), 'Deep'; ...
             (stage==5), 'REM'; ...
             (stage==0), 'Awake'};
for i = 1:size(sleepCats,1)
    subT = T(sleepCats{i,1}, :);
    save(fullfile(outdir, ['testTable_SleepStage_' sleepCats{i,2} '.mat']), 'subT');
    writetable(subT, fullfile(outdir, ['testTable_SleepStage_' sleepCats{i,2} '.csv']));
end

%% --- 6. Body Position ---
posLabels = unique(T.MajorityPosLabel);
for i = 1:length(posLabels)
    idx = strcmpi(T.MajorityPosLabel, posLabels{i});
    safeName = regexprep(posLabels{i}, '[^a-zA-Z0-9]', '');
    subT = T(idx, :);
    save(fullfile(outdir, ['testTable_Position_' safeName '.mat']), 'subT');
    writetable(subT, fullfile(outdir, ['testTable_Position_' safeName '.csv']));
end

%% --- 7. AHI Group ---
ahi = T.nsrr_ahi_hp3r_aasm15;
ahiCats = {ahi < 5,                  'Normal'; ...
           ahi >= 5 & ahi < 15,      'Mild'; ...
           ahi >= 15 & ahi < 30,     'Moderate'; ...
           ahi >= 30,                'Severe'};
for i = 1:size(ahiCats,1)
    subT = T(ahiCats{i,1}, :);
    save(fullfile(outdir, ['testTable_AHI_' ahiCats{i,2} '.mat']), 'subT');
    writetable(subT, fullfile(outdir, ['testTable_AHI_' ahiCats{i,2} '.csv']));
end

%% --- 8. Sleep Efficiency Group ---
eff = T.slp_eff5;
effCats = {eff > 90,                'Excellent'; ...
           eff > 85 & eff <= 90,    'Good'; ...
           eff > 80 & eff <= 85,    'Fair'; ...
           eff <= 80,               'Poor'};
for i = 1:size(effCats,1)
    subT = T(effCats{i,1}, :);
    save(fullfile(outdir, ['testTable_SleepEff_' effCats{i,2} '.mat']), 'subT');
    writetable(subT, fullfile(outdir, ['testTable_SleepEff_' effCats{i,2} '.csv']));
end

%% --- 9. Epworth Sleepiness Scale (epslpscl5c) ---
ess = T.epslpscl5c;
essCats = {ess >= 0 & ess <= 5,         'Low'; ...
           ess >= 6 & ess <= 10,        'High'; ...
           ess >= 11 & ess <= 12,       'MildExcess'; ...
           ess >= 13 & ess <= 15,       'ModerateExcess'; ...
           ess >= 16 & ess <= 24,       'SevereExcess'};
for i = 1:size(essCats,1)
    subT = T(essCats{i,1}, :);
    save(fullfile(outdir, ['testTable_ESS_' essCats{i,2} '.mat']), 'subT');
    writetable(subT, fullfile(outdir, ['testTable_ESS_' essCats{i,2} '.csv']));
end

%% --- 10. Work Schedule ---
workSched = unique(T.wrksched5);
for i = 1:length(workSched)
    idx = T.wrksched5 == workSched(i);
    subT = T(idx, :);
    save(fullfile(outdir, ['testTable_WorkSched_' num2str(workSched(i)) '.mat']), 'subT');
    writetable(subT, fullfile(outdir, ['testTable_WorkSched_' num2str(workSched(i)) '.csv']));
end

%% --- 11. Smoking Status (smkstat5) ---
smokeStat = unique(T.smkstat5);
for i = 1:length(smokeStat)
    idx = T.smkstat5 == smokeStat(i);
    subT = T(idx, :);
    save(fullfile(outdir, ['testTable_SmokeStat_' num2str(smokeStat(i)) '.mat']), 'subT');
    writetable(subT, fullfile(outdir, ['testTable_SmokeStat_' num2str(smokeStat(i)) '.csv']));
end

%% --- 12. Type of Person (types5, chronotype) ---
typePerson = unique(T.types5);
for i = 1:length(typePerson)
    idx = T.types5 == typePerson(i);
    subT = T(idx, :);
    save(fullfile(outdir, ['testTable_Chronotype_' num2str(typePerson(i)) '.mat']), 'subT');
    writetable(subT, fullfile(outdir, ['testTable_Chronotype_' num2str(typePerson(i)) '.csv']));
end

%% --- 13. Doctor-diagnosed SLEEP APNEA (slpapnea5), INSOMNIA (insmnia5), RESTLESS LEGS (rstlesslgs5) ---
for varName = {'slpapnea5','insmnia5','rstlesslgs5','cpap5','dntaldv5','uvula5'}
    vals = unique(T.(varName{1}));
    for i = 1:length(vals)
        idx = T.(varName{1}) == vals(i);
        subT = T(idx, :);
        save(fullfile(outdir, ['testTable_' varName{1} '_' num2str(vals(i)) '.mat']), 'subT');
        writetable(subT, fullfile(outdir, ['testTable_' varName{1} '_' num2str(vals(i)) '.csv']));
    end
end
%% --- 14. PSG Overall Study Quality (overall5) ---
% Codes: 2=FAILED, 3=FAIR, 4=GOOD, 5=VERY GOOD, 6=EXCELLENT, 7=OUTSTANDING
overallCodes = [2 3 4 5 6 7];
overallNames = {'Failed','Fair','Good','VeryGood','Excellent','Outstanding'};
for i = 1:length(overallCodes)
    idx = T.overall5 == overallCodes(i);
    subT = T(idx, :);
    if ~isempty(subT)
        save(fullfile(outdir, ['testTable_OverallQuality_' overallNames{i} '.mat']), 'subT');
        writetable(subT, fullfile(outdir, ['testTable_OverallQuality_' overallNames{i} '.csv']));
    end
end

%% --- 15. Chin Signal Quality (quchin5) ---
% Codes: 1 = <25%, 2 = 25-49%, 3 = 50-74%, 4 = 75-94%, 5 = >=95%
chinCodes = 1:5;
chinNames = {'LT25pct','25to49pct','50to74pct','75to94pct','GE95pct'};
for i = 1:length(chinCodes)
    idx = T.quchin5 == chinCodes(i);
    subT = T(idx, :);
    if ~isempty(subT)
        save(fullfile(outdir, ['testTable_ChinQuality_' chinNames{i} '.mat']), 'subT');
        writetable(subT, fullfile(outdir, ['testTable_ChinQuality_' chinNames{i} '.csv']));
    end
end

%% --- 16. Recording Site (site5c) ---
% Codes: 3=WFU, 4=COL, 5=JHU, 6=UMN, 7=NWU, 8=UCLA
siteCodes = [3 4 5 6 7 8];
siteNames = {'WFU','COL','JHU','UMN','NWU','UCLA'};
for i = 1:length(siteCodes)
    idx = T.site5c == siteCodes(i);
    subT = T(idx, :);
    if ~isempty(subT)
        save(fullfile(outdir, ['testTable_Site_' siteNames{i} '.mat']), 'subT');
        writetable(subT, fullfile(outdir, ['testTable_Site_' siteNames{i} '.csv']));
    end
end

disp('--- All group tables created in subfolder (including new stratifiers) ---');