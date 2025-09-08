%%Part 1: Subject Selection from CSV Data
csv_file = 'D:/Adil Research work/.../mesa-sleep-dataset-0.7.0.csv'; 
mesa_data = readtable(csv_file);

% Inclusion criteria
strict_idx = (mesa_data.overall5 == 7) & (mesa_data.quchin5 == 5) & ...
             (mesa_data.status_psg5 == 1) & ~isnan(mesa_data.oahi4pa5);
strict_data = mesa_data(strict_idx, :);

% Pilot group (10 subjects)
selected10 = [338; 427; 2551; 2651; 3013; 3168; 3407; 3717; 5369; 6065];
selected10_data = strict_data(ismember(strict_data.mesaid, selected10), :);

% Main cohort (remaining subjects)
additional_pool = strict_data(~ismember(strict_data.mesaid, selected10), :);
final_subjects = [selected10_data; additional_pool];
final_ids = final_subjects.mesaid;

%% Part 2: File Extraction
edf_dir = 'D:\Adil Research work\mesa\edfs\'; 
xml_dir = 'D:\Adil Research work\mesa\annotations-events-nsrr\'; 

% Copy EDF/XML files to destination folders
for i = 1:length(final_ids)
    subj_id = final_ids(i);
    edf_filename = sprintf('mesa-sleep-%04d.edf', subj_id);
    xml_filename = sprintf('mesa-sleep-%04d-nsrr.xml', subj_id);
    
    % Copy EDF
    edf_file = fullfile(edf_dir, edf_filename);
    if exist(edf_file, 'file')
        copyfile(edf_file, dest_edf_dir);
    end
    
    % Copy XML
    xml_file = fullfile(xml_dir, xml_filename);
    if exist(xml_file, 'file')
        copyfile(xml_file, dest_xml_dir);
    end
end