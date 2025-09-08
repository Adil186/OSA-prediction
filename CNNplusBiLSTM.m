clear all;clc; close all;

% Define file path
matFilePath = 'all_preOSA_preNormal_DLdata_EMG_ECG_RMOVED_for_all_subjects.mat';

%% 1. Load Data
fprintf('Loading data from %s...\n', matFilePath);
try
    % Load the.mat file. This will load variables 'X' and 'y' into the workspace.
    load(matFilePath);
    fprintf('Data loaded successfully.\n');
    
    % Ensure X is a cell array of sequences and y is a numeric array
    % X should be 14139x1 cell, each cell containing a 7680x1 double array
    % y should be 14139x1 double array
    
    % Verify initial data dimensions and types
    if ~iscell(X) || size(X, 1) ~= 89038 || size(X, 2) ~= 1
        error('X variable is not a 14139x1 cell array.');
    end
    if ~isnumeric(y) || size(y, 1) ~= 89038 || size(y, 2) ~= 1
        error('y variable is not a 14139x1 numeric array.');
    end
    
    % Convert labels to categorical for deep learning
    Y_categorical = categorical(y, [0 1], {'preNormal', 'preOSA'});
    
catch ME
    fprintf('Error loading or verifying data: %s\n', ME.message);
    return; % Exit if data loading fails
end

%% 2. Data Preprocessing (Splitting and Normalization)
fprintf('Preprocessing data (splitting and normalization)...\n');
% Assume you have X and Y_categorical already loaded
% X: cell array (N x 1), Y_categorical: categorical (N x 1)

N = numel(Y_categorical);
assert(numel(X) == N, 'X and y length mismatch.');
trainRatio = 0.7;
valRatio   = 0.1;
testRatio  = 0.2;

rng(42); % For reproducibility

idxTrain = [];
idxVal = [];
idxTest = [];

classLabels = categories(Y_categorical);

for iClass = 1:numel(classLabels)
    className = classLabels{iClass};
    idxThisClass = find(Y_categorical == className);
    nThis = numel(idxThisClass);

    nTrain = round(trainRatio * nThis);
    nVal   = round(valRatio * nThis);
    nTest  = nThis - nTrain - nVal; % Whatever remains

    idxShuffled = idxThisClass(randperm(nThis));
    idxTrain = [idxTrain; idxShuffled(1:nTrain)];
    idxVal   = [idxVal;   idxShuffled(nTrain+1:nTrain+nVal)];
    idxTest  = [idxTest;  idxShuffled(nTrain+nVal+1:end)];
end

% Shuffle all splits
idxTrain = idxTrain(randperm(numel(idxTrain)));
idxVal   = idxVal(randperm(numel(idxVal)));
idxTest  = idxTest(randperm(numel(idxTest)));

% Make your splits
XTrain = X(idxTrain);      YTrain = Y_categorical(idxTrain);
XVal   = X(idxVal);        YVal   = Y_categorical(idxVal);
XTest  = X(idxTest);       YTest  = Y_categorical(idxTest);

fprintf('Stratified split: Train (%d), Val (%d), Test (%d), Total (%d)\n', ...
    numel(XTrain), numel(XVal), numel(XTest), numel(XTrain)+numel(XVal)+numel(XTest));

% Normalize data (Z-score normalization) [1, 2, 3, 4, 5]
% Calculate mean and std deviation ONLY from the training data [1, 2, 3, 4]
% The input is a cell array of sequences, so we need to concatenate for mean/std
allTrainData = cell2mat(XTrain);
mu = mean(allTrainData, 'all');
sigma = std(allTrainData, 0, 'all');

% Apply normalization to all datasets [2, 3, 5]
XTrain_norm = cellfun(@(x) (x - mu)./ sigma, XTrain, 'UniformOutput', false);
XVal_norm = cellfun(@(x) (x - mu)./ sigma, XVal, 'UniformOutput', false);
XTest_norm = cellfun(@(x) (x - mu)./ sigma, XTest, 'UniformOutput', false);

fprintf('Data normalized using training set statistics.\n');

%% Optional: Data Augmentation (Targeted for 'preNormal' flatlines)
% Based on your observation: "even though there is no event but the presence of airflow signal
% in 0-1 Hz range is close to the baseline. But in case of actual airflow signal this is not the case.
% So chin EMG in this case resemble like airflow signal during the OA event."
% This suggests some 'preNormal' samples might look like 'preOSA' precursors.
% We can augment 'preNormal' samples by adding subtle noise to help the model differentiate. [6, 4, 7, 8, 9]

% Find indices of 'preNormal' samples in the training set
idxPreNormalTrain = find(YTrain == 'preNormal');
numPreNormalToAugment = round(0.5 * numel(idxPreNormalTrain)); % Augment 50% of preNormal samples

% Select a subset of preNormal samples for augmentation
rng('shuffle'); % Use a different seed for augmentation randomness
selectedIdxForAugmentation = idxPreNormalTrain(randperm(numel(idxPreNormalTrain), numPreNormalToAugment));

XAugmented = cell(numPreNormalToAugment, 1);
YAugmented = categorical(zeros(numPreNormalToAugment, 1), [0 1], {'preNormal', 'preOSA'});

% Define noise level (e.g., 0.5% of the signal's standard deviation)
% This noise should be subtle enough not to change the fundamental pattern,
% but to introduce variability for robustness.
noise_std_dev = 0.005 * sigma; 

for i = 1:numPreNormalToAugment
    original_signal = XTrain_norm{selectedIdxForAugmentation(i)};
    % Add zero-mean white Gaussian noise [9]
    noisy_signal = original_signal + noise_std_dev * randn(size(original_signal));
    XAugmented{i} = noisy_signal;
    YAugmented(i) = YTrain(selectedIdxForAugmentation(i)); % Keep original label
end

% Combine original and augmented training data
XTrain_final =[XTrain_norm;XAugmented];
YTrain_final =[YTrain;YAugmented];

fprintf('Data augmented with noise for %d preNormal samples. New training size: %d\n',...
    numel(XAugmented), numel(XTrain_final));

%% 3. Define Network Architecture (Hybrid 1D CNN-BiLSTM)
fprintf('Defining 1D CNN-BiLSTM network architecture...\n');

% Input size: 7680 time steps, 1 channel (for 0-1Hz DWT signal) [10, 11, 12, 13, 14]
inputSize = [7680 1]; 
numClasses = numel(categories(Y_categorical)); % 2 classes: preNormal, preOSA

% Architecture inspired by [15, 16] (4 CNN layers, 3 LSTM layers, dropout)
layers = [
    % Input Layer [13, 14]
    sequenceInputLayer(inputSize, 'Name', 'input') 
    
    % 1D CNN Block 1 [17, 13, 18]
    convolution1dLayer(5, 32, 'Padding', 'same', 'Name', 'conv1d_1') % Filter size 5, 32 filters [16]
    batchNormalizationLayer('Name', 'bn_1') % Stabilizes training [19, 17, 20, 6, 21, 4]
    reluLayer('Name', 'relu_1') % Activation function [19, 16, 17]
    maxPooling1dLayer(2, 'Stride', 2, 'Name', 'maxpool_1') % Reduce dimensionality [16]
    
    % 1D CNN Block 2
    convolution1dLayer(5, 64, 'Padding', 'same', 'Name', 'conv1d_2') % 64 filters
    batchNormalizationLayer('Name', 'bn_2')
    reluLayer('Name', 'relu_2')
    maxPooling1dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
    
    % 1D CNN Block 3
    convolution1dLayer(5, 128, 'Padding', 'same', 'Name', 'conv1d_3') % 128 filters
    batchNormalizationLayer('Name', 'bn_3')
    reluLayer('Name', 'relu_3')
    maxPooling1dLayer(2, 'Stride', 2, 'Name', 'maxpool_3')
    
    % 1D CNN Block 4
    convolution1dLayer(5, 256, 'Padding', 'same', 'Name', 'conv1d_4') % 256 filters
    batchNormalizationLayer('Name', 'bn_4')
    reluLayer('Name', 'relu_4')
    maxPooling1dLayer(2, 'Stride', 2, 'Name', 'maxpool_4')
    
    % Dropout layer after CNN blocks [19, 16, 20, 6, 21, 4]
    dropoutLayer(0.5, 'Name', 'dropout_cnn') % Prevents overfitting [19, 16, 20, 6, 21, 4]
    
    % BiLSTM Layer 1
    bilstmLayer(128, 'OutputMode', 'sequence', 'Name', 'bilstm_1') % 128 hidden units, output sequence
    
    % BiLSTM Layer 2
    bilstmLayer(128, 'OutputMode', 'sequence', 'Name', 'bilstm_2')
    
    % BiLSTM Layer 3 (output last element for classification) [16]
    bilstmLayer(128, 'OutputMode', 'last', 'Name', 'bilstm_3') % Output only the last time step [16]
    
    % Dropout layer after BiLSTM blocks [19, 16, 20, 6, 21, 4]
    dropoutLayer(0.4, 'Name', 'dropout_lstm') % Prevents overfitting [19, 16, 20, 6, 21, 4]
    
    % Fully connected layer and output layers [16]
    fullyConnectedLayer(numClasses, 'Name', 'fc') % Maps features to class scores [16]
    softmaxLayer('Name', 'softmax') % Converts scores to probabilities [16]
    % REMOVED: classificationLayer('Name', 'output') % This layer is implicitly added by trainnet [21];
    ];
% Analyze network to check for errors
analyzeNetwork(layers);
fprintf('Network architecture defined and analyzed.\n');

%% 4. Specify Training Options
fprintf('Specifying training options...\n');

% --- IMPORTANT CHANGE: REMOVING CLASS WEIGHTING FOR BALANCED DATA ---
% As per your clarification, classes are almost equal.
% Therefore, inverse-frequency weighting is not needed and can introduce bias.
% We set weights to [1 1] to ensure equal importance for both classes.
weightsForLoss = [1 1]; % Equal weights for 'preNormal' (0) and 'preOSA' (1)

% Define training options
options = trainingOptions('adam',... % Adam optimizer recommended for time series [22, 23]
    'MaxEpochs', 5,... % Number of training epochs (can be tuned)
    'MiniBatchSize', 128,... % Mini-batch size (can be tuned)
    'InitialLearnRate', 0.0001,... % Initial learning rate (can be tuned)
    'GradientThreshold', 1,... % Gradient clipping to prevent exploding gradients
    'Shuffle', 'every-epoch',... % Shuffle data every epoch [2, 24, 5]
    'ValidationData', {XVal_norm, YVal},... % Validation data for monitoring [25, 26, 27, 28]
    'ValidationFrequency', 10,... % Validate every 10 iterations [27, 28]
    'ValidationPatience', 50,... % Early stopping: stop if validation metric doesn't improve for 10 validations
    'Plots', 'training-progress',... % Display training progress plot [13, 2, 21, 25, 24, 5, 27, 28]
    'Verbose', false,... % Suppress verbose output to command window [13, 2, 21, 25, 24, 5, 27, 28]
    'Metrics', {'accuracy', 'precision', 'recall', 'fscore'},... % Monitor key metrics
    'OutputNetwork', 'best-validation'); % Save network with best validation performance [28]

% Define loss function (crossentropy without class weighting, or with [1 1] weights)
lossFcn = @(Y,T) crossentropy(Y, T, 'Weights', weightsForLoss, 'WeightsFormat', 'C'); % 'C' for channel-wise weights

fprintf('Training options specified, class weights set to equal importance.\n');

%% 5. Train Network
fprintf('Training network...\n');

% Train the network [13, 2, 21, 25, 24, 5, 27, 28]
[net, info] = trainnet(XTrain_final, YTrain_final, layers, lossFcn, options);

fprintf('Network training complete.\n');
%% Validation
fprintf('Evaluating network performance on validation set...\n');

% Predict on validation set
YPredVal_scores = minibatchpredict(net, XVal_norm);
YPredVal = scores2label(YPredVal_scores, categories(Y_categorical)); % Or categories(YVal)

% Overall accuracy
accuracyVal = mean(YPredVal == YVal);
fprintf('Validation Accuracy: %.2f%%\n', accuracyVal * 100);

% Confusion Matrix
figure;
cmVal = confusionchart(YVal, YPredVal, 'Title', 'Validation Set Confusion Matrix');
cmVal.ColumnSummary = 'column-normalized'; % Precision (PPV)
cmVal.RowSummary = 'row-normalized';       % Recall (TPR)

% Precision, Recall, F1-score
truePosVal = sum(YVal == 'preOSA' & YPredVal == 'preOSA');
falsePosVal = sum(YVal == 'preNormal' & YPredVal == 'preOSA');
falseNegVal = sum(YVal == 'preOSA' & YPredVal == 'preNormal');

precisionVal = truePosVal / (truePosVal + falsePosVal + eps); % Avoid division by zero
recallVal    = truePosVal / (truePosVal + falseNegVal + eps);
f1ScoreVal   = 2 * (precisionVal * recallVal) / (precisionVal + recallVal + eps);

fprintf('Validation Precision (preOSA): %.4f\n', precisionVal);
fprintf('Validation Recall (preOSA): %.4f\n', recallVal);
fprintf('Validation F1-Score (preOSA): %.4f\n', f1ScoreVal);
%% 6. Evaluate Network Performance
fprintf('Evaluating network performance on test set...\n');

% Predict on the test set
YPred_scores = minibatchpredict(net, XTest_norm); % Get raw scores [13, 29, 21, 25, 24]
YPred = scores2label(YPred_scores, categories(Y_categorical)); % Convert scores to labels [13, 29, 21, 25, 24]

% Overall accuracy
accuracy = mean(YPred == YTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Confusion Matrix
figure;
cm = confusionchart(YTest, YPred, 'Title', 'Test Set Confusion Matrix');
cm.ColumnSummary = 'column-normalized'; % Display precision (PPV)
cm.RowSummary = 'row-normalized'; % Display recall (TPR)
fprintf('Confusion Matrix displayed.\n');

% Calculate additional metrics (Precision, Recall, F1-score, AUC)
% For binary classification, we focus on the 'preOSA' class (positive class)
truePos = sum(YTest == 'preOSA' & YPred == 'preOSA');
falsePos = sum(YTest == 'preNormal' & YPred == 'preOSA');
falseNeg = sum(YTest == 'preOSA' & YPred == 'preNormal');
trueNeg = sum(YTest == 'preNormal' & YPred == 'preNormal');

precision = truePos / (truePos + falsePos);
recall = truePos / (truePos + falseNeg);
f1Score = 2 * (precision * recall) / (precision + recall);

fprintf('Precision (preOSA): %.4f\n', precision);
fprintf('Recall (preOSA): %.4f\n', recall);
fprintf('F1-Score (preOSA): %.4f\n', f1Score);
