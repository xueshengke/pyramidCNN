% new code for CNN  
clear all; close all; clc;
addpath(genpath('DeepLearnToolbox'));
%% load data from jpg or file system
width=128;
height=128;
imageSize = 128;
%% load dataset
[trainData, trainLabel, testData, testLabel] = cnnGenerateData();
%% if data have exist as mat in file system, just load
% load dcm/trainData;
% load dcm/trainLabel;
% end

%% batch mean to zero
 trainData = trainData-repmat(mean(trainData,2),1,size(trainData,2));
 testData = testData-repmat(mean(testData,2),1,size(testData,2));

trainData = reshape(trainData, width, height, size(trainData, 2));
testData = reshape(testData, width, height, size(testData, 2));
trainNum = size(trainData, 3) ;
testNum = size(testData, 3) ;

fprintf('prepare trainData %d * %d * %d \n', size(trainData, 1), size(trainData, 2), size(trainData, 3));
fprintf('prepare trainLabel %d * %d \n', size(trainLabel, 1), size(trainLabel, 2));
fprintf('prepare testData %d * %d * %d \n', size(testData, 1), size(testData, 2), size(testData, 3));
fprintf('prepare testLabel %d * %d \n', size(testLabel, 1), size(testLabel, 2));

%% create a 3c-2s first pyramid convolutional neural network 
fprintf('construct pyramid cnn ... \n');
% rand('state',0);
pcnnFirst.inputmaps = 1;         % gray image
pcnnFirst.classNumber = size(trainLabel, 1);
pcnnFirst.layers = {
    struct('type', 'i')                                     %input layer
    struct('type', 'c', 'outputmaps', 3, 'kernelsize', 5)   %convolution layer
    struct('type', 's', 'scale', 2)                         %subsampling layer
};
%% create a 6c-2s second pyramid convolutional neural network 
fprintf('construct pyramid cnn ... \n');
% rand('state',0);
pcnnSecond.inputmaps = pcnnFirst.layers{2}.outputmaps;         % gray image
pcnnSecond.classNumber = size(trainLabel, 1);
pcnnSecond.layers = {
    struct('type', 'i')                                     %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 3)   %convolution layer
    struct('type', 's', 'scale', 2)                         %subsampling layer
};
%% create a 5c-2s second pyramid convolutional neural network 
fprintf('construct pyramid cnn ... \n');
% rand('state',0);
pcnnThird.inputmaps = pcnnSecond.layers{2}.outputmaps;         % gray image
pcnnThird.classNumber = size(trainLabel, 1);
pcnnThird.layers = {
    struct('type', 'i')                                     %input layer
    struct('type', 'c', 'outputmaps', 5, 'kernelsize', 5)   %convolution layer
    struct('type', 's', 'scale', 2)                         %subsampling layer
};
%% create a 3c-2s-6c-2s-5c-2s pyramid convolutional neural network 
fprintf('construct pyramid cnn ... \n');
% rand('state',0);
pcnn.inputmaps = 1;         % gray image
pcnn.classNum = size(trainLabel, 1);
pcnn.layers = {
    struct('type', 'i')                                     % input layer
    struct('type', 'c', 'outputmaps', 3, 'kernelsize', 5)   % convolution layer
    struct('type', 's', 'scale', 2)                         % subsampling layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 3)   % convolution layer
    struct('type', 's', 'scale', 2)                         % subsampling layer
    struct('type', 'c', 'outputmaps', 5, 'kernelsize', 5)   % convolution layer
    struct('type', 's', 'scale', 2)                         % subsampling layer
};

opts.alpha = 1 ;
opts.batchsize = 57 ;      % needs to change according to trainNum 57 * 9 = 513
opts.numepochs = 150;      % long time  seconds per poches
opts.lowThreshold = 1e-6 ;

%% extract batches from train data
fprintf('extract batches from train data ... \n');
batchSize = zeros(3,1);
batchSize(3) = imageSize;
batchSize(2) = 60;
batchSize(1) = 30;
% batchSize(2) = (batchSize(3) - pcnnFirst.layers{2}.kernelsize + 1) / pcnnFirst.layers{3}.scale;
% batchSize(1) = (batchSize(2) - pcnnSecond.layers{2}.kernelsize + 1) / pcnnSecond.layers{3}.scale;

[batchesThird, labelsThird ] = extractPyramidBatches(trainData, trainLabel, batchSize(3));
[batchesSecond,labelsSecond] = extractPyramidBatches(trainData, trainLabel, batchSize(2));
[batchesFirst, labelsFirst ] = extractPyramidBatches(trainData, trainLabel, batchSize(1));

%% initiate first pcnn 
fprintf('initiate first pcnn ... \n');
pcnnFirst  = cnnsetup(pcnnFirst,  batchesFirst,  labelsFirst);

%% train first pyramid cnn
fprintf('train first pyramid cnn ... \n');
tic ;
pcnnFirst = cnntrain(pcnnFirst, batchesFirst, labelsFirst, opts);
toc ;

%% initiate second pcnn
fprintf('initiate second pcnn ... \n');
pcnnFirst = cnnPyramidFeedForward(pcnnFirst, batchesSecond);
firstLayerFeatures = pcnnFirst.layers{numel(pcnnFirst.layers)}.a ;
pcnnSecond = cnnsetup(pcnnSecond, firstLayerFeatures{1}, labelsSecond);

%% train second pyramid cnn
fprintf('train second pyramid cnn ... \n');
tic;
featureSize = size(firstLayerFeatures{1});
trainFeatures = zeros(featureSize(1), featureSize(2), featureSize(3) * numel(firstLayerFeatures));
labelFeatures = zeros(size(labelsSecond, 1), size(labelsSecond, 2) * numel(firstLayerFeatures));
for i = 1 : numel(firstLayerFeatures)
    trainFeatures(:, :, 1 + featureSize(3) * (i - 1) : featureSize(3) * i) = firstLayerFeatures{i};
    labelFeatures(:, 1 + size(labelsSecond, 2) * (i - 1) : size(labelsSecond, 2) * i) = labelsSecond;
end
pcnnSecond = cnntrain(pcnnSecond, trainFeatures, labelFeatures, opts);
toc;

%% initiate third pcnn
fprintf('initiate third pcnn ... \n');
pcnnConnected = mergePyramidCNN(pcnnFirst, pcnnSecond);
pcnnConnected = cnnPyramidFeedForward(pcnnConnected, batchesThird);
secondLayerFeatures = pcnnConnected.layers{numel(pcnnConnected.layers)}.a ;
pcnnThird = cnnsetup(pcnnThird, secondLayerFeatures{1}, labelsThird);

%% train third pyramid cnn
fprintf('train third pyramid cnn ... \n');
tic;
featureSize = size(secondLayerFeatures{1});
trainFeatures = zeros(featureSize(1), featureSize(2), featureSize(3) * numel(secondLayerFeatures));
labelFeatures = zeros(size(labelsThird, 1), size(labelsThird, 2) * numel(secondLayerFeatures));
for i = 1 : numel(secondLayerFeatures)
    trainFeatures(:, :, 1 + featureSize(3) * (i - 1) : featureSize(3) * i) = secondLayerFeatures{i};
    labelFeatures(:, 1 + size(labelsThird, 2) * (i - 1) : size(labelsThird, 2) * i) = labelsThird;
end
pcnnThird = cnntrain(pcnnThird, trainFeatures, labelFeatures, opts);
toc;

%% merge pyramid cnn into one
fprintf('merge pyramid cnn ... \n');
pcnn = cnnsetup(pcnn,  trainData,  trainLabel);
pcnn.layers{2} = pcnnFirst. layers{2};
pcnn.layers{3} = pcnnFirst. layers{3};
pcnn.layers{4} = pcnnSecond.layers{2};
pcnn.layers{5} = pcnnSecond.layers{3};
pcnn.layers{6} = pcnnThird. layers{2};
pcnn.layers{7} = pcnnThird. layers{3};
pcnn.ffW = pcnnThird.ffW;
pcnn.ffb = pcnnThird.ffb;

%% test pyramid cnn before tune
fprintf('test entire pyramid cnn before tune ... \n');
[ratio, er, ~] = cnntest(pcnn, testData, testLabel);
fprintf('Accuracy %.2f %% \n', ratio * 100) ;

%% tune entire pyramid cnn
fprintf('tune entire pyramid cnn ... \n');
opts.numepochs = 5 ;       % do not require too large value
tic ;
pcnn = cnntrain(pcnn, trainData, trainLabel, opts);
toc ;

%% test pyramid cnn after tune
fprintf('test entire pyramid cnn after tune ... \n');
[ratio, er, ~] = cnntest(pcnn, testData, testLabel);
fprintf('Accuracy %.2f %% \n', ratio * 100) ;

%% plot mean squared error
% figure(1);
% plot(pcnn.rL);
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 20);
% ylabel('training error');
% xlabel('iterate');

% assert(er<0.12, 'Too big error');
