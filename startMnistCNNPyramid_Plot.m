%% Pyramid convolutional neural network application in mnist data
clear all; clc;
addpath(genpath('DeepLearnToolbox'));
load mnist_uint8;

%% reconstruct data and normalize
imageSize = 28;
trainNumber = 60000;
testNumber = 10000;

trainData = double(reshape(train_x',imageSize,imageSize,trainNumber)) / 255;
trainLabel = double(train_y');
testData = double(reshape(test_x',imageSize,imageSize, testNumber)) / 255;
testLabel = double(test_y');
clear train_x train_y test_x test_y;

fprintf('prepare trainData %d * %d * %d \n', size(trainData, 1), size(trainData, 2), size(trainData, 3));
fprintf('prepare trainLabel %d * %d \n', size(trainLabel, 1), size(trainLabel, 2));
fprintf('prepare testData %d * %d * %d \n', size(testData, 1), size(testData, 2), size(testData, 3));
fprintf('prepare testLabel %d * %d \n', size(testLabel, 1), size(testLabel, 2));

%% create a 6c-2s first pyramid convolutional neural network 
fprintf('construct pyramid cnn ... \n');
% rand('state',0);
pcnnFirst.inputmaps = 1;         % gray image
pcnnFirst.classNumber = size(trainLabel, 1);
pcnnFirst.layers = {
    struct('type', 'i')                                     %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)   %convolution layer
    struct('type', 's', 'scale', 2)                         %subsampling layer
};
%% create a 12c-2s second pyramid convolutional neural network 
fprintf('construct pyramid cnn ... \n');
% rand('state',0);
pcnnSecond.inputmaps = pcnnFirst.layers{2}.outputmaps;         % gray image
pcnnSecond.classNumber = size(trainLabel, 1);
pcnnSecond.layers = {
    struct('type', 'i')                                     %input layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5)  %convolution layer
    struct('type', 's', 'scale', 2)                         %subsampling layer
};
%% create a 6c-2s-12c-2s pyramid convolutional neural network 
fprintf('construct pyramid cnn ... \n');
% rand('state',0);
pcnn.inputmaps = 1;         % gray image
pcnn.classNumber = size(trainLabel, 1);
pcnn.layers = {
    struct('type', 'i')                                     %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)   %convolution layer
    struct('type', 's', 'scale', 2)                         %subsampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5)  %convolution layer
    struct('type', 's', 'scale', 2)                         %subsampling layer
};

opts.alpha = 1 ;
opts.batchsize = 100 ;      % select batch from train data
opts.numepochs = 20 ;       % do not require too large value
opts.lowThreshold = 1e-6 ;

%% extract batches from train data
fprintf('extract batches from train data ... \n');
batchSize = zeros(2,1);
batchSize(2) = imageSize;
batchSize(1) = (batchSize(2) - pcnnFirst.layers{2}.kernelsize + 1) / pcnnFirst.layers{3}.scale;

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

featureSize = size(firstLayerFeatures{1});
trainFeatures = zeros(featureSize(1), featureSize(2), featureSize(3) * numel(firstLayerFeatures));
labelFeatures = zeros(size(labelsSecond, 1), size(labelsSecond, 2) * numel(firstLayerFeatures));
for i = 1 : numel(firstLayerFeatures)
    trainFeatures(:, :, 1 + featureSize(3) * (i - 1) : featureSize(3) * i) = firstLayerFeatures{i};
    labelFeatures(:, 1 + size(labelsSecond, 2) * (i - 1) : size(labelsSecond, 2) * i) = labelsSecond;
end

% modify code for record test error
[testBatches, testLabels ] = extractPyramidBatches(testData, testLabel, batchSize(1));
opts.numepochs = 1;
iterateNumber = 10;
testError = zeros(iterateNumber, 1);
for i = 1 : iterateNumber
    fprintf('train epoch %d /%d \n', i ,iterateNumber);
    tic;
    pcnnSecond = cnntrain(pcnnSecond, trainFeatures, labelFeatures, opts);
    toc;
    fprintf('test second pyramid cnn ... \n');
    [ratio, er, ~] = cnntest(pcnnSecond, testBatches, testLabels);
    fprintf('Accuracy %.2f %% \n', ratio * 100) ;
    testError(i) = er;
end

%% merge pyramid cnn into one
fprintf('merge pyramid cnn ... \n');
pcnn = cnnsetup(pcnn,  trainData,  trainLabel);
pcnn.layers{2} = pcnnFirst. layers{2};
pcnn.layers{3} = pcnnFirst. layers{3};
pcnn.layers{4} = pcnnSecond.layers{2};
pcnn.layers{5} = pcnnSecond.layers{3};
pcnn.ffW = pcnnSecond.ffW;
pcnn.ffb = pcnnSecond.ffb;

%% test pyramid cnn before tune
fprintf('test entire pyramid cnn before tune ... \n');
[ratio, er, ~] = cnntest(pcnn, testData, testLabel);
fprintf('Accuracy %.2f %% \n', ratio * 100) ;

% modify code for record test error
testError(end + 1) = er;

%% tune entire pyramid cnn
fprintf('tune entire pyramid cnn ... \n');

% modify code to record test error
opts.numepochs = 1;
iterateNumber = 5;
for i = 1 : iterateNumber
    fprintf('train epoch %d /%d \n', i ,iterateNumber);
    tic ;
    pcnn = cnntrain(pcnn, trainData, trainLabel, opts);
    toc ;
    fprintf('test entire pyramid cnn ... \n');
    [ratio, er, ~] = cnntest(pcnn, testData, testLabel);
    fprintf('Accuracy %.2f %% \n', ratio * 100) ;
    testError(end + 1) = er;
end

%% plot test error
plot(testError);
grid on;
title('pyramid CNN');
xlabel('epoch');
ylabel('test error rate');

%% plot mean squared error
% figure(1);
% plot(pcnn.rL);
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 20);
% ylabel('training error');
% xlabel('iterate');

% assert(er<0.12, 'Too big error');
