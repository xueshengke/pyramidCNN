% stlSubset for CNN  
clear all; close all; clc;
addpath(genpath('DeepLearnToolbox'));

%% load dataset
[trainData, trainLabel, testData, testLabel] = calTech101GenerateData();

fprintf('prepare trainData %d * %d * %d \n', size(trainData, 1), size(trainData, 2), size(trainData, 3));
fprintf('prepare trainLabel %d * %d \n', size(trainLabel, 1), size(trainLabel, 2));
fprintf('prepare testData  %d * %d * %d \n', size(testData, 1), size(testData, 2), size(testData, 3));
fprintf('prepare testLabel  %d * %d \n', size(testLabel, 1), size(testLabel, 2));

%% CNN design
% rand('state',0);
cnn.inputmaps = 1;         % 
cnn.classNum = size(trainLabel, 1);
cnn.layers = {
    struct('type', 'i') 
    struct('type', 'c', 'outputmaps', 6,  'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 12,  'kernelsize', 3)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 10,  'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 8,  'kernelsize', 5)
    struct('type', 's', 'scale', 2)
};
opts.alpha = 1 ;
opts.batchsize = 100 ;     % needs to change according to train number
opts.numepochs = 100;      % long time  seconds per poches
opts.lowThreshold = 1e-6 ;
%% initialize cnn 
fprintf('initiate cnn....\n');
cnn = cnnsetup(cnn, trainData, trainLabel);

%% start training cnn
fprintf('commence training cnn ... \n');
tic ;
cnn = cnntrain(cnn, trainData, trainLabel, opts);
toc ;

%% start testing cnn
fprintf('commence testing cnn ... \n');
[ratio, error, bad] = cnntest(cnn, testData, testLabel);
fprintf('Accuracy %.2f %%\n', ratio * 100) ;

%% plot test error rate 
% plot(testErrorRate);
% grid on ;
% title('stl CNN');
% xlabel('epoch');
% ylabel('test error rate');


