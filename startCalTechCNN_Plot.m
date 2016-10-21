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
opts.numepochs = 1;      % long time  seconds per poches
opts.lowThreshold = 1e-6 ;
%%
fprintf('initiate cnn....\n');
cnn = cnnsetup(cnn, trainData, trainLabel);

iterNumber = 500;
testErrorRate = zeros(iterNumber, 1);
runningTime = zeros(iterNumber, 1);
for i = 1 : iterNumber
    %% start training cnn network
%    fprintf('commence training cnn ... \n');
    tic ;
    cnn = cnntrain(cnn, trainData, trainLabel, opts);
    loopTime = toc ;
    fprintf('eclapsed time %d seconds \n', loopTime);
    if i == 1
        runningTime(i) = loopTime;
    else
        runningTime(i) = runningTime(i - 1) + loopTime;
    end
    %% start test cnn network
%    fprintf('commence testing cnn ... \n');
    [ratio, error, bad] = cnntest(cnn, testData, testLabel);
    fprintf('%d / %d, Accuracy %.2f %%\n', i, iterNumber, ratio * 100) ;
    testErrorRate(i) = error ;
end
%% plot test error rate 
plot(testErrorRate);
grid on ;
title('calTech CNN');
xlabel('epoch');
ylabel('test error rate');


