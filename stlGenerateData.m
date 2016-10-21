function [dataTrain, labelsTrain, dataTest, labelsTest] = stlGenerateData()
%% load data in path
addpath('../stlSubset');
load stlTrainSubset;
load stlTestSubset;
%% get parameters of images and labels
imageSize = size(trainImages, 1);
imageMaps = size(trainImages, 3);
trainNumber = size(trainImages, 4);
testNumber = size(testImages, 4);
classNumber = max(trainLabels);
%% reconstruct labels
labelsTrain = zeros(classNumber, imageMaps, trainNumber);
for i = 1 : trainNumber
    switch trainLabels(i)
        case 1
            label = [1 0 0 0];
        case 2
            label = [0 1 0 0];
        case 3
            label = [0 0 1 0];
        case 4
            label = [0 0 0 1];
    end
    for j = 1 : imageMaps
       labelsTrain(:, j, i)  = label;
    end
end
labelsTest = zeros(classNumber, imageMaps, testNumber);
for i = 1 : testNumber
    switch testLabels(i)
        case 1
            label = [1 0 0 0];
        case 2
            label = [0 1 0 0];
        case 3
            label = [0 0 1 0];
        case 4
            label = [0 0 0 1];
    end
    for j = 1 : imageMaps
       labelsTest(:, j, i)  = label;
    end
end
%% reshape data and labels
dataTrain = reshape(trainImages, imageSize, imageSize, imageMaps * trainNumber);
dataTest  = reshape(testImages,  imageSize, imageSize, imageMaps * testNumber );
labelsTrain = reshape(labelsTrain, classNumber, imageMaps * trainNumber);
labelsTest  = reshape(labelsTest,  classNumber, imageMaps * testNumber );
%% shuffle data and labels    
randNum = randperm(size(dataTrain, 3));
dataTrain = dataTrain(:, :, randNum);
labelsTrain = labelsTrain(:, randNum);

randNum = randperm(size(dataTest, 3));
dataTest = dataTest(:, :, randNum);
labelsTest = labelsTest(:, randNum);

end