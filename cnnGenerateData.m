function [dataTrain, labelsTrain, dataTest, labelsTest] = cnnGenerateData()
%% data path
path=cell(2,3);
path{1,1}='/home/xueshengke/matlab/data128-c4/bz1';
path{1,2}='/home/xueshengke/matlab/data128-c4/norm1';
path{1,3}='/home/xueshengke/matlab/data128-c4/other1';
path{2,1}='/home/xueshengke/matlab/data128-c4/bz2';
path{2,2}='/home/xueshengke/matlab/data128-c4/norm2';
path{2,3}='/home/xueshengke/matlab/data128-c4/other2';
imgs=cell(size(path));
for i=1:size(path,1)
    for j=1:size(path,2)
        files=dir(fullfile(path{i,j},'*.jpg'));
        disp([path{i,j}, ' has ', num2str(length(files)), ' images.']);
        for k=1:length(files)
            img=imread([path{i,j}, '/', files(k).name]);
            img=double(img);
            img=img/max(max(img));        % normalize
            img=img(:)-mean(img(:));      % mean to zero
            imgs{i,j}.img(:,k)=img(:);
        end
    end
end
ratio=0.65;
numBz1=size(imgs{1,1}.img, 2);
numNorm1=floor(numBz1*0.75);
numOther1=floor(numBz1*0.35);
numBz2=size(imgs{2,1}.img, 2);
numNorm2=floor(numBz2*0.75);
numOther2=floor(numBz2*0.35);
%% rand select 65% from data
sizeBz1=size(imgs{1,1}.img,2);
randNum11=randperm(sizeBz1);
trainData{1,1}.img=imgs{1,1}.img(:, randNum11(1:floor(ratio*numBz1)));
testData{1,1}.img=imgs{1,1}.img(:, randNum11(floor(ratio*numBz1)+1:numBz1));
sizeNorm1=size(imgs{1,2}.img,2);
randNum12=randperm(sizeNorm1);
trainData{1,2}.img=imgs{1,2}.img(:, randNum12(1:floor(ratio*numNorm1)));
testData{1,2}.img=imgs{1,2}.img(:, randNum12(floor(ratio*numNorm1)+1:numNorm1));
sizeOther1=size(imgs{1,3}.img,2);
randNum13=randperm(sizeOther1);
trainData{1,3}.img=imgs{1,3}.img(:,randNum13(1:floor(ratio*numOther1)));
testData{1,3}.img=imgs{1,3}.img(:,randNum13(floor(ratio*numOther1)+1:numOther1));


sizeBz2=size(imgs{2,1}.img,2);
randNum21=randperm(sizeBz2);
trainData{2,1}.img=imgs{2,1}.img(:, randNum21(1:floor(ratio*numBz2)));
testData{2,1}.img=imgs{2,1}.img(:, randNum21(floor(ratio*numBz2)+1:numBz2));
sizeNorm2=size(imgs{2,2}.img,2);
randNum22=randperm(sizeNorm2);
trainData{2,2}.img=imgs{2,2}.img(:, randNum22(1:floor(ratio*numNorm2)));
testData{2,2}.img=imgs{2,2}.img(:, randNum22(floor(ratio*numNorm2)+1:numNorm2));
sizeOther2=size(imgs{2,3}.img,2);
randNum23=randperm(sizeOther2);
trainData{2,3}.img=imgs{2,3}.img(:,randNum23(1:floor(ratio*numOther2)));
testData{2,3}.img=imgs{2,3}.img(:,randNum23(floor(ratio*numOther2)+1:numOther2));

%% train data
djjBzTrainData=trainData{1,1}.img;
xjjBzTrainData=trainData{2,1}.img;
normTrainData=[trainData{1,2}.img,trainData{2,2}.img];
otherTrainData=[trainData{1,3}.img,trainData{2,3}.img];
dataTrain=[djjBzTrainData,xjjBzTrainData,normTrainData,otherTrainData];

%% test data
djjBzTestData=testData{1,1}.img;
xjjBzTestData=testData{2,1}.img;
normTestData=[testData{1,2}.img,testData{2,2}.img];
otherTestData=[testData{1,3}.img,testData{2,3}.img];
dataTest=[djjBzTestData,xjjBzTestData,normTestData,otherTestData];

% labels for softmax classifier
% trainLabels=[repmat(1,1,size(djjBzTrainData,2)),repmat(2,1,size(xjjBzTrainData,2)),...
%     repmat(3,1,size(normTrainData,2)), repmat(4,1,size(otherTrainData,2))];
% testLabels=[repmat(1,1,size(djjBzTestData,2)),repmat(2,1,size(xjjBzTestData,2)),...
%     repmat(3,1,size(normTestData,2)), repmat(4,1,size(otherTestData,2))];

% labels for one of k classifier
trainLabels=[repmat([1,0,0,0]',1,size(djjBzTrainData,2)),repmat([0,1,0,0]',1,size(xjjBzTrainData,2)),...
    repmat([0,0,1,0]',1,size(normTrainData,2)), repmat([0,0,0,1]',1,size(otherTrainData,2))];
testLabels=[repmat([1,0,0,0]',1,size(djjBzTestData,2)),repmat([0,1,0,0]',1,size(xjjBzTestData,2)),...
    repmat([0,0,1,0]',1,size(normTestData,2)), repmat([0,0,0,1]',1,size(otherTestData,2))];
%% shuffle train data and test data
trainNum=size(trainLabels,2);
randNum=randperm(trainNum);
dataTrain=dataTrain(:,randNum);
labelsTrain=trainLabels(:, randNum);

%% shuffle test data
 testNum=size(testLabels,2);
 randNum=randperm(testNum);
 dataTest=dataTest(:,randNum);
 labelsTest=testLabels(:, randNum);
% labelsTest = testLabels ;

%{
save('dataTrain','dataTrain','-v7.3');
save('labelsTrain','labelsTrain','-v7.3');
save('dataTest','dataTest','-v7.3');
save('labelsTest','labelsTest','-v7.3');
%}

end