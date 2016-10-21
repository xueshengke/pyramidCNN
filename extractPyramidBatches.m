function [batches, labels] = extractPyramidBatches(trainData, trainLabel, batchSize)
    imageSize = size(trainData, 1);
    trainNumber = size(trainData, 3);
    classNumber = size(trainLabel, 1);
    if batchSize >= imageSize
        batches = trainData;
        labels = trainLabel;
        return ;
    end
    batchNumber = imageSize - batchSize + 1;
    batchLength = floor(batchNumber / 2);
%     batches = zeros(batchSize, batchSize, batchLength ^ 2, trainNumber);
%     labels  = zeros(classNumber, batchLength ^ 2, trainNumber);
    batches = zeros(batchSize, batchSize, batchLength, trainNumber);
    labels  = zeros(classNumber, batchLength, trainNumber);
    
    rowPos    = randperm(batchNumber);
    columnPos = randperm(batchNumber);
    
%     for i = 1 : trainNumber
%         for j = 1 : batchLength
%             for k = 1 : batchLength
%                 batches(:,:, (j - 1) * batchLength + k, i) = ...
%                     trainData(rowPos(j) : rowPos(j) + batchSize - 1, ...
%                         columnPos(k) : columnPos(k) + batchSize - 1, i);
%                 labels(:, (j - 1) * batchLength + k, i) = ...
%                     trainLabel(:, i);
%             end
%         end
%     end    

    for i = 1 : trainNumber
        for j = 1 : batchLength
            batches(:,:, j, i) = ...
                trainData(rowPos(j) : rowPos(j) + batchSize - 1, ...
                    columnPos(j) : columnPos(j) + batchSize - 1, i);
            labels(:, j, i) = ...
                trainLabel(:, i);
        end
    end   
    
%     batches = reshape(batches, batchSize, batchSize, batchLength ^ 2 * trainNumber);
%     labels  = reshape(labels, classNumber, batchLength ^ 2 * trainNumber);
    batches = reshape(batches, batchSize, batchSize, batchLength * trainNumber);
    labels  = reshape(labels, classNumber, batchLength * trainNumber);
    
    randNum = randperm( size(batches, 3) );
    batches = batches(:, :, randNum);
    labels  = labels(:, randNum);
end
