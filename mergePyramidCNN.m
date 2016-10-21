function [ pcnnMerged ] = mergePyramidCNN(pcnnLast, pcnnPresent)
    pcnnMerged = pcnnLast;
    pcnnMerged.layers{end + 1} = pcnnPresent.layers{2};
    pcnnMerged.layers{end + 1} = pcnnPresent.layers{3};
end
