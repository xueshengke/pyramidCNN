function [pred] = softmaxPredict(softmaxModel, data)
theta = softmaxModel.optTheta;
[~,pred]=max(theta*data);
end

