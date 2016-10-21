function [outputFeatures] = cnnLayerFeedForward(net, num, inputFeatures)
	%% convolutional layer
    for j = 1 : net.layers{num}.outputmaps   %  for each output map
        %  create temp output map
        %  initiate all to zero
        z = zeros(size(inputFeatures) - [net.layers{num}.kernelsize - 1 net.layers{num}.kernelsize - 1 0]);
        if num > 2
            inputmaps = net.layers{num - 2}.outputmaps;
        else
            inputmaps = 1;
        end
        for i = 1 : inputmaps   %  for each input map
            %  convolve with corresponding kernel and add to temp output map
            z = z + convn(inputFeatures, net.layers{num}.k{i}{j}, 'valid');
        end
        %  add bias, pass through nonlinearity
        net.layers{num}.a{j} = sigm(z + net.layers{num}.b{j});
    end
    %% downsample layer
    for j = 1 : net.layers{num}.outputmaps
        z = convn(net.layers{num}.a{j}, ones(net.layers{num + 1}.scale) / (net.layers{num + 1}.scale ^ 2), 'valid');   %  !! replace with variable
        net.layers{num + 1}.a{j} = z(1 : net.layers{num + 1}.scale : end, 1 : net.layers{num + 1}.scale : end, :);
    end
    outputFeatures = net.layers{num + 1}.a;
end

function X = sigm(P)
	X = 1./(1+exp(-P));
end