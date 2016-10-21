% net:  cnn
% y:     labels according to data
function net = relucnnbp(net, y)
    n = numel(net.layers);
    %   error
    net.e = net.o - y;
    %  loss function
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);

    %%  backprop deltas
    % delta of output layer
 %  net.od = net.e .* dsoftplus(net.zzz);           %  output delta softplus
     net.od = net.e .* net.o .* (1 - net.o);          %  output delta sigmoid
    net.fvd = (net.ffW' * net.od);                %  feature vector delta
    % gradient of current layer depends on convolutional layer or downsampling layer
    % in downsampling layer, activating function is linear
    if strcmp(net.layers{n}.type, 'c')         %  if last layer is convolutional layer
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
        error('never happened.');
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)  % delta is divided by number of elements when upsampling
                net.layers{l}.d{j} = drelu(net.layers{l}.z{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
            end
        elseif strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)  % upsampling layer, activating funciton is linear, the gradient is 1
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %%  calc gradients
    % gradients of delta respect to kernels (kernels like weights in traditional neural networks)
    % flipall = rot180
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
