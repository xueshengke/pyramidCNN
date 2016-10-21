% net:  cnn 
% x:     part of sample , patch
function net = relucnnff(net, x)
    n = numel(net.layers);
    inputmaps = net.inputmaps;  % 50 to net.inputmaps
    net.layers{1}.a{1} = x;             % should change ?
    % try
%    bathsize = size(x, 3) ;
    
    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                %  initiate all to zero
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);  % 4 dimensions  0 0
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                net.layers{l}.z{j} = z + net.layers{l}.b{j} ;
                net.layers{l}.a{j} = relu( net.layers{l}.z{j} );
                % try  mean to zero 
%                tempwidth = size(net.layers{l}.a{j}, 1) ;
%                tempheight = size(net.layers{l}.a{j}, 2) ;
%                net.layers{l}.a{j}(:) = net.layers{l}.a{j}(:) - mean(net.layers{l}.a{j}(:)) ;
%                net.layers{l}.a{j} = reshape(net.layers{l}.a{j}(:), [tempwidth, tempheight, bathsize]) ;

            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    %  construct map of last layer into a column vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    %  feedforward into output perceptrons
    % check the rows and columns of matrice are suitable
    net.zzz = net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)) ;
    net.o = sigm( net.zzz );

end
