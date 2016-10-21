% net:  cnn trained
% x:     train data
% y:     train label
% opts: parameters
function net = cnntrain(net, x, y, opts)
    % sample number
    m = size(x, 3);      
    % times of each batchsize
    numbatches = m / opts.batchsize;    
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    % save value of cost
    net.rL = [];
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        %tic;
        % m integers randomly
        kk = randperm(m);
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));    
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            % compute the output of cnn, feedforward
            net = cnnff(net, batch_x);
            % compute the delta of cnn, back propagation
            % obtain gradients respect to weights and biases
            net = cnnbp(net, batch_y);
            % update the value of weights and biases, using the gradients
            net = cnnapplygrads(net, opts);
            net.rL(end + 1) = net.L;
            if net.L < opts.lowThreshold
                disp('train process stops due to lower threshold');
                return ;
            end
        end
        %toc;
    end
    
end
