function net = relucnntrain( net, x, y, opts )
   % sample number
    m = size(x, 3);     % 3 to 4
    % times of each batchsize
    numbatches = m / opts.batchsize;    % 
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    % save value of cost
    net.rL = [];
    for i = 1 : opts.numepochs
      %  disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        %tic;
        % m integers randomly
        kk = randperm(m);
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));    
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            % compute the output of cnn, feedforward
            net = relucnnff(net, batch_x);
            % compute the delta of cnn, back propagation
            % obtain gradients respect to weights and biases
            net = relucnnbp(net, batch_y);
            % update the value of weights and biases, using the gradients
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
%             net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
            net.rL(end + 1) = net.L;
%             if net.L < opts.minerror
%                 disp('error lower than threshold 10^(-6)');
%                 return ;
%             end
        end
        %toc;
    end
    
end

