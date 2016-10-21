function [ratio, er, bad] = cnntest(net, x, y)
    %  feedforward
    net = cnnff(net, x);
    [~, h] = max(net.o);
    [~, a] = max(y);
    correct = find(h == a);
    bad = find(h ~= a);
    ratio = numel(correct) / size(y, 2);
    er = numel(bad) / size(y, 2);
    %{
    testNum = size(net.o, 2);
    dxjj = h(:, 1 : 86 + 47) ;
    normother = h(:, 86 + 47 + 1 : end) ;
    dxjjCorrect = size( find( dxjj == repmat(1,1,size(dxjj, 2) ) ) , 2) + size( find( dxjj == repmat(2,1,size(dxjj, 2) ) ) , 2) ;
    normotherCorrect = size( find( normother == repmat(3,1,size(normother, 2) ) ) , 2) + size( find( normother == repmat(4,1,size(normother, 2) ) ) , 2) ;
    fprintf(' in djj/xjj correct %.3f , wrong %.3f \n', dxjjCorrect / (86 + 47), 1 - dxjjCorrect / (86 + 47) ) ;
    fprintf(' in norm/other correct %.3f , wrong %.3f \n', normotherCorrect / (testNum - 86 - 47), 1 - normotherCorrect / (testNum - 86 - 47) ) ;
    %}
end