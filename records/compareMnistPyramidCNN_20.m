clear all; clc;
addpath('./records');
load mnistCNNError_20 ;
load pyramidCNNError_20;
figure
plot(mnistCNNTime_20, mnistCNNError_20, 'r');
hold on;
plot(pyramidCNNTime_20(end - 5 : end), pyramidCNNError_20(end - 5 : end), 'g');
grid on;
set(gca, 'FontSize', 10);
xlabel('epoch');
ylabel('test error');
legend('mnist CNN','pyramid mnist CNN');
