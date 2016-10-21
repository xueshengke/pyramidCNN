clear all; clc;
addpath('./records');
load mnistCNNError_50 ;
load pyramidCNNError_50;
figure
plot(mnistCNNTime_50, mnistCNNError_50, 'r');
hold on;
plot(pyramidCNNTime_50(end - 5 : end), pyramidCNNError_50(end - 5 : end), 'g');
grid on;
set(gca, 'FontSize', 12);
xlabel('epoch');
ylabel('test error');
legend('mnist CNN','pyramid mnist CNN');
