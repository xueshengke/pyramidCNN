clear all; clc;
addpath('./records');
load pyramidCNNError_20 ;
load pyramidCNNError_50;
figure
plot(pyramidCNNError_20, 'r');
hold on;
plot(pyramidCNNError_50, 'g');
grid on;
set(gca, 'FontSize', 10);
xlabel('epoch');
ylabel('test error');
legend('pyramid CNN error 20','pyramid CNN error 50');
