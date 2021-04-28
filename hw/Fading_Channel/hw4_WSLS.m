% WSLS
A_estimate = x2(1,1) + 1i*x2(1,2);
A_history = zeros(10000,1);
A_history(1) = A_estimate;
var_x = 0.1; %可改变权重的值观察实验结果
var_estimate = var_x;
K = 0;
J_min = 0;
J_history = zeros(10000,1);
var_history = zeros(10000,1);
var_history(1) = var_x;
for i=2:10000
    K = var_estimate/(var_estimate+var_x);
    J_min = J_min + (x2(i,1)+1i*x2(i,2)-A_estimate)^2/(var_estimate+var_x);
    J_history(i,1) = J_min;
    A_estimate = A_estimate + K*(x2(i,1)+1i*x2(i,2)-A_estimate);
    A_history(i) = A_estimate;
    var_estimate = (1-K)*var_estimate;
    var_history(i) = var_estimate;
end

for i=1:10000
    A_history(i) = sqrt(imag(A_history(i))^2+(real(A_history(i)))^2);
end

subplot(1,2,1);
plot(1:10000,var_history);
title('Real time variance');
xlabel('Time squence n');
ylabel('Variance of estimation');
subplot(1,2,2);
plot(1:10000,A_history);
title('Real time estimate result');
xlabel('Time squence n');
ylabel('Absolute estimate value');