% kalman filter
var_prediction = 1;
var_detection = 1;
f = 0.99;
% initialize
A_estimate = x2(1,1) + 1i*x2(1,2);
A_history = zeros(10000,1);
A_history(1) = A_estimate;
MSE = var_prediction;
A_prediction = 0;
MSE_prediction = 0;
kalman_gain = 0;

for i=2:10000
    A_prediction = f*A_estimate;
    MSE_prediction = f^2*MSE+var_prediction;
    kalman_gain = MSE_prediction/(var_detection+MSE_prediction);
    A_estimate = A_prediction+kalman_gain*(x2(i,1)+1i*x2(i,2)-A_prediction);
    A_history(i) = A_estimate;
    MSE = (1-kalman_gain)*MSE_prediction;
end

for i=1:10000
    A_history(i) = sqrt(imag(A_history(i))^2+(real(A_history(i)))^2);
end

plot(1:10000,A_history);
title('Real time estimate result');
xlabel('Time squence n');
ylabel('Absolute estimate value');