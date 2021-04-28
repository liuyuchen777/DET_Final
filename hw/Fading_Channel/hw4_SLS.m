% SLS
A_estimate = x2(1,1) + 1i*x2(1,2);
A_history = zeros(10000,1);
A_history(1) = A_estimate;

for i=2:10000
    A_estimate = A_estimate+(x2(i,1)+1i*x2(i,2)-A_estimate)/(i+1);
    A_history(i) = A_estimate;
end

for i=1:10000
    A_history(i) = sqrt(imag(A_history(i))^2+(real(A_history(i)))^2);
end

figure
plot(1:10000,A_history);
title('Real time estimate result');
xlabel('Time squence n');
ylabel('Absolute estimate value');