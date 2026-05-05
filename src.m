M = 4
clc; clear; close all;
% Step 1: Define standard constellations
% QPSK (unit-energy)
X_QPSK = [1 1; 1 -1; -1 1; -1 -1]/sqrt(2);
% 4-PAM along x-axis (1D)
X_4PAM = [-sqrt(9/5) 0; -sqrt(1/5) 0; sqrt(1/5) 0; sqrt(9/5) 0];
% Normalize to zero-mean and unit energy
normalize = @(X) (X - mean(X)) / sqrt(mean(sum((X - mean(X)).^2,2)));
X_QPSK   = normalize(X_QPSK);
X_4PAM   = normalize(X_4PAM);
% Step 2: Optimize 4-symbol constellation
obj_dmin = @(X) -min(pdist(reshape(X,[4,2])));
% Nested constraint function
function [c, ceq] = constraints4(X)
   pts = reshape(X,[4,2]);
   ceq = [sum(pts(:,1)); sum(pts(:,2)); mean(sum(pts.^2,2)) - 1];
   c = [];
end
X0 = randn(8,1); % initial guess
opts = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxFunctionEvaluations',1e5);
X_opt = fmincon(obj_dmin, X0, [], [], [], [], [], [], @constraints4, opts);
X_custom = reshape(X_opt,[4,2]);
X_custom = normalize(X_custom);
fprintf('Symbol\tIn-Phase\tQuadrature\n');
for k = 1:4
   fprintf('%d\t%.4f\t\t%.4f\n', k, X_custom(k,1), X_custom(k,2));
end
% Step 3: Compute minimum distances
dmin_QPSK   = min(pdist(X_QPSK));
dmin_4PAM   = min(pdist(X_4PAM));
dmin_custom = min(pdist(X_custom));
fprintf('Minimum distance dmin:\nQPSK: %.4f\nCustom: %.4f\n4-PAM: %.4f\n', ...
   dmin_QPSK, dmin_custom, dmin_4PAM);
% Step 4: Plot constellations
% QPSK
figure;
scatter(X_QPSK(:,1), X_QPSK(:,2), 100, 'bo','filled'); hold on;
voronoi(X_QPSK(:,1), X_QPSK(:,2));
text(X_QPSK(:,1)+0.05, X_QPSK(:,2)+0.05, {'1','2','3','4'});
title('QPSK Constellation');
xlabel('In-Phase'); ylabel('Quadrature'); grid on; axis equal;
% Optimal Design
figure;
scatter(X_custom(:,1), X_custom(:,2), 100, 'ro','filled'); hold on;
voronoi(X_custom(:,1), X_custom(:,2));
text(X_custom(:,1)+0.05, X_custom(:,2)+0.05, {'1','2','3','4'});
title('Optimal Design');
xlabel('In-Phase'); ylabel('Quadrature'); grid on; axis equal;
% 4-PAM
% 4-PAM with Decision Regions
figure; hold on; grid on;
% Plot the 4-PAM symbols
scatter(X_4PAM(:,1), X_4PAM(:,2), 100, 'gs', 'filled');
% Compute the decision boundaries (midpoints between adjacent symbols)
x_sorted = sort(X_4PAM(:,1));
decision_boundaries = (x_sorted(1:end-1) + x_sorted(2:end)) / 2;
% Plot vertical decision boundary lines
for b = decision_boundaries'
   xline(b, 'k--', 'LineWidth', 1.2);
end
% Plot a reference axis
yline(0, 'k:');
% Label each symbol
text(X_4PAM(:,1)+0.05, X_4PAM(:,2)+0.05, {'1','2','3','4'});
% Set plot formatting
title('4-PAM Constellation');
xlabel('In-Phase (I)');
ylabel('Quadrature (Q)');
axis equal;
xlim([min(X_4PAM(:,1))-0.5, max(X_4PAM(:,1))+0.5]);
ylim([-0.5, 0.5]);
% Step 5: Theoretical SER Upper Bounds
M = 4;
SNR_dB = 0:3:15;             
sigma = 10.^-(SNR_dB./20);
SER_QPSK_theory   = (M-1) * qfunc(dmin_QPSK ./ (2*sigma));
SER_custom_theory = (M-1) * qfunc(dmin_custom ./ (2*sigma));
SER_4PAM_theory   = (M-1) * qfunc(dmin_4PAM ./ (2*sigma));
% Clip SER to 1
SER_QPSK_theory(SER_QPSK_theory>1) = 1;
SER_custom_theory(SER_custom_theory>1) = 1;
SER_4PAM_theory(SER_4PAM_theory>1) = 1;
% Step 6: Simulated SER
Nsim = 1e6;  % number of transmitted symbols
SER_custom_sim = zeros(size(sigma));
SER_QPSK_sim   = zeros(size(sigma));
SER_4PAM_sim   = zeros(size(sigma));
for k = 1:length(sigma)
   % Noise standard deviation for this SNR
   s = sigma(k);
   % Optimal
   idx = randi(M, Nsim, 1);          % random symbols
   Xtx = X_custom(idx,:);            % transmitted points
   N = s * randn(Nsim,2);           % AWGN noise
   Y = Xtx + N;                      % received points
   % Nearest-neighbor detection
   [~, idx_hat] = min(pdist2(Y, X_custom), [], 2);
   SER_custom_sim(k) = mean(idx ~= idx_hat);
   % QPSK
   idx = randi(M, Nsim, 1);
   Xtx = X_QPSK(idx,:);
   N = s * randn(Nsim,2);
   Y = Xtx + N;
   [~, idx_hat] = min(pdist2(Y, X_QPSK), [], 2);
   SER_QPSK_sim(k) = mean(idx ~= idx_hat);
   % 4-PAM
   idx = randi(M, Nsim, 1);
   Xtx = X_4PAM(idx,:);
   N = s * randn(Nsim,2);
   Y = Xtx + N;
   [~, idx_hat] = min(pdist2(Y, X_4PAM), [], 2);
   SER_4PAM_sim(k) = mean(idx ~= idx_hat);
end
% Step 7: Plot simulated vs theoretical SER
% Optimal constellation
figure; hold on; grid on;
semilogy(SNR_dB, SER_custom_sim, 'ro-', 'LineWidth',1.5,'MarkerSize',8);
semilogy(SNR_dB, SER_custom_theory, 'r--','LineWidth',2);
xlabel('SNR [dB]'); ylabel('Symbol Error Rate (SER)');
title('Optimal Constellation Design: Simulated vs Theoretical SER (M=4)');
legend('Simulated SER','Theoretical Upper Bound','Location','southwest');
set(gca,'YScale','log'); ylim([1e-5 1]); xlim([min(SNR_dB) max(SNR_dB)]);
% QPSK and 4-PAM
figure; hold on; grid on;
semilogy(SNR_dB, SER_QPSK_sim, 'ko-', 'LineWidth',1.5,'MarkerSize',8);
semilogy(SNR_dB, SER_QPSK_theory, 'k--','LineWidth',2);
semilogy(SNR_dB, SER_4PAM_sim, 'gs-', 'LineWidth',1.5,'MarkerSize',8);
semilogy(SNR_dB, SER_4PAM_theory, 'g--','LineWidth',2);
xlabel('SNR [dB]'); ylabel('Symbol Error Rate (SER)');
title('QPSK and 4-PAM: Simulated vs Theoretical SER');
legend('QPSK Sim','QPSK Theory','4-PAM Sim','4-PAM Theory','Location','southwest');
set(gca,'YScale','log'); ylim([1e-5 1]); xlim([min(SNR_dB) max(SNR_dB)]);
figure; hold on; grid on;
semilogy(SNR_dB, SER_QPSK_sim, 'bo-', 'LineWidth',1.5, 'MarkerSize',8);
semilogy(SNR_dB, SER_custom_sim, 'ro-', 'LineWidth',1.5, 'MarkerSize',8);
semilogy(SNR_dB, SER_4PAM_sim, 'gs-', 'LineWidth',1.5, 'MarkerSize',8);
xlabel('SNR [dB]');
ylabel('Symbol Error Rate (SER)');
title('Simulated Symbol Error Rate vs SNR (M = 4)');
legend('QPSK','Optimal Design','4-PAM','Location','southwest');
set(gca,'YScale','log');
ylim([1e-5 1]);
xlim([0 15]);
grid on;
M = 8
clc; clear; close all;
% Step 1: Define standard 8-PSK constellation
X_8PSK = [ 1 0; -1 0; 0 1; 0 -1; ...
          1/sqrt(2) 1/sqrt(2); -1/sqrt(2) 1/sqrt(2); ...
          1/sqrt(2) -1/sqrt(2); -1/sqrt(2) -1/sqrt(2)];
% Normalize to zero-mean and unit-energy
normalize = @(X) (X - mean(X)) / sqrt(mean(sum((X - mean(X)).^2,2)));
X_8PSK = normalize(X_8PSK);
% Step 2: Optimize 8-symbol constellation
obj_dmin8 = @(X) -min(pdist(reshape(X,[8,2])));
function [c, ceq] = constraints8(X)
   pts = reshape(X,[8,2]);
   ceq = [sum(pts(:,1)); sum(pts(:,2)); mean(sum(pts.^2,2)) - 1];
   c = [];
end
X0 = randn(16,1);
opts = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxFunctionEvaluations',1e5);
X_opt8 = fmincon(obj_dmin8, X0, [],[],[],[],[],[], @constraints8, opts);
X_custom8 = reshape(X_opt8,[8,2]);
X_custom8 = normalize(X_custom8);
fprintf('Symbol\tIn-Phase\tQuadrature\n');
for k = 1:8
   fprintf('%d\t%.4f\t\t%.4f\n', k, X_custom8(k,1), X_custom8(k,2));
end
% Step 3: Compute minimum distances
dmin_8PSK = min(pdist(X_8PSK));
dmin_custom8 = min(pdist(X_custom8));
fprintf('Minimum distance dmin (M=8):\n8-PSK: %.4f\nCustom: %.4f\n', dmin_8PSK, dmin_custom8);
% Step 4: Plot constellations
figure;
scatter(X_8PSK(:,1), X_8PSK(:,2), 100, 'bo','filled'); hold on;
voronoi(X_8PSK(:,1), X_8PSK(:,2));
text(X_8PSK(:,1)+0.05, X_8PSK(:,2)+0.05, arrayfun(@num2str,1:8,'UniformOutput',false));
title('8-PSK Constellation');
xlabel('In-Phase'); ylabel('Quadrature'); grid on; axis equal;
figure;
scatter(X_custom8(:,1), X_custom8(:,2), 100, 'ro','filled'); hold on;
voronoi(X_custom8(:,1), X_custom8(:,2));
text(X_custom8(:,1)+0.05, X_custom8(:,2)+0.05, arrayfun(@num2str,1:8,'UniformOutput',false));
title('Optimal Constellation');
xlabel('In-Phase'); ylabel('Quadrature'); grid on; axis equal;
% Step 5: Theoretical SER upper bounds
M = 8;
SNR_dB = 0:3:15;
sigma = 10.^-(SNR_dB./20);
SER_8PSK_theory   = (M-1) * qfunc(dmin_8PSK ./ (2*sigma));
SER_custom_theory = (M-1) * qfunc(dmin_custom8 ./ (2*sigma));
SER_8PSK_theory(SER_8PSK_theory>1) = 1;
SER_custom_theory(SER_custom_theory>1) = 1;
% Step 6: Simulated SER
Nsim = 1e6;
SER_8PSK_sim = zeros(size(sigma));
SER_custom_sim = zeros(size(sigma));
for k = 1:length(sigma)
   s = sigma(k);
  
   % 8-PSK simulation
   idx = randi(M, Nsim, 1);
   Xtx = X_8PSK(idx,:);
   N = s*randn(Nsim,2);
   Y = Xtx + N;
   [~, idx_hat] = min(pdist2(Y, X_8PSK), [], 2);
   SER_8PSK_sim(k) = mean(idx ~= idx_hat);
  
   % Optimal 8-symbol simulation
   idx = randi(M, Nsim, 1);
   Xtx = X_custom8(idx,:);
   N = s*randn(Nsim,2);
   Y = Xtx + N;
   [~, idx_hat] = min(pdist2(Y, X_custom8), [], 2);
   SER_custom_sim(k) = mean(idx ~= idx_hat);
end
