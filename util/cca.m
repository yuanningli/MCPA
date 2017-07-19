function [Wx, Wy, r] = cca(X,Y)

% this function computes canonical correlation analysis between X and Y and 
% returns the canonical correlation weights for the two sets of variables
% as well as the canonical correlations for each pair of canonical variates
%
% Inputs:
%     X - N x p1 data matrix for the first set of variables
%     Y - N x p2 data matrix for the second set of variables
%
% Outputs:
%     Wx - p1 x p canonical coefficients for the first set of variables
%     Wy - p2 x p canonical coefficients for the second set of variables
%     r - p x 1 dimensional vector with corresponding canonical correlations
%
% Adapted from Magnus Borga, 2000 @ Linköping University

% --- Calculate covariance matrices ---

X = X';
Y = Y';
z = [X;Y];
C = cov(z.');
sx = size(X,1);
sy = size(Y,1);
Cxx = C(1:sx, 1:sx) + 10^(-6)*eye(sx);
Cxy = C(1:sx, sx+1:sx+sy);
Cyx = Cxy';
Cyy = C(sx+1:sx+sy, sx+1:sx+sy) + 10^(-6)*eye(sy);

% --- Calcualte Wx and r ---

[Wx,r] = eig(Cxx\Cxy/Cyy*Cyx); % Basis in X
r = sqrt(real(r));      % Canonical correlations

% --- Sort correlations ---

V = fliplr(Wx);		% reverse order of eigenvectors
r = flipud(diag(r));	% extract eigenvalues anr reverse their orrer
[r,I]= sort((real(r)));	% sort reversed eigenvalues in ascending order
r = flipud(r);		% restore sorted eigenvalues into descending order
for j = 1:length(I)
  Wx(:,j) = V(:,I(j));  % sort reversed eigenvectors in ascending order
end
Wx = fliplr(Wx);	% restore sorted eigenvectors into descending order

% --- Calcualte Wy  ---

Wy = Cyy\Cyx*Wx;     % Basis in Y
Wy = Wy./repmat(sqrt(sum(abs(Wy).^2)),sy,1); % Normalize Wy
