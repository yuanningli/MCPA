function [pred_tag, corr_cond1, corr_cond2] = mcpa(train_set_cond1_node1,train_set_cond2_node1,...
    train_set_cond1_node2,train_set_cond2_node2,test_data_node1,test_data_node2,...
    varargin)
% The main function that performs multi-connection pattern analysis, a
% binary classification based on single trial funcional connectivity
% pattern
%
% Utility:
% [pred_tag, corr_cond1, corr_cond2] = mcpa(train_set_cond1_node1,train_set_cond2_node1,...
%     train_set_cond1_node2,train_set_cond2_node2,test_data_node1,test_data_node2);
%
% [pred_tag, corr_cond1, corr_cond2] = mcpa(train_set_cond1_node1,train_set_cond2_node1,...
%     train_set_cond1_node2,train_set_cond2_node2,test_data_node1,test_data_node2,...
%     num_cc);
%
% [pred_tag, corr_cond1, corr_cond2] = mcpa(train_set_cond1_node1,train_set_cond2_node1,...
%     train_set_cond1_node2,train_set_cond2_node2,test_data_node1,test_data_node2,...
%     num_cc,pc_thres);
%
% Input:
%     train_set_cond1_node1 - N1xp1 data matrix for condition 1, ROI 1
%     train_set_cond2_node1 - N2xp1 data matrix for condition 2, ROI 1
%     train_set_cond1_node2 - N1xp2 data matrix for condition 1, ROI 2
%     train_set_cond2_node2 - N2xp2 data matrix for condition 2, ROI 2
%     test_data_node1 - Nxp1 data data for ROI1
%     test_data_node2 - Nxp2 data data for ROI2
%     num_cc(optional) - number of pairs of canonical variates to use
%     pc_thres(optional) -  percent of variance used for PCA
%
% Output:
%     pred_tag - N x 1 vector as the predictions for the test data
%     corr_cond1 - N x 1 vector, reconstruction correlation under condition 1
%     corr_cond2 - N x 1 vector, reconstruction correlation under condition 2
%
% @ 2017 Yuanning Li    ynli@cmu.edu

if ~isempty(varargin)  %  if number of CC to use is specified
    num_cc = varargin{1};
else
    num_cc = min(size(train_set_cond1_node1,2),size(train_set_cond1_node2),2);
end

if length(varargin) == 2  % if PCA is specified
    pc_thres = varargin{2};

    [w_pc_cond1_node1,~,latent_cond1_node1] = pca(train_set_cond1_node1);
    num_pc_cond1_node1 = find_numpc(latent_cond1_node1, pc_thres);
    [w_pc_cond1_node2,~,latent_cond1_node2] = pca(train_set_cond1_node2);
    num_pc_cond1_node2 = find_numpc(latent_cond1_node2, pc_thres);
    [w_pc_cond2_node1,~,latent_cond2_node1] = pca(train_set_cond2_node1);
    num_pc_cond2_node1 = find_numpc(latent_cond2_node1, pc_thres);
    [w_pc_cond2_node2,~,latent_cond2_node2] = pca(train_set_cond2_node2);
    num_pc_cond2_node2 = find_numpc(latent_cond2_node2, pc_thres);

    num_pc_node1 = min(num_pc_cond1_node1,num_pc_cond2_node1);
    num_pc_node2 = min(num_pc_cond1_node2,num_pc_cond2_node2);

    train_set_cond1_node1 = train_set_cond1_node1 * w_pc_cond1_node1(:,1:num_pc_node1);
    train_set_cond1_node2 = train_set_cond1_node2 * w_pc_cond1_node2(:,1:num_pc_node2);
    train_set_cond2_node1 = train_set_cond2_node1 * w_pc_cond2_node1(:,1:num_pc_node1);
    train_set_cond2_node2 = train_set_cond2_node2 * w_pc_cond2_node2(:,1:num_pc_node2);
    
    test_data_node1_cond1 = test_data_node1 * w_pc_cond1_node1(:,1:num_pc_node1);
    test_data_node1_cond2 = test_data_node1 * w_pc_cond2_node1(:,1:num_pc_node1);
    test_data_node2_cond1 = test_data_node2 * w_pc_cond1_node2(:,1:num_pc_node2);
    test_data_node2_cond2 = test_data_node2 * w_pc_cond2_node2(:,1:num_pc_node2);

    num_pc = min(num_pc_node1,num_pc_node2);
    num_cc = min(num_cc,num_pc);
else
    test_data_node1_cond1 = test_data_node1;
    test_data_node1_cond2 = test_data_node1;
    
    test_data_node2_cond1 = test_data_node2;
    test_data_node2_cond2 = test_data_node2;
end

% compute the CCA model
[ccA1,ccB1,r1] = cca(train_set_cond1_node1,train_set_cond1_node2);
[ccA2,ccB2,r2] = cca(train_set_cond2_node1,train_set_cond2_node2);

% if use the builtin cca function by MATLAB
% [ccA1,ccB1,r1] = canoncorr(train_set_cond1_node1,train_set_cond1_node2);
% [ccA2,ccB2,r2] = canoncorr(train_set_cond2_node1,train_set_cond2_node2);

A1 = ccA1(:,1:num_cc);
B1 = ccB1(:,1:num_cc);
A2 = ccA2(:,1:num_cc);
B2 = ccB2(:,1:num_cc);

% build bi-directional linear mappings between the two ROIs using OLS
A_cond1 = A1 * B1'/(B1*B1' + (1e-6)*eye(size(B1,1)));
A_cond2 = A2 * B2'/(B2*B2' + (1e-6)*eye(size(B2,1)));
B_cond1 = B1 * A1'/(A1*A1' + (1e-6)*eye(size(A1,1)));
B_cond2 = B2 * A2'/(A2*A2' + (1e-6)*eye(size(A2,1)));

% project and reconstruct the testing data throught the linear mappings

test_data_cond1_recon_node2 = test_data_node1_cond1 * A_cond1;
test_data_cond2_recon_node2 = test_data_node1_cond2 * A_cond2;

test_data_cond1_recon_node1 = test_data_node2_cond1 * B_cond1;
test_data_cond2_recon_node1 = test_data_node2_cond2 * B_cond2;

% compute correlation between reconstructed data and actual testing data
% assign classification labels to the trials
for n = 1 : size(test_data_node1,1)
    corr_cond1_node2 = test_data_cond1_recon_node2(n,:) * test_data_node2_cond1(n,:)'...
        / (norm(test_data_cond1_recon_node2(n,:),2) * norm(test_data_node2_cond1(n,:),2));
    corr_cond2_node2 = test_data_cond2_recon_node2(n,:) * test_data_node2_cond2(n,:)'...
        / (norm(test_data_cond2_recon_node2(n,:),2) * norm(test_data_node2_cond2(n,:),2));
    corr_cond1_node1 = test_data_cond1_recon_node1(n,:) * test_data_node1_cond1(n,:)'...
        / (norm(test_data_cond1_recon_node1(n,:),2) * norm(test_data_node1_cond1(n,:),2));
    corr_cond2_node1 = test_data_cond2_recon_node1(n,:) * test_data_node1_cond2(n,:)'...
        / (norm(test_data_cond2_recon_node1(n,:),2) * norm(test_data_node1_cond2(n,:),2));
    
    corr_cond1(n) = (corr_cond1_node1 + corr_cond1_node2)/2;
    corr_cond2(n) = (corr_cond2_node1 + corr_cond2_node2)/2;
    
    if corr_cond1(n) >= corr_cond2(n)
        pred_tag(n,1) = 1;
    else
        pred_tag(n,1) = 2;
    end
end