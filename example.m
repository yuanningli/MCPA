%% load example dataset
load ./data/sample_data.mat
num_conditions = length(unique(labels));
num_cc = min(size(data_node1,2),size(data_node2,2));

%% sort the trials according to their conditions
category_cnt = zeros(num_conditions,1);
clear category1 category2

for n = 1 : size(labels,1)
    cond = labels(n);
    category_cnt(cond) = category_cnt(cond) + 1;
    category1{cond}(category_cnt(cond),:) = data_node1(n,:);
    category2{cond}(category_cnt(cond),:) = data_node2(n,:);
end


%% evaluate the performance of MCPA using leave-one-out cross-validation

% loop through each pair of conditions
condidx = 1:num_conditions;
for i = 1:length(condidx)
    for j = (i+1) : length(condidx)
        
        data_node1_cond1 = category1{condidx(i)}(:,:);
        data_node1_cond2 = category1{condidx(j)}(:,:);
        data_node2_cond1 = category2{condidx(i)}(:,:);
        data_node2_cond2 = category2{condidx(j)}(:,:);
        
        data_tags = [ones(size(data_node1_cond1,1),1); 2*ones(size(data_node1_cond2,1),1)];
        data_node1_orig = cat(1,data_node1_cond1,data_node1_cond2);
        data_node2_orig = cat(1,data_node2_cond1,data_node2_cond2);
        
        pred_tag = zeros(size(data_tags));
        
        % leave-one-out cross-validation
        for n = 1 : size(data_tags,1)
            train_idx = setdiff(1:size(data_tags,1),n);
            test_idx = n;
            train_tags = data_tags(train_idx);
            test_tags = data_tags(test_idx);
            train_data_node1 = data_node1_orig(train_idx,:);
            train_data_node2 = data_node2_orig(train_idx,:);
            test_vector_node1 = data_node1_orig(test_idx,:);
            test_vector_node2 = data_node2_orig(test_idx,:);
                        
            train_set_cond1_node1 = train_data_node1(train_tags==1,:);
            train_set_cond2_node1 = train_data_node1(train_tags==2,:);
            train_set_cond1_node2 = train_data_node2(train_tags==1,:);
            train_set_cond2_node2 = train_data_node2(train_tags==2,:);
            
            train_set_cond1_node1_mean = mean(train_set_cond1_node1,1);
            train_set_cond1_node2_mean = mean(train_set_cond1_node2,1);
            train_set_cond2_node1_mean = mean(train_set_cond2_node1,1);
            train_set_cond2_node2_mean = mean(train_set_cond2_node2,1);
            
            % run MCPA
            [pred_tag(n), corr_cond1(n), corr_cond2(n)] = mcpa(train_set_cond1_node1,train_set_cond2_node1,train_set_cond1_node2,train_set_cond2_node2,test_vector_node1,test_vector_node2,num_cc);
            
        end
        
        % compute the accuracy and sensitivity index d'
        idx1 = find(data_tags == 1);
        idx2 = find(data_tags == 2);
        tp(i,j) = length(find(pred_tag(idx1)==1)) / length(idx1);
        tp(i,j) = max(tp(i,j),1/length(idx1));
        tp(i,j) = min(tp(i,j),1-1/length(idx1));
        fp(i,j) = length(find(pred_tag(idx2)==1)) / length(idx2);
        fp(i,j) = max(fp(i,j),1/length(idx2));
        fp(i,j) = min(fp(i,j),1-1/length(idx2));
        
        acc(i,j) = length(find(pred_tag == data_tags)) / length(data_tags);
        dp(i,j) = norminv(tp(i,j)) - norminv(fp(i,j));
        
        fprintf('Condition %d vs Condition %d, accuracy = %1.4f\n',i,j,acc(i,j))
        
    end
end

mean_dp = sum(sum(dp))/nchoosek(num_conditions,2);
mean_acc = sum(sum(acc))/nchoosek(num_conditions,2);

fprintf('overall accuracy = %1.4f \n',mean_acc)
fprintf('overall dprime = %1.4f \n',mean_dp)
