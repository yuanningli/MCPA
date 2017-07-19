function eq_pc = find_numpc(latent, thres)
% determin the number of PCs that accounts for a specific amount of
% variance

explained = latent ./ sum(latent);
eq_pc = 0;
cummVar = 0;
while cummVar < thres
    eq_pc = eq_pc + 1;
    cummVar = cummVar + explained(eq_pc);
end
