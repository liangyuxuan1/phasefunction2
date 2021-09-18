% 2021-09-23, V4 dataset
% Simplify the problem by fixing the ua and us
% Use the ua an us of typical tissues and vary only g

% 2021-09-05,V5 dataset
% using optical parameters in Ren Shenghan PlosOne paper
% add tissue type in datalist file

% 2021-09-12, re-run the simulation using new large phantom

clear all;
close all;
clc;

phantomFile = 'newProject_CW_500.mse';

% absorption coefficient, [0.01, 10] mm^-1
% scattering coefficient, [0.1, 100] mm^-1

% parameter at 600 nm, P9, Ren Shenghan, Plos One, 2013
tissueName = {'Surface', 'Lung', 'Kidney', 'Heart', 'Stomach', 'Liver', 'Tumor'};
tissuePara = [0.004, 20.13, 0.94; 
    0.196, 36.23, 0.94;
    0.066, 16.09, 0.86;
    0.059, 6.42,  0.85;
    0.011, 17.96, 0.92;
    0.035, 6.78,  0.9;
    0.55,  29.5,  0.9];           

n  = 1.37;               % refractive index, no need to vary for single layer slab
trainNum = 1000;         % training number of runs (images) for each set of parameters
testNum = 10;            % test number of runs (images) for each set of parameters

dataPath = 'rawDataCW_v5';   % path to store the raw simulation results
if ~exist(dataPath,'dir')
    mkdir(dataPath);
end

varNames = {'Image', 'ua', 'us', 'g', 'Tissue', 'Events'};
varTypes = {'string', 'double', 'double', 'double', 'string', 'string'};

totalTrainNum   = length(tissueName)*trainNum;
trainTableCW = table('Size', [totalTrainNum,length(varNames)], 'VariableTypes',varTypes,'VariableNames',varNames);
% train dataset
for p = 1:length(tissueName)
    if ~exist(fullfile(dataPath, tissueName{p}))
        mkdir(fullfile(dataPath, tissueName{p}));
    end
    
    for i=1:trainNum
        ua = tissuePara(p,1);
        us = tissuePara(p,2);
        g  = tissuePara(p,3);
        
        dataFileName = sprintf("%04d", i);
        fullDataFileName = fullfile(dataPath, tissueName{p}, dataFileName);
   
        parameters = ['MOSE\moseVCTest.exe', phantomFile, fullDataFileName, num2str(ua), num2str(us), num2str(g), num2str(n)];
        cmdLine = strjoin(parameters, ' ');
        
        % system(cmdLine);    % call MOSE
        
        dataFileName = strcat(dataFileName, '.T.CW');
        samplePath = fullfile(tissueName{p}, dataFileName);
        row = {samplePath, ua, us, g, tissueName{p}, 'Train'};
        trainTableCW((p-1)*trainNum+i, :) = row;
    end 
end % of train
writetable(trainTableCW, [dataPath, filesep,  'TrainDataCW_v5.csv']);

