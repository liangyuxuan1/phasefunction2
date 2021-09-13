% 2021-09-23, V4 dataset
% Simplify the problem by fixing the ua and us
% Use the ua an us of typical tissues and vary only g

% 2021-09-05,V5 dataset
% add small distrube to g, generate image variance
% put train,val,test into same named dir
% using optical parameters in Ren Shenghan PlosOne paper
% add tissue type in datalist file

clear all;
close all;
clc;

phantomFile = 'newProject_CW_500.mse';

% parameter range, refer to: Brett H. Hokr1 and Joel N. Bixler2, Machine
% learning estimation of tissue optical properties, Scientific Reports,
% 11:6561, 2021
% absorption coefficient, [0.01, 10] mm^-1
% scattering coefficient, [0.1, 100] mm^-1
ua_min = 0.01;
ua_max = 10;
us_min = 0.1;
us_max = 100;

ua = 10.^linspace(log10(ua_min), log10(ua_max), 10);
us = 10.^linspace(log10(us_min), log10(us_max), 10);
g  = 0.55:0.1:0.95;       

n  = 1.37;               % refractive index, no need to vary for single layer slab
trainNum = 20;           % training number of runs (images) for each set of parameters
testNum  = 20;           % test number of runs (images) for each set of parameters

dataPath = 'rawDataCW_v6';   % path to store the raw simulation results
if ~exist(dataPath,'dir')
    mkdir(dataPath);
end

varNames = {'Image', 'ua', 'us', 'g', 'Tissue', 'Events'};
varTypes = {'string', 'double', 'double', 'double', 'string', 'string'};

totalTrainNum = length(ua)*length(us)*length(g)*trainNum;
trainTableCW = table('Size', [totalTrainNum,length(varNames)], 'VariableTypes',varTypes,'VariableNames',varNames);
% train dataset
for ia = 1:length(ua)
    for is = 1:length(us)
        for ig = 1:length(g)
            for i=1:trainNum
                dataFileName = sprintf("a%02d_s%02d_g%02d_%03d",ia, is, ig, i);  % data file name;
                fullDataFileName = fullfile(dataPath, dataFileName);

                parameters = ['MOSE\moseVCTest.exe', phantomFile, fullDataFileName, num2str(ua(ia)), num2str(us(is)), num2str(g(ig)), num2str(n)];
                cmdLine = strjoin(parameters, ' ');

                % rum MOSE
                % system(cmdLine); 

                dataFileName = strcat(dataFileName, '.T.CW');
                tissueName = (ia-1)*length(us)*length(g) + (is-1)*length(g) + ig;
                row = {dataFileName, ua(ia), us(is), g(ig), num2str(tissueName), 'Train'};
                idx = (ia-1)*length(us)*length(g)*trainNum + (is-1)*length(g)*trainNum + (ig-1)*trainNum + i;
                trainTableCW(idx, :) = row;
            end % i
        end % ig
    end % is
end % ia
writetable(trainTableCW, [dataPath, filesep,  'TrainDataCW_v6.csv']);
