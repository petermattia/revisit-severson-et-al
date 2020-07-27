%% Peter Attia
% Generate voltage arrays
% Windows 10 path conventions

%% Paths
data_path = 'D:\Data_Matlab\Batch_data';
save_path = 'data';

%% Clear directories
rmdir(save_path,'s')
mkdir(save_path)
mkdir([save_path,'\','cycle_lives'])
mkdir([save_path,'\','train'])
mkdir([save_path,'\','test1'])
mkdir([save_path,'\','test2'])

%% Load data and remove bad cells
close all; clc
if ~exist('batch','var') % only load if we haven't loaded yet
    % Batch1
    load([data_path,'\2017-05-12_batchdata_updated_struct_errorcorrect'])
    
    batch1 = batch;
    numBat1 = size(batch1,2);
    
    % Batch2
    load([data_path,'\2017-06-30_batchdata_updated_struct_errorcorrect'])
    
    %remove batteries continuing from batch 1
    batch([8:10,16:17]) = [];
    batch2 = batch;
    numBat2 = size(batch2,2);
    clearvars batch
    
    % Batch8
    load([data_path,'\2018-04-12_batchdata_updated_struct_errorcorrect'])
    batch8 = batch;
    batch8(38) = []; %remove channel 46 upfront
    numBat8 = size(batch8,2);
    endcap8 = zeros(numBat8,1);
    clearvars batch
    for i = 1:numBat8
        endcap8(i) = batch8(i).summary.QDischarge(end);
    end
    rind = find(endcap8 > 0.885);
    batch8(rind) = [];
    
    %remove the noisy Batch 8 batteries
    nind = [3, 40:41];
    batch8(nind) = [];
    numBat8 = size(batch8,2);
    
    batch = [batch1, batch2, batch8];
    numBat = numBat1 + numBat2 + numBat8;
    
    %remove the batteries that do not finish in Batch 1
    batch([9,11,13,14,23]) = [];
    numBat = numBat - 5;
    numBat1 = numBat1 - 5;
    
    clearvars -except batch numBat1 numBat2 numBat8 numBat data_path save_path
end

%% Train, test1, test2 per Severson et al
cells = 1:numBat; % all cells
cells_test1 = [1:2:numBat1+numBat2,84];
cells_train = 1:numBat1+numBat2;
cells_train(cells_test1) = [];
cells_test2 = (numBat1+numBat2+1):numBat;

%% extract the number of cycles to 0.88
cycle_lives = zeros(numBat,1);
for i = 1:numBat
    if batch(i).summary.QDischarge(end) < 0.88
        cycle_lives(i) = find(batch(i).summary.QDischarge < 0.88,1);
    else
        cycle_lives(i) = size(batch(i).cycles,2) + 1;
    end
    
end

cycle_lives(1:5) = [1852; 2160; 2237; 1434; 1709]; %manual correction

%% Save
csvwrite([save_path,'\cycle_lives\train_cycle_lives.csv'],cycle_lives(cells_train));
csvwrite([save_path,'\cycle_lives\test1_cycle_lives.csv'],cycle_lives(cells_test1));
csvwrite([save_path,'\cycle_lives\test2_cycle_lives.csv'],cycle_lives(cells_test2));

%% Generate voltage arrays
% Save one csv per cell with each 1000x100 voltage array

% Variables
nV = length(batch(1).cycles(2).Qdlin); % Number of voltage sample points
nC = 100; % First 100 cycles

% Loop through each battery
for c = cells
    % Preinitialize
    voltage_array = zeros(nV,nC-1); % exclude first cycle
        
    for cycle_index = 2:nC
        vector = batch(c).cycles(cycle_index).Qdlin;
        voltage_array(:,cycle_index-1) = vector;
    end
    
    % Identify train/test1/test2 and save
    if ismember(c,cells_train)
        folder = 'train';
        idx = num2str(find(c==cells_train));
    elseif ismember(c,cells_test1)
        folder = 'test1';
        idx = num2str(find(c==cells_test1));
    elseif ismember(c,cells_test2)
        folder = 'test2';
        idx = num2str(find(c==cells_test2));
    else
        disp('Error: cell not categorized')
    end
    
    csvwrite([save_path,'\',folder,'\cell',idx,'.csv'],voltage_array);
end