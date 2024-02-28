% Afroza 28 feb 2024
% Create Neural Network in MATLAB

%Close all open figures
 close all
%Clear the workspace
 clear
%Clear the command window
 clc
%Load the dataset
 load crab_dataset;

sizeOfCrabInput = size(crabInputs); 
sizeOfCrabTargets = size(crabTargets); 

numberOfTargetVariables = sizeOfCrabTargets(1);
numberOfInstances = sizeOfCrabTargets(2); 

%Create a two-layer feed-forward network. The network has one hidden layer % with 10 neurons. The training algorithm 'trainlm' is selected.
net = feedforwardnet(10, 'trainlm');

%Configure the network inputs and outputs to best match input and target data
net = configure(net, crabInputs, crabTargets);

%Train the neural network (net) with inputs (crabInputs) and target (crabTargets)
[net,tr] = train(net,crabInputs,crabTargets);

testInput = crabInputs(:,tr.testInd); 
testTarget = crabTargets(:,tr.testInd); 
testY = net(testInput);


% Now, plot confusion matrix
plotconfusion(testTarget, testY);
