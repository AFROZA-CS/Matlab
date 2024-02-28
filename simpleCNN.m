% Afroza 28 feb 2024
% Create Neural Network in MATLAB

%Close all open figures
 close all
%Clear the workspace
 clear
%Clear the command window
 clc

%An imageDatastore automatically labels the images based on folder names and stores the data as an ImageDatastore object
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', 'nndemos', 'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource','foldernames');

%Display some of the images in the datastore
figure;
perm = randperm(10000,20); for i = 1:20
subplot(4,5,i); imshow(imds.Files{perm(i)});
end

%Calculate the number of images in each category
labelCount = countEachLabel(imds)

%Specify image sizes
img = readimage(imds,1); size(img)

%Specify Training and Validation Sets
numTrainFiles =  750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%Define the convolutional neural network architecture
layers = [
imageInputLayer([28 28 1])
convolution2dLayer(3,8,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,32,'Padding','same')
batchNormalizationLayer
reluLayer
fullyConnectedLayer(10)
softmaxLayer
classificationLayer];

%Specify Training Options

options = trainingOptions('sgdm', ...
'InitialLearnRate',0.01, ...
'MaxEpochs',4, ...
'Shuffle','every-epoch', ...
'ValidationData',imdsValidation, ...
'ValidationFrequency',30, ...
'Verbose',false, ...
'Plots','training-progress');

%Train Network Using Training Data

net = trainNetwork(imdsTrain,layers,options);

%Classify Validation Images and Compute Accuracy

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);



% Display the accuracy in the command window
disp(['Accuracy: ', num2str(accuracy)]);