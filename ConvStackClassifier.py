import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
import os


def ExtractStructureFromFilename(filename):
    tokens = os.path.basename(filename).split('_') # Remove the directory path
    #print ("ExtractStructureFromFilename(): tokens = {}".format(tokens))
    if not tokens[0] == 'ConvStack':
        raise RuntimeError("ConvStackClassisifer.ExtractStructureFromFilename(): The filename '{}' doesn't start with 'ConvStack_'".format(filename))
    numberOfInputChannels = int(tokens[1])
    numberOfConvolutionBlocks = int(tokens[2])
    convolutionTrios = []
    for convolutionLayerNdx in range(numberOfConvolutionBlocks):
        numberOfConvolutions = int (tokens[3 + 3 * convolutionLayerNdx])
        kernelSize = int (tokens[3 + 3 * convolutionLayerNdx + 1])
        pooling = int (tokens[3 + 3 * convolutionLayerNdx + 2])
        convolutionTrios.append((numberOfConvolutions, kernelSize, pooling))
    numberOfClasses = int(tokens[3 + 3 * numberOfConvolutionBlocks])
    imageSize = int(tokens[3 + 3 * numberOfConvolutionBlocks + 1])
    dropoutRatio = float(tokens[3 + 3 * numberOfConvolutionBlocks + 2])
    return (numberOfInputChannels, numberOfConvolutionBlocks, convolutionTrios, numberOfClasses, imageSize, dropoutRatio)


class NeuralNet(nn.Module):
    def __init__(self,
                 numberOfConvolutions_KernelSize_Pooling_List=[(32, 7, 2)], numberOfInputChannels=1,
                 classesNbr=2, imageSize=32, dropoutRatio=0.5,
                 structure=None #'ConvStack_1_2_32_7_2_32_7_2_2_64_0.5'
                ):
        """print("NeuralNet.__init__({}, {}, {}, {}, {})__".format(numberOfConvolutions_KernelSize_Pooling_List, numberOfInputChannels,
                 classesNbr, imageSize, dropoutRatio))
        """
        super(NeuralNet, self).__init__()
        if structure is not None:
            (numberOfInputChannels, numberOfConvolutionBlocks, numberOfConvolutions_KernelSize_Pooling_List,
             classesNbr, imageSize, dropoutRatio) = ExtractStructureFromFilename(structure)

        if len(numberOfConvolutions_KernelSize_Pooling_List) < 1:
            raise RuntimeError("ConvStackClassifier.NeuralNet.__init__(): The number of convolution layers is 0")
        self.lastLayerImageSize = imageSize
        layersDict = OrderedDict()

        layersDict['conv0'] = nn.Sequential(
            nn.Conv2d(numberOfInputChannels, numberOfConvolutions_KernelSize_Pooling_List[0][0],
                      numberOfConvolutions_KernelSize_Pooling_List[0][1],
                      padding=(int) (numberOfConvolutions_KernelSize_Pooling_List[0][1]/2) ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(numberOfConvolutions_KernelSize_Pooling_List[0][2]))
        self.lastLayerImageSize = (int) (self.lastLayerImageSize/ numberOfConvolutions_KernelSize_Pooling_List[0][2])
        self.structure = 'ConvStack_{}_{}_{}_{}_{}'.format(numberOfInputChannels,
                                                           len(numberOfConvolutions_KernelSize_Pooling_List),
                                                        numberOfConvolutions_KernelSize_Pooling_List[0][0],
                                                        numberOfConvolutions_KernelSize_Pooling_List[0][1],
                                                        numberOfConvolutions_KernelSize_Pooling_List[0][2])
        for layerNdx in range(1, len(numberOfConvolutions_KernelSize_Pooling_List)):
            layersDict['conv{}'.format(layerNdx)] = nn.Sequential(
                nn.Conv2d(numberOfConvolutions_KernelSize_Pooling_List[layerNdx - 1][0], numberOfConvolutions_KernelSize_Pooling_List[layerNdx][0],
                          numberOfConvolutions_KernelSize_Pooling_List[layerNdx][1],
                          padding=(int)(numberOfConvolutions_KernelSize_Pooling_List[layerNdx][1] / 2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(numberOfConvolutions_KernelSize_Pooling_List[layerNdx][2]))
            self.lastLayerImageSize = (int)(self.lastLayerImageSize / numberOfConvolutions_KernelSize_Pooling_List[layerNdx][2])
            self.structure += '_{}_{}_{}'.format(numberOfConvolutions_KernelSize_Pooling_List[layerNdx][0],
                                                        numberOfConvolutions_KernelSize_Pooling_List[layerNdx][1],
                                                        numberOfConvolutions_KernelSize_Pooling_List[layerNdx][2])

        self.convLayers = nn.Sequential(layersDict)
        self.lastLayerNumberOfChannels = numberOfConvolutions_KernelSize_Pooling_List[-1][0]
        self.numberOfConvolutionLayers = len(layersDict)
        self.linear1 = nn.Linear(self.lastLayerImageSize * self.lastLayerImageSize *
                                 self.lastLayerNumberOfChannels, classesNbr)
        self.dropout = nn.Dropout2d(p=dropoutRatio)
        self.numberOfConvolutionLayers = len(numberOfConvolutions_KernelSize_Pooling_List)
        self.structure += '_{}_{}_{}'.format(classesNbr, imageSize, dropoutRatio)
        self.inputImageSize = (imageSize, imageSize)


    def forward(self, inputs):
        activation = self.convLayers[0](inputs)
        for layerNdx in range(1, self.numberOfConvolutionLayers):
            activation = self.convLayers[layerNdx](activation)

        vector = activation.view(-1, self.lastLayerImageSize * self.lastLayerImageSize * self.lastLayerNumberOfChannels)
        drop = self.dropout(vector)
        outputLin = self.linear1(drop)
        #print ("forward(): outputLin.shape = {}".format(outputLin.shape))
        return F.log_softmax(outputLin, dim=1)

    def Save(self, directory, filenameSuffix):
        filepath = os.path.join(directory, self.structure + '_' + filenameSuffix)
        torch.save(self.state_dict(), filepath)

    def Load(self, filepath, useCuda=True):
        structureList = ExtractStructureFromFilename(filepath)
        self.__init__(structureList[2], structureList[0], structureList[3], structureList[4], structureList[5])
        if useCuda and torch.cuda.is_available():
            self.load_state_dict(torch.load(filepath))
        else:
            self.load_state_dict(torch.load(filepath, map_location=lambda storage, location: storage))

    def InputImageSize(self):
        return self.inputImageSize

