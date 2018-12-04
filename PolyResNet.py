import torch
import torch.nn
import torch.nn.functional
import torchvision
import math
from collections import OrderedDict
import os
import PIL.Image


def ResidualBlock(numberOfChannels, kernelSize, dropoutRatio):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=numberOfChannels, out_channels=numberOfChannels, kernel_size=kernelSize, padding=int(kernelSize/2)),
        #torch.nn.BatchNorm2d(num_features=numberOfChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(in_channels=numberOfChannels, out_channels=numberOfChannels, kernel_size=kernelSize, padding=int(kernelSize/2)),
        #torch.nn.BatchNorm2d(num_features=numberOfChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.Dropout(p=dropoutRatio)
    )


class NeuralNet(torch.nn.Module):
    def __init__(self,
                 inputNumberOfChannels=1,
                 hiddenBlocksNumberOfChannels=32,
                 initialConvolutionKernelSize=3,
                 numberOfResidualBlocks=3,
                 residualBlocksKernelSize=5,
                 residualBlocksDropoutRatio=0.5,
                 polynomialsDegree=2,
                 imageSize=(57, 57), # (W, H)
                 numberOfClasses=2
                 ):
        super(NeuralNet, self).__init__()
        self.initialConv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=inputNumberOfChannels, out_channels=hiddenBlocksNumberOfChannels,
                            kernel_size=initialConvolutionKernelSize, padding=int(initialConvolutionKernelSize/2)),
            torch.nn.BatchNorm2d(num_features=hiddenBlocksNumberOfChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        residualLayersDict = OrderedDict()
        #layersPolynomialCoefficientsDict = OrderedDict()
        self.layersPolynomialCoefficientsDict = torch.nn.ParameterDict()

        for residualLayerNdx in range(numberOfResidualBlocks):
            layerName = 'residualBlock' + str(residualLayerNdx)
            residualLayersDict[layerName] = ResidualBlock(hiddenBlocksNumberOfChannels, residualBlocksKernelSize,
                                                          residualBlocksDropoutRatio)

            # Polynomial coefficients
            #coefficientsList = torch.nn.ParameterList()
            for exponent in range(1, polynomialsDegree + 1):
                coefficientName = 'coef_' + str(residualLayerNdx) + '_' + str(exponent)
                self.layersPolynomialCoefficientsDict[coefficientName] = torch.nn.Parameter(torch.zeros([1]))
                #coefficientsList.append(torch.nn.Parameter(torch.zeros([1])))
            #self.layersPolynomialCoefficientsDict[layerName] = coefficientsList
        self.residualLayers = torch.nn.Sequential(residualLayersDict)

        self.numberOfResidualBlocks = numberOfResidualBlocks

        """self.residual1 = ResidualBlock(32, 5)
        self.coef1_1 = torch.nn.Parameter(torch.zeros([1]))
        self.coef1_2 = torch.nn.Parameter(torch.zeros([1]))
        self.residual2 = ResidualBlock(32, 5)
        self.coef2_1 = torch.nn.Parameter(torch.zeros([1]))
        self.coef2_2 = torch.nn.Parameter(torch.zeros([1]))
        self.residual3 = ResidualBlock(32, 5)
        self.coef3_1 = torch.nn.Parameter(torch.zeros([1]))
        self.coef3_2 = torch.nn.Parameter(torch.zeros([1]))
        """

        #self.dropout = torch.nn.Dropout2d(p=0.5)
        self.linear1 = torch.nn.Linear(imageSize[0] * imageSize[1] * hiddenBlocksNumberOfChannels, numberOfClasses)

        self.imageSize = imageSize
        self.numberOfClasses = numberOfClasses
        self.hiddenBlocksNumberOfChannels = hiddenBlocksNumberOfChannels
        self.polynomialsDegree = polynomialsDegree

    def forward(self, inputs):
        hiddenState = self.initialConv(inputs)

        for residualBlockNdx in range(self.numberOfResidualBlocks):
            layerName = 'residualBlock' + str(residualBlockNdx)
            initialState = hiddenState
            residual = self.residualLayers[residualBlockNdx](initialState)
            hiddenState = residual
            for exponent in range(1, self.polynomialsDegree + 1):
                coefficientName = 'coef_' + str(residualBlockNdx) + '_' + str(exponent)
                #print ("PolyResNet.py forward(): self.layersPolynomialCoefficientsDict[layerName][polynomialNdx] : {}".format(self.layersPolynomialCoefficientsDict[layerName][polynomialNdx] ))
                hiddenState += self.layersPolynomialCoefficientsDict[coefficientName] * torch.pow(initialState, exponent)

        """residual1 = self.residual1(initialConv)
        output1 = self.coef1_1 * initialConv + self.coef1_2 * torch.pow(initialConv, 2) + residual1

        residual2 = self.residual2(output1)
        output2 = self.coef2_1 * output1 + self.coef2_2 * torch.pow(output1, 2) + residual2

        residual3 = self.residual3(output2)
        output3 = self.coef3_1 * output2 + self.coef3_2 * torch.pow(output2, 2) + residual3
        """

        #droppedOutput2 = self.dropout(output2)
        #vector = output3.view(-1, 57 * 57 * 32)
        vector = hiddenState.view(-1, self.imageSize[0] * self.imageSize[1] * self.hiddenBlocksNumberOfChannels)

        outputLin = self.linear1(vector)
        return torch.nn.functional.log_softmax(outputLin, dim=1)

    def Save(self, directory, filenameSuffix):
        filepath = os.path.join(directory, 'PolyResNet' + '_' + filenameSuffix)
        torch.save(self.state_dict(), filepath)

    def Load(self, filepath, useCuda=True):
        self.__init__()
        if useCuda and torch.cuda.is_available():
            self.load_state_dict(torch.load(filepath))
        else:
            self.load_state_dict(torch.load(filepath, map_location=lambda storage, location: storage))


def main():
    print ("PolyResNet.py main()")
    neuralNet = NeuralNet()
    inputImg = PIL.Image.open('/home/segilber/Documents/OPOS/Datasets/LFP_dataset_2018-11-21/test/42179_X1.84_Y4.47.png').convert('L')
    #inputImg = PIL.Image.open('/home/segilber/Pictures/chestnut-horse-autumn_1000.jpg')
    #print ("main() inputImg.size = {}".format(inputImg.size)) # (W, H)
    imageToTensorConverter = torchvision.transforms.ToTensor()
    inputArr = imageToTensorConverter(inputImg)
    print ("main(): inputArr.shape = {}".format(inputArr.shape))
    minibatchTensor = torch.zeros( [1, inputArr.shape[0], inputArr.shape[1], inputArr.shape[2] ])
    print ("main(): minibatchTensor.shape = {}".format(minibatchTensor.shape))
    minibatchTensor[0] = inputArr

    output = neuralNet(minibatchTensor)
    print ("main(): output = {}".format(output))
    print("main(): output.shape = {}".format(output.shape))

if __name__ == '__main__':
    main()