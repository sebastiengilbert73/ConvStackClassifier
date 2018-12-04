import torch
import torch.nn
import torch.nn.functional
import torchvision
import math
from collections import OrderedDict
import os
import PIL.Image


def ResidualBlock(numberOfChannels, kernelSize):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=numberOfChannels, out_channels=numberOfChannels, kernel_size=kernelSize, padding=int(kernelSize/2)),
        #torch.nn.BatchNorm2d(num_features=numberOfChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(in_channels=numberOfChannels, out_channels=numberOfChannels, kernel_size=kernelSize, padding=int(kernelSize/2)),
        #torch.nn.BatchNorm2d(num_features=numberOfChannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.Dropout(p=0.5)
    )


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.initialConv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        """self.residual1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        """
        self.residual1 = ResidualBlock(32, 5)
        self.coef1_1 = torch.nn.Parameter(torch.zeros([1]))
        self.coef1_2 = torch.nn.Parameter(torch.zeros([1]))
        self.residual2 = ResidualBlock(32, 5)
        self.coef2_1 = torch.nn.Parameter(torch.zeros([1]))
        self.coef2_2 = torch.nn.Parameter(torch.zeros([1]))
        self.residual3 = ResidualBlock(32, 5)
        self.coef3_1 = torch.nn.Parameter(torch.zeros([1]))
        self.coef3_2 = torch.nn.Parameter(torch.zeros([1]))

        #self.dropout = torch.nn.Dropout2d(p=0.5)
        self.linear1 = torch.nn.Linear(57 * 57 * 32, 2)

    def forward(self, inputs):
        initialConv = self.initialConv(inputs)
        residual1 = self.residual1(initialConv)
        output1 = self.coef1_1 * initialConv + self.coef1_2 * torch.pow(initialConv, 2) + residual1

        residual2 = self.residual2(output1)
        output2 = self.coef2_1 * output1 + self.coef2_2 * torch.pow(output1, 2) + residual2

        residual3 = self.residual3(output2)
        output3 = self.coef3_1 * output2 + self.coef3_2 * torch.pow(output2, 2) + residual3


        #droppedOutput2 = self.dropout(output2)
        vector = output3.view(-1, 57 * 57 * 32)

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