import torch
import torch.nn
import torch.nn.functional
import torchvision
import math
from collections import OrderedDict
import os
import PIL.Image


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.residual1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        )
        self.coef1_1 = torch.Tensor([1.0])
        self.linear1 = torch.nn.Linear(57 * 57 * 32, 2)

    def forward(self, inputs):
        residual1 = self.residual1(inputs)
        output1 = self.coef1_1 * inputs + residual1
        vector = output1.view(-1, 57 * 57 * 32)
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