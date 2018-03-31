import ConvStackClassifier
import PIL
import torchvision
import torch

print("ConvStackClassifierTester.py")
torch.manual_seed(0)

requiredImageSize = (28, 28)

# ------------------------------ Utilities ---------------------------------------
def MostProbableClass(outputVector):
    mostProbableClass = -1;
    highestOutput = -float('inf')
    for classNdx in range ( list(outputVector.size()) [1]):
        if outputVector[0, classNdx].data[0] > highestOutput:
            highestOutput = outputVector[0, classNdx].data[0]
            mostProbableClass = classNdx
    return mostProbableClass, highestOutput

# --------------------------------------------------------------------------------

neuralNet = ConvStackClassifier.NeuralNet()
neuralNet.Load("/home/sebastien/PycharmProjects/mnist/ConvStack_1_3_32_7_2_32_7_2_32_7_2_10_28_0.5_0.0366")

# Load an image
imageFilepath = "/home/sebastien/MachineLearning/BenchmarkDatasets/mnist/img/testDigit.png"
testImg = PIL.Image.open(imageFilepath).convert('L') # Open in RGB, then convert to grayscale
# Resize the image to the required size
testImg = testImg.resize(requiredImageSize, PIL.Image.BILINEAR)

imgTensorConverter = torchvision.transforms.ToTensor()
testTensor = imgTensorConverter(testImg)
#print ("testTensor = {}".format(testTensor))

# Make it a batch
testBatchTensor = testTensor.unsqueeze(0) # [1 x 28 x 28] -> [1 x 1 x 28 x 28]
# Convert to a Variable
testBatchVar = torch.autograd.Variable(testBatchTensor, requires_grad=False)

outputTensor = neuralNet(testBatchVar)
print ("outputTensor = {}".format(outputTensor))

# Convert to probabilities
outputProbabilities = torch.exp(outputTensor)
print ("outputProbabilities = {}".format(outputProbabilities))
mostProbableClass, confidence = MostProbableClass(outputProbabilities)
print("mostProbableClass = {}; \nconfidence = {}".format(mostProbableClass, confidence))
"""sum = 0.
print("outputProbabilities.size() = {}".format(outputProbabilities.size()))
print("list(outputProbabilities.size()) = {}".format(list(outputProbabilities.size()) ))
print("(list(outputProbabilities.size() ) [1] ) = {}".format((list(outputProbabilities.size() ) [1] )))
for elementNdx in range( (list(outputProbabilities.size() ) [1] ) ): # list(*) converts torch.Size to list(int): [1, 10]
    print("elementNdx = {}".format(elementNdx))
    sum += outputProbabilities[0, elementNdx] # outputProbabilities has size [1, N]
print ("sum = {}".format(sum))
"""