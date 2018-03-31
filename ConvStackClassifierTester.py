import ConvStackClassifier
import PIL
import torchvision

print("ConvStackClassifierTester.py")

neuralNet = ConvStackClassifier.NeuralNet()
neuralNet.Load("/home/sebastien/PycharmProjects/mnist/ConvStack_1_3_32_7_2_32_7_2_32_7_2_10_28_0.5_0.0366")

# Load an image
imageFilepath = "/home/sebastien/MachineLearning/BenchmarkDatasets/mnist/img/5_0.png"
testImg = PIL.Image.open(imageFilepath).convert('L') # Open in RGB, then convert to grayscale
imgTensorConverter = torchvision.transforms.ToTensor()
testTensor = imgTensorConverter(testImg)
print ("testTensor = {}".format(testTensor))