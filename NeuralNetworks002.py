import math
import random
import NeuralNetworks

class NeuralLayer:
    def __init__(self, numberOfInputs, numberOfOutputs, activationFunction, weights = None):
        self.Weights = []
        self.NumberOfInputs = numberOfInputs
        self.NumberOfNeurons = numberOfOutputs
        self.ActivationFunction = activationFunction
        if weights == None:
            for i in range(self.NumberOfNeurons):
                self.Weights.append([random.uniform(-1,1) for j in range(self.NumberOfInputs + 1)])
        else:
            for i in range(self.NumberOfNeurons):
                self.Weights.append(weights[i])
    def Calculate(self, inputs):
        tempInputs = inputs.copy()
        tempInputs.insert(0,1.0)
        outputs = []
        for i in range(self.NumberOfNeurons):
            sum = 0
            for j in range(len(self.Weights[i])):
                sum += tempInputs[j] * self.Weights[i][j]
            outputs.append(self.ActivationFunction(sum))
        return outputs
class NeuralNetwork:
    def __init__(self, map, activationFunction, weights = None):
        self.NeuronLayers = []
        self.Map = map
        if weights == None:
            weights = [None for i in range(len(map) - 1)]
        for i in range(len(map) - 1):
            self.NeuronLayers.append(NeuralLayer(map[i],map[i+1],activationFunction,weights[i]))
    def Calculate(self, inputs):
        temp = inputs
        for i in range(len(self.NeuronLayers)):
            temp = self.NeuronLayers[i].Calculate(temp)
        return temp

class NeuralTeacherTanh:
    def __init__(self, neuralNetwork, learningRate):
        self.NeuralNetwork = neuralNetwork
        self.LearningRate = learningRate
        self.NeuronInputs = [[] for i in range(len(self.NeuralNetwork.NeuralLayers))]
        self.NeuronOutputs = [[] for i in range(len(self.NeuralNetwork.NeuralLayers))]
        self.DeltaOfWeights = None
    def CalculateAllDeltaWeights(self, inputs, requiredOutputs):
        self.DeltaOfWeights = [[[0.0 for k in range(self.NeuralNetwork.NeuralLayers[i].NumberOfInputs + 1)] for j in range(self.NeuralNetwork.NeuralLayers[i].NumberOfNeurons)] for i in range(len(self.NeuralNetwork.NeuralLayers))]
        self.CalculateAllNeuronInputsAndOutputs(inputs)
        outputError = []

    def CalculateAllNeuronInputsAndOutputs(self, inputs):
        for i in range(len(self.NeuralNetwork.NeuralLayers)):
            if i == 0:
                self.NeuronInputs[i] = inputs
            else:
                self.NeuronInputs[i] = self.NeuronOutputs[i-1]
            self.NeuronOutputs[i] = self.NeuralNetwork.NeuralLayers[i].Calculate(self.NeuronInputs[i])
    def GetDeltaOfNeuron(self, inputs, weights, output, requiredOutput):
        dWeights = [0.0 for i in range(len(weights))]
        tempInputs = inputs.copy()
        tempInputs.insert(0,1)
        for i in range(len(weights)):
            dWeights[i] += tempInputs[i] * (1 - output) * (requiredOutput - output) * self.LearningRate
        return dWeights
    def CalculateNeuronOutput(self, inputs, weights):
        tempInputs = inputs.copy()
        tempInputs.insert(0,1)
        sum = 0
        for i in range(len(weights)):
            sum += tempInputs[i] * weights[i]
        return math.tanh(sum)
def aFunc(x):
    return math.tanh(x)
nn0 = NeuralNetwork([3,4,2],aFunc,[[[0.5,0.3,-0.3,-0.9],[0.2,-0.3,0.4,-0.9],[0.1,0.3,0.2,-0.6],[0.5,0.3,-0.3,-0.9]],[[0.3,0.2,-0.6,0.2,-0.3],[-0.3,-0.9,0.3,0.2,-0.6]]])
nn = NeuralNetworks.NeuralNetwork(3,[4,2],aFunc,[[0.5,0.3,-0.3,-0.9,0.2,-0.3,0.4,-0.9,0.1,0.3,0.2,-0.6,0.5,0.3,-0.3,-0.9],[0.3,0.2,-0.6,0.2,-0.3,-0.3,-0.9,0.3,0.2,-0.6]])
print(nn0.Calculate([0.2,1,-2]))
print(nn.Calculate([0.2,1,-2]))
inputs = [0.3,-0.5,1]
weights = [1.5,-0.3,0.6,0.7]
error = 0
for i in range(30):
    o = NeuralTeacherTanh.CalculateNeuronOutput(inputs,weights)
    print(o)
    error = 0 - o
    print(error)
    dW = NeuralTeacherTanh.GetDeltaOfNeuron(inputs,weights,error)
    print(dW)
    weights = [weights[j]+dW[j] for j in range(len(weights))]
