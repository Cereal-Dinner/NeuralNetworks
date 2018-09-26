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
class TanhTeachingNeuralLayer(NeuralLayer):
    def __init__(self, sourceNeuralLayer):
        super().__init__(sourceNeuralLayer.NumberOfInputs, sourceNeuralLayer.NumberOfNeurons, sourceNeuralLayer.ActivationFunction, sourceNeuralLayer.Weights)
        self.Inputs = [0.0 for i in range(sourceNeuralLayer.NumberOfInputs)]
        self.Outputs = [0.0 for i in range(sourceNeuralLayer.NumberOfNeurons)]
        self.Errors = [0.0 for i in range(sourceNeuralLayer.NumberOfNeurons)]
        self.DeltaWeights = [[0.0 for j in range(len(sourceNeuralLayer.Weights[i]))] for i in range(len(sourceNeuralLayer.Weights))]
    def Calculate(self, inputs):
        self.Inputs = inputs
        self.Outputs = super().Calculate(inputs)
        return self.Outputs
    def CalculateDeltaWeights(self):
        self.DeltaWeights = [[0.0 for j in range(len(self.Weights[i]))] for i in range(len(self.Weights))]
        for i in range(len(self.Weights)):
            for j in range(len(self.Weights[i])):
                if j == 0:
                    self.DeltaWeights[i][j] = 1 * (1 - self.Outputs[i]) * self.Errors[i]
                else:
                    self.DeltaWeights[i][j] = self.Inputs[j-1] * (1 - self.Outputs[i]) * self.Errors[i]
        return self.DeltaWeights
    def BackPropgateError(self, errors):
        self.Errors = errors
        backLayerError = [0.0 for i in range(self.NumberOfInputs)]
        for i in range(len(self.Weights)):
            sum = 0
            for j in range(len(self.Weights[i])):
                sum += abs(self.Weights[i][j])
            for j in range(1,len(self.Weights[i])):
                backLayerError[j-1] += self.Errors[i] * (self.Weights[i][j]/sum)
        return backLayerError

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
        self.TeachingNeuronLayers = [TanhTeachingNeuralLayer(nL) for nL in self.NeuralNetwork.NeuronLayers]
    def Teach(self, inputs, requiredOutputs):
        numberOfLayers = len(self.TeachingNeuronLayers)
        numberOfOutputs = self.TeachingNeuronLayers[numberOfLayers-1].NumberOfNeurons
        #Feed Forward
        self.TeachingNeuronLayers[0].Calculate(inputs)
        for i in range(1,numberOfLayers):
            self.TeachingNeuronLayers[i].Calculate(self.TeachingNeuronLayers[i-1].Outputs)
        #Calculate all output errors
        tempErrors = [requiredOutputs[i] - self.TeachingNeuronLayers[numberOfLayers-1].Outputs[i] for i in range(numberOfOutputs)]
        for i in range(numberOfLayers-1,0,-1):
            tempErrors = self.TeachingNeuronLayers[i].BackPropgateError(tempErrors)

        for nL in self.TeachingNeuronLayers:
            nL.CalculateDeltaWeights()
        for i in range(numberOfLayers):
            for j in range(self.NeuralNetwork.Map[i+1]):
                for w in range(len(self.NeuralNetwork.NeuronLayers[i].Weights[j])):
                    self.NeuralNetwork.NeuronLayers[i].Weights[j][w] += self.TeachingNeuronLayers[i].DeltaWeights[j][w] * self.LearningRate
        return self.NeuralNetwork
def aFunc(x):
    return math.tanh(x)
nn = NeuralNetwork([784,128,10],aFunc)
teacher = NeuralTeacherTanh(nn,0.1)
