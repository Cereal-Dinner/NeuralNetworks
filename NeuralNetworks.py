import random

class NeuronLayer:
    def __init__(self, numberOfInputs, numberOfNeurons, activationFunction, weights = None):
        self.NumberOfNeurons = numberOfNeurons
        self.NumberOfInputs = numberOfInputs
        self.ActivationFunction = activationFunction
        self.Weights = []
        if weights == None:
            for i in range((self.NumberOfInputs + 1) * self.NumberOfNeurons):
                self.Weights.append(random.uniform(-1.0, 1.0))
        else:
            for i in range((self.NumberOfInputs + 1) * self.NumberOfNeurons):
                self.Weights.append(weights[i])

    def Calculate(self, inputs):
        tempInputs = inputs.copy()
        tempInputs.insert(0,1.0)
        outputs = []
        for i in range(self.NumberOfNeurons):
            sum = 0
            for j in range(len(tempInputs)):
                sum += self.Weights[(i * len(tempInputs)) + j] * tempInputs[j]
            outputs.append(self.ActivationFunction(sum))
            
        return outputs
    
    def ToString(self):
        return str(self.Weights)

class NeuralNetwork:
    def __init__(self, numberOfInputs, numberOfNeurons, activationFunction, weights = None):
        self.NeuronLayers = []
        if weights == None:
            weights = [None for i in range(len(numberOfNeurons))]
        self.NeuronLayers.append(NeuronLayer(numberOfInputs, numberOfNeurons[0], activationFunction, weights[0]))
        for i in range(1,len(numberOfNeurons)):
            self.NeuronLayers.append(NeuronLayer(numberOfNeurons[i-1],numberOfNeurons[i],activationFunction,weights[i]))
    def Calculate(self,inputs):
        tempInputs = inputs
        tempOut = None
        for i in range(len(self.NeuronLayers)):
            tempOut = self.NeuronLayers[i].Calculate(tempInputs)
            tempInputs = tempOut
        return tempOut
    def ToString(self):
        s = ''
        for nL in self.NeuronLayers:
            s+= nL.ToString() + ' '
        return s
