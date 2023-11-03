import random
import numpy as np

class MEANSQUARED():

    @staticmethod
    def derivative(output, target):
        return output - target

class LINEAR():

    @staticmethod
    def output(x):
        return x

    @staticmethod
    def derivative(x):
        return 1

class TANH():

    @staticmethod
    def output(x):
        return np.tanh(x) 

    @staticmethod
    def derivative(x):
        output = self.output(x)

        return 1 - output*output

class SIGMOID():

    @staticmethod
    def output(x):
        return 1 / (1 + np.exp(-x)) 

    @staticmethod
    def derivative(x):
        output = self.output(x)

        return output * (1 - output)

class RELU():

    @staticmethod
    def output(x):
        return max(0, x) 

    @staticmethod
    def derivative(x):
        output = self.output(x)
        
        if output:
            return 1
        return 0

class Perceptron():

    def __init__(self, bias, learningRate, activation):
        self.weights = []
        self.inputs = [bias] 
        self.learningRate = learningRate
        self.activation = activation

    def regression(self):
        inputs = self.inputs
        weights = self.weights

        return np.dot(np.array(inputs), np.array(weights).T)

    def delta(self, error):
        return self.activation.derivative(self.regression()) * error

    def input(self, inputs):
        self.inputs = inputs + [self.inputs[-1]]

        if len(self.weights):
            return

        self.weights = [random.random()]*len(self.inputs)

    def output(self):
        return self.activation.output(self.regression())

    def backpropagate(self, delta):
        for index in range(len(self.weights)):
            delta = self.learningRate * self.inputs[index] * delta
            self.weights[index] = self.weights[index] - delta 

class Layer():
    learningRate = float(0)
 
    def __init__(self, units, activation, bias = random.random()):
        self.perceptrons = [Perceptron(bias, self.learningRate, activation)]*units

    def input(self, inputs):
        for perceptron in self.perceptrons:
            perceptron.input(inputs)

    def output(self):
        outputs = []

        for perceptron in self.perceptrons:
            outputs.append(perceptron.output())

        return outputs

    def backpropagate(self, errors): 
        weights = []
        deltas = []

        for perceptron, error in zip(self.perceptrons, errors):
            deltas.append(perceptron.delta(error))
            weights.append(perceptron.weights)
            perceptron.backpropagate(deltas[-1])

        return np.dot(deltas, np.vstack(weights))

class Model():

    def __init__(self, learningRate, cost):
        Layer.learningRate = learningRate
        self.layers = []
        self.cost = cost

    def add(self, layer):
        self.layers.append(layer)

    def errors(self, outputs, targets):
        errors = []

        for output, target in zip(outputs, targets):
            errors.append(self.cost.derivative(output, target))

        return errors

    def input(self, inputs, targets):
        errors = self.errors(self.output(inputs), targets)

        for layer in reversed(self.layers):
            errors = layer.backpropagate(errors)

    def output(self, inputs):
        for layer in self.layers:
            layer.input(inputs)
            inputs = layer.output()

        return inputs
  
# CREATING THE MODEL
model = Model(learningRate=0.012, cost=MEANSQUARED)

#ADDING LAYERS TO THE MODEL
model.add(Layer(units=1, activation=LINEAR))
model.add(Layer(units=2, activation=LINEAR))
model.add(Layer(units=2, activation=LINEAR))
model.add(Layer(units=1, activation=LINEAR))

# TRAINING DATA SET
x = [[1], [2], [3], [4]]
y = [[1], [3], [5], [7]]

# TRAIN THE NEAURAL NETWORK MODEL
for epoch in range(20):
    for index in range(len(x)):
        model.input(x[index], y[index])

# PRINT THE OUTPUT
print(model.output([5]))
