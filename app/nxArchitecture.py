import numpy as np
from matplotlib import pyplot
from math import sin, pi
import matplotlib.pyplot as plt

NORM_FONT = ('Helvetica', 10)

class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=True, color='#29526D', zorder=10)
        pyplot.gca().add_patch(circle)

class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.horizontal_distance_between_layers = 6.5
        self.vertical_distance_between_neurons = 6
        self.neuron_radius = 1.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.x = self.__calculate_layer_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        y = self.__calculate_margin_so_layer_is_top_alignment(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(self.x, y)
            neurons.append(neuron)
            y += self.vertical_distance_between_neurons
        return neurons

    def __calculate_margin_so_layer_is_top_alignment(self, number_of_neurons):
##        return (self.number_of_neurons_in_widest_layer - number_of_neurons) * self.vertical_distance_between_neurons
        return self.vertical_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_position(self):
        if self.previous_layer:
            return self.previous_layer.x + self.horizontal_distance_between_layers + 5
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        x1, y1 = neuron1.x, neuron1.y
        x2, y2 = neuron2.x, neuron2.y
        A = abs(y1 - y2)/2
        if y1 > y2:
            x_adj = np.linspace(x1, x2, 100)
            a = -pi/(x2-x1)
            b = pi*(0.5 + x1/(x2-x1))
            y_adj = A*np.sin(a*x_adj + b) + y1 - A
        elif y1 < y2:
            x_adj = np.linspace(x1, x2, 100)
            a = pi/(x2-x1)
            b = -pi*(0.5 + x1/(x2-x1))
            y_adj = A*np.sin(a*x_adj + b) + y2 - A
        else:
            x_adj = (x1, x2)
            y_adj = (y1, y1)
        line = pyplot.Line2D(x_adj, y_adj, zorder=5, color='red')
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw( self.neuron_radius )
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
        # write Text
        y_text = self.number_of_neurons_in_widest_layer * self.vertical_distance_between_neurons
        if layerType == 0:
            pyplot.text(self.x-1, y_text, 'I-L', fontsize = 10)
        elif layerType == -1:
            pyplot.text(self.x-1.5, y_text, 'O-L', fontsize = 10)
        else:
            pyplot.text(self.x-1.5, y_text, 'H-L'+str(layerType), fontsize = 10)

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def nn_draw(self):
##        pyplot.figure(figsize=(9, 5))
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw(i)
##        pyplot.axis('scaled')
##        pyplot.axis('off')
##        plt.savefig('../pyqt5/fig/netArchi.png')
##        pyplot.title( 'Neural Network architecture', fontsize=15 )
##        pyplot.show()

class DrawNN():
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def nn_config(self):
        return self.neural_network

    def getWidestLayer(self):
        widest_layer = max(self.neural_network)
        return widest_layer
    
##    def draw(self):
##        widest_layer = max(self.neural_network)
##        network = NeuralNetwork(widest_layer)
##        for l in self.neural_network:
##            network.add_layer(l)
##        network.nn_draw()

##network = DrawNN([8,4,1])
##network.draw()
