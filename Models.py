import numpy as np

import random as rnd
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Softmax, Conv2D, MaxPooling2D, Flatten, ReLU
import tensorflow as tf
from CustomReLU import CustomReLU
import copy


"""
---------------------------------------------
UTILITY FUNCTIONS
---------------------------------------------
"""

# Deletes null values in tensor to avoid peak values in histograms


def delete_zeros(tensor):
    return tensor[tensor != 0]

# Plots the distribution in difference in models


def plot_hist_diff_layer(x, model_1, model_2, activation_index, no_0=False, activation=False):
    model_1_outputs = model_1.get_activation_layer_outputs(
        x, activation_index, activation)
    model_2_outputs = model_2.get_activation_layer_outputs(
        x, activation_index, activation)

    fig = plt.figure()

    if (no_0):
        diff = abs(delete_zeros(model_2_outputs - model_1_outputs))
    else:
        diff = abs(model_2_outputs - model_1_outputs)

    counts, bins = np.histogram(diff, bins=30)
    plt.hist(bins[:-1], bins, weights=counts,
             label="Difference of outputs distribution")
    plt.title("Difference between model {} and model {} at layer {}".format(
        model_1.name, model_2.name, activation_index))
    plt.xlabel("Outputs values")
    plt.ylabel("# Elements")
    plt.grid(True)
    plt.legend(loc="upper right")

    return fig


"""
---------------------------------------------
HAMMING WEIGHT FUNCTIONS
---------------------------------------------

Functions that are used to convert values to their 
hamming weights. 

"""

# Hamming weight table to convert values
_HW8_table = np.array([bin(x).count('1')
                      for x in range(256)], dtype=(np.uint8))

# Hamming weight convert functions


def hamming_weight_8(x):
    return _HW8_table[x]


def hamming_weight_16(x):
    return hamming_weight_8(x & 255) + hamming_weight_8(x >> 8)


def HW_IEEE754(x):
    return hamming_weight_16(x & 65535) + hamming_weight_16(x >> 16)


def to_HW_layer_output(layer_outputs):
    int32_layer_output = np.ravel(layer_outputs).view(np.uint32)
    HW_layer_output = np.squeeze(HW_IEEE754(
        int32_layer_output).astype(np.uint8))
    return HW_layer_output


"""
---------------------------------------------
CUSTOM CLASS OF TENSORFLOW MODEL FOR TRAINING
---------------------------------------------

This class is used for training only.

Initialisation parameters:
    path : string to indicate where to save the model and its test images.
    dataset_name : string to indicate which dataset should be used ('MNIST' or 'CIFAR10').
    layers_list : list which includes all the layers of the model for training.
                Note -> each ReLU layer will be replaced by CustomReLUs in the Testing Model.   
                Note 2 -> Model already has a Softmax layer at the end.
    name : string for model name.
    input_shape : tuple for shape of inputs.

"""


class Training_Model(Model):
    def __init__(self, path, dataset_name='', layers_list=[Flatten(), Dense(32, name='layer0'), ReLU(), Dense(16, name='layer1'), ReLU(), Dense(16, name='layer2'), ReLU(), Dense(10, name='layer3')], name="Model_0", input_shape=(28, 28, 1)):
        super(Training_Model, self).__init__(name=name)
        self.layers_list = layers_list
        self._input_shape = input_shape
        self.path = path
        self.dataset_name = dataset_name

        # Counts the number of ReLU activations
        count = 0
        for elem in self.layers_list:
            if isinstance(elem, ReLU):
                count += 1
        self.activation_nb = count

    def call(self, x):
        for i in range(len(self.layers_list)):
            x = self.layers_list[i](x)
        return Softmax()(x)


"""
---------------------------------------------
CUSTOM CLASS OF TENSORFLOW MODEL FOR TESTING
---------------------------------------------

This class is used for testing using an already trained model.

Initialisation parameters:
    config : dict with information on the parameters tested. See Config.py.
    training_model : Training model that is already trained. 
                    The object Testing Model created will have the same architecture 
                    as the Training Model where each ReLU will be replaced by a CustomReLU.
                    Note -> 
    c_relus_list : list of CustomReLUs depending on layer. Initialized with classic ReLU parameters. 

"""


class Testing_Model(Model):
    def __init__(self, config, training_model=Training_Model(path=''), c_relus_list=[CustomReLU(1, 0, np.inf)]):
        super(Testing_Model, self).__init__(name=training_model.name)
        self.layers_list = training_model.layers_list
        self.training_model = training_model
        self.path = training_model.path

        # activation_nb is the number of ReLU activations to be replaced in the testing model
        self.activation_nb = training_model.activation_nb
        self.config = config
        self.dataset_name = training_model.dataset_name
        self._input_shape = training_model._input_shape

        # List of CustomReLUs of the testing model
        self.c_relus_list = c_relus_list * self.activation_nb
        self.c_relus_list = self.c_relus_list[:self.activation_nb]

        # Matrix with CustomReLUs parameters depending on the layer
        self.c_relus_matrix = [c_relu.get_parameters()
                               for c_relu in self.c_relus_list]

        # Model preparation
        self.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=["accuracy"],
        )
        self.build(input_shape=self._input_shape)
        self.set_weights(training_model.get_weights())

    # Activation index represents the index of the replaced ReLU starting with 0
    def call(self, x):
        activation_index = 0
        for i in range(len(self.layers_list)):
            # Every occurence of a ReLU layer in the training model is replaced by the corresponding index Custom ReLU in c_relus_list / c_relus_matrix
            if isinstance(self.layers_list[i], ReLU):
                x = tf.keras.activations.relu(self.c_relus_matrix[activation_index][0]*(
                    x - self.c_relus_matrix[activation_index][1]), max_value=self.c_relus_matrix[activation_index][2])
                activation_index += 1
            else:
                x = self.layers_list[i](x)
        return Softmax()(x)

    # Allows to replace a CustomReLU of the model
    def change_c_relu(self, c_relu, activation_i):
        if (activation_i < 0 or activation_i >= self.activation_nb):
            raise AssertionError("Activation index is incorrect")
        else:
            c_relu_i = copy.deepcopy(c_relu)
            self.c_relus_list[activation_i] = c_relu_i
            self.c_relus_matrix = [c_relu.get_parameters()
                                   for c_relu in self.c_relus_list]
            self.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.RMSprop(),
                metrics=["accuracy"],
            )

    # Allows to replace all CustomReLUs of the model by the same new CustomReLU
    def change_all_c_relus(self, c_relu):
        for i in range(self.activation_nb):
            self.c_relus_list[i] = c_relu

        self.c_relus_matrix = [c_relu.get_parameters()
                               for c_relu in self.c_relus_list]
        self.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=["accuracy"],
        )

    def clone(self):
        # Returns a copy of the testing model
        model_copy = Testing_Model(
            config=self.config, training_model=self.training_model, c_relus_list=self.c_relus_list)
        model_copy.build(input_shape=self._input_shape)
        model_copy.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=["accuracy"],
        )
        model_copy.set_weights(self.training_model.get_weights())
        return model_copy

    # Method to get outputs of a specific activation layer
        #   activation = False returns outputs before going through corresponding activation_index CustomReLU
        #   activation = True returns outputs after going through corresponding activation_index CustomReLU

    def get_activation_layer_outputs(self, x, activation_index, activation=False):

        if (activation_index < 0):
            raise AssertionError("Activation index is incorrect")
        elif activation:
            if activation_index > self.activation_nb:
                return self.call(x)
            else:
                activation_j = 0
                for i in range(len(self.layers_list)):
                    if activation_j > activation_index:
                        return x
                    else:
                        if isinstance(self.layers_list[i], ReLU):
                            x = tf.keras.activations.relu(self.c_relus_matrix[activation_index][0]*(
                                x - self.c_relus_matrix[activation_index][1]), max_value=self.c_relus_matrix[activation_index][2])
                            activation_j += 1
                        else:
                            x = self.layers_list[i](x)
        else:
            activation_j = 0
            for i in range(len(self.layers_list)):
                if activation_j >= activation_index - 1:
                    p = i
                    while not isinstance(self.layers_list[p], ReLU):
                        x = self.layers_list[p](x)
                        p += 1
                    return x
                else:
                    if isinstance(self.layers_list[i], ReLU):
                        x = tf.keras.activations.relu(self.c_relus_matrix[activation_index][0]*(
                            x - self.c_relus_matrix[activation_index][1]), max_value=self.c_relus_matrix[activation_index][2])
                        activation_j += 1
                    else:
                        x = self.layers_list[i](x)

    # Plots an histogram of the distribution depending on the activation index
        #   no_0 = True plots without zero value outputs
        #   no_0 = False plots with zero value outputs

    def plot_hist_distrib_layer(self, x, activation_index, no_0=False):

        outputs_before_activation = self.get_activation_layer_outputs(
            x, activation_index, activation=False)
        outputs_after_activation = self.get_activation_layer_outputs(
            x, activation_index, activation=True)

        fig = plt.figure()

        counts, bins = np.histogram(outputs_before_activation, bins=30)
        plt.hist(bins[:-1], bins, weights=counts, label="Before activation")

        if (no_0):
            counts, bins = np.histogram(delete_zeros(
                outputs_after_activation), bins=30)
            plt.hist(bins[:-1], bins, weights=counts, label='After activation')

        else:
            counts, bins = np.histogram(outputs_after_activation, bins=30)
            plt.hist(bins[:-1], bins, weights=counts, label='After activation')

        plt.title(
            "Model distribution of outputs at layer {} - Model : {}".format(activation_index, self.name))
        plt.xlabel("Outputs values")
        plt.ylabel("# Elements")
        plt.grid(True)
        plt.legend(loc="upper right")

        return fig

    # Plots an histogram of the distribution depending on the activation index with hamming weights
        #   no_0 = True plots without zero value outputs
        #   no_0 = False plots with zero value outputs

    def plot_HW_distrib_hist_layer(self, x, activation_index, no_0=False, activation=True):
        layer_outputs = self.get_activation_layer_outputs(
            x, activation_index, activation)
        HW_layer_outputs = to_HW_layer_output(layer_outputs)
        nb_bit = np.array([x for x in range(32)])

        fig = plt.figure()

        counts, bins = np.histogram(HW_layer_outputs, bins=nb_bit)

        if (no_0):
            plt.hist(bins[1:-1], bins, weights=counts[1:], label="HW")
        else:
            plt.hist(bins[:-1], bins, weights=counts, label="HW")

        plt.title("Model distribution of hamming weights outputs at layer - Model : {}".format(
            activation_index, self.name))
        plt.xlabel("Hamming weight values")
        plt.ylabel("# Elements")
        plt.grid(True)
        plt.legend(loc="upper right")

        return fig

    # Print Custom ReLUs parameters on all layers
    def print_parameters(self):
        print("Model's Custom ReLUs parameters :")
        for i in range(len(self.c_relus_list)):
            print("\nActivation layer {} ==> ".format(i))
            self.c_relus_list[i].print_parameters()
