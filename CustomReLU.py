import random as rnd
import matplotlib.pyplot as plt
import numpy as np

"""
---------------------------------------------
FUNCTIONAL CLASS OF CUSTOM RELU
---------------------------------------------

This class is used to create a Custom ReLU.


Initialisation parameters:
  beta, gamma, clipping : floating values for Custom ReLU parameters.
  
"""

class CustomReLU():
  """"""
  def __init__(self, *args):
    if len(args) == 3:
      #Can take direct parameters as args ...
      self.beta, self.gamma, self.clipping = args
    else:
      raise AssertionError('Invalid parameters')

  #Computes CustomReLU(x) where x is float
  def value(self, x):
    value = x - self.gamma
    if value > 0 and self.beta*value > self.clipping:
      return self.clipping
    elif value > 0:
      return self.beta*value
    else:
      return 0

  #Computes CustomReLU(X) where X is an array
  def array(self,X):
    array = []
    for x in X:
      array.append(self.value(x))
    return array
  
  def print_parameters(self):
    print("beta = {:.3f}/ gamma = {:.3f} / clipping = {:.3f}".format(self.beta, self.gamma, self.clipping))

  def get_parameters(self):
    return [self.beta,self.gamma, self.clipping]
  
  #Returns a new CustomReLU with a new parameter value
  def change_parameter(self, parameter, parameter_value):
    
    if not parameter in ['beta', 'gamma', 'clipping']:
      raise AssertionError("Invalid parameter. Parameter must be 'beta', 'gamma' or 'clipping'")
    else : 
      if parameter == 'beta':
        return CustomReLU(parameter_value, self.gamma, self.clipping)
      if parameter == 'gamma':
        return CustomReLU(self.beta, parameter_value,  self.clipping)
      if parameter == 'clipping':
        return CustomReLU(self.beta, self.gamma, parameter_value)

  #Shows the graph of the function
  def show_graph(self):
    X = np.linspace(-20,20,1000)
    Y = self.array(X)
    Y_relu = [max(value,0) for value in X]
    plt.plot(X, Y, label = 'Custom ReLU')
    plt.plot(X,Y_relu, label='ReLU')
    plt.hlines(y=0,xmin=-20,xmax=20,colors='red', linestyles='dashed', lw=0.5)
    plt.vlines(x=0,ymin = min(Y[0],Y_relu[0]),ymax = max(Y[-1],Y_relu[-1]),colors='red', linestyles='dashed', lw=0.5)
    plt.xlabel('x')
    plt.ylabel('Custom ReLU(x)')
    plt.title("Comparison between CustomReLU and ReLU")
    plt.legend()
    plt.show()
