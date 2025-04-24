from ..abs.perceptron import Perceptron
import numpy as np

class PerceptronBase(Perceptron):
  def __init__(self, f_activation, input_units, init_random_hiperparameters:bool=False, verbose=False):
    self.verbose = verbose
    self.f_activation = f_activation
    self.units = input_units
    self.weights = self._weights(input_units, init_random_hiperparameters)
    self.bias = self._bias(init_random_hiperparameters)

    self.last_z = 0

    self.error_history:list[list[np.ndarray]] = []

  def _bias(self, init_random:bool):
    return 0 if not init_random else np.random.rand()
  
  def _weights(self, input_units, init_random:bool):
    a = np.random.rand(input_units) if init_random else np.zeros(input_units)
    return a

  def _verbose_train(self, epoch, error=1, epochs=1):
    if self.verbose:
      print(f"\rEpoch: {epoch+1}/{epochs} | Error: {error}", end="")

  def train(self, x, y, alpha, epochs):
    for _ in range(epochs):
      ssr = list()      
      for xi, yi in zip(x, y):
        y_pred = self.predict(xi)

        error = yi - y_pred
        self.weights += alpha * error * xi
        self.bias += alpha * error

        ssr.append(error)
      self.error_history.append(ssr)
      self._verbose_train(_, np.mean(ssr), epochs)
    
    print ("\n" if self.verbose else "")

  def predict(self, x):
    self.last_z = x
    dot = np.dot(x, self.weights) + self.bias
    return self.f_activation(dot)
  
  def __repr__(self):
    return f"PerceptronBase(f_activation={self.f_activation}, input_units={self.units}, weights={self.weights}, bias={self.bias})"