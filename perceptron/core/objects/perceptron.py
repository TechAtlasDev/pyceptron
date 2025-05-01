from ..abs.perceptron import Perceptron
from ...utils.analyzer import Analyzer
from ..enums.middleware_training_monolayer import MiddlewareTrainingMonolayer

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
    self._analyzer_middleware = None
    self._options_middleware = []

  def _bias(self, init_random:bool):
    return 0 if not init_random else np.random.rand()
  
  def _weights(self, input_units, init_random:bool):
    a = np.random.rand(input_units) if init_random else np.zeros(input_units)
    return a

  def _verbose_train(self, epoch, error=1, epochs=1):
    if self.verbose:
      print(f"\r[ Epoch: {epoch+1}/{epochs} | Error: {error} ]", end="")

  def train(self, x, y, alpha, epochs):
    for _ in range(epochs):
      ssr = list()
      for xi, yi in zip(x, y):
        # - TRAINING -
        y_pred = self.predict(xi)

        error = yi - y_pred
        self.weights += alpha * error * xi
        self.bias += alpha * error

        ssr.append(error)

        if self._analyzer_middleware:
          data_middleware = {
            "weights": self.weights.copy(),
            "bias": self.bias.copy(),
            "error": error,
            "y_pred": y_pred,
            "y_true": yi,
            "xi": xi.copy(),
            "yi": yi,
            "ssr": ssr.copy(),
            "last_z": self.last_z.copy() if isinstance(self.last_z, np.ndarray) else self.last_z,
          }
          self._compile_in_training(data_middleware)

        # - END TRAINING -
      self.error_history.append(ssr)
      self._verbose_train(_, np.mean(ssr), epochs)

    print ("\n" if self.verbose else "")

  def linear_combination(self, x):
    return np.dot(x, self.weights) + self.bias

  def predict(self, x):
    self.last_z = x
    dot = self.linear_combination(x)
    return self.f_activation(dot)
  
  def in_training(self, analyzer:Analyzer=None, options:list[MiddlewareTrainingMonolayer]=[]):
    if analyzer:
      if not options:
        raise ValueError("¡Define las operaciones que serán ejecutadas durante el entrenamiento!")
      
    self._analyzer_middleware = analyzer
    self._options_middleware = options

  def _compile_in_training(self, data_middleware:dict):
    for option in self._options_middleware:
      if option.value["target"] not in self._analyzer_middleware.middleware_results:
        self._analyzer_middleware.middleware_results[option.value["target"]] = []
      
      q_data = data_middleware[option.value['target']]
      # DEBUG -- print (f"Compiling {option.value['target']} -> {q_data}")
      self._analyzer_middleware.middleware_results[option.value["target"]].append(q_data)
  
  def __repr__(self):
    return f"PerceptronBase(f_activation={self.f_activation}, input_units={self.units}, weights={self.weights}, bias={self.bias})"