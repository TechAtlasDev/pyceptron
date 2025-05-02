from abc import ABC, abstractmethod

class PerceptronABC(ABC):
  def __init__(self, f_activation:callable, input_units:int):
    pass

  @abstractmethod
  def train(self, x, y, alpha, epochs):
    pass

  @abstractmethod
  def predict(self, x):
    pass