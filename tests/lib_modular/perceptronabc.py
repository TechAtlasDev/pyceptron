from abc import ABC, abstractmethod

class PerceptronABC(ABC):
  def __init__(self, *args, **kwargs):
    super().__init__(
      *args,
      **kwargs
    )

  @abstractmethod
  def train(self, epochs:int, alpha:float):
    """
    Método abstracto para entrenar el perceptrón.
    """
    pass

  @abstractmethod
  def predict(self, x):
    """
    Método abstracto para predecir el valor de salida del perceptrón.
    """
    pass

  @abstractmethod
  def in_training(self, analyzer=None, options:list=[]):
    """
    Método abstracto para definir el analizador y las opciones de entrenamiento.
    """
    pass