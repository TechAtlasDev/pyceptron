from .layer import Layer
import numpy as np

class Sequential:
  def __init__(self):
    self.layers:list[Layer] = []
    self._last_input_dims:int = -1

  def train(self, epochs:int=10):
    pass

  def predict(self, x:np.array):
    z = self._pre_forward(x)
    return z
  
  def add_layers(self, layers:Layer|list[Layer]):
    if isinstance(layers, list):
      for layer in layers:
        if layer.sinaptic_conections == 0:
          if self._last_input_dims == -1:
            raise("Establece un input_units en la primer capa.")
          
          else:
            layer.sinaptic_conections = self._last_input_dims
            self._last_input_dims = layer.units
        else:
          self._last_input_dims = layer.units

        self.layers.append(layer)
    
    else:
      if self._last_input_dims == -1 and layers.sinaptic_conections == 0:
        raise("Establece un input_units a la capa.")
      self.layers.append(layers)

  def _pre_forward(self, x:np.array):
    last_input = x
    z = 0

    for i, layer in enumerate(self.layers):
      z1 = 0
      for j, perceptron in enumerate(layer.perceptrones):
        z1 += perceptron.predict(last_input)
      last_input = z1
      z += z1
    return z

  def __repr__(self):
    return f"Sequential(layers={len(self.layers)})"