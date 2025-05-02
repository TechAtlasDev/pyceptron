from ..abs.perceptron import PerceptronABC
from ...utils.analyzer import Analyzer
from ..enums.middleware_training_monolayer import MiddlewareTrainingMonolayer

import numpy as np

class PerceptronGradient(PerceptronABC):
  def __init__(self, f_activation, input_units, init_random_hiperparameters:bool=False, verbose=False):
    """
    Inicializa el Perceptrón con descenso de gradiente.

    Args:
        f_activation: La función de activación (e.g., sigmoide, ReLU, tanh).
        input_units: El número de características de entrada.
        init_random_hiperparameters (bool, optional): Si inicializar pesos y bias aleatoriamente. Defaults to False.
        verbose (bool, optional): Si imprimir información durante el entrenamiento. Defaults to False.
    """
    self.verbose = verbose
    self.f_activation = f_activation
    self.units = input_units
    self.weights = self._weights(input_units, init_random_hiperparameters)
    self.bias = self._bias(init_random_hiperparameters)

    self.last_input = None # Guardará la última entrada x vista
    self.last_z = 0        # Guardará la última combinación lineal z = w*x + b

    self.error_history:list[list[np.ndarray]] = []
    self._analyzer_middleware = None
    self._options_middleware = []

  # Métodos _bias, _weights, _verbose_train son iguales a PerceptronBase
  def _bias(self, init_random:bool):
    return 0 if not init_random else np.random.rand()

  def _weights(self, input_units, init_random:bool):
    a = np.random.rand(input_units) if init_random else np.zeros(input_units)
    return a

  def _verbose_train(self, epoch, error=1, epochs=1):
    if self.verbose:
      # Calcula la norma del error o el error cuadrático medio si ssr contiene errores al cuadrado
      # Aquí asumimos que error es el error absoluto medio de la época
      print(f"\r[ Epoch: {epoch+1}/{epochs} | Mean Abs Error: {error:.6f} ]", end="")

  # --- Entrenamiento con Descenso de Gradiente ---
  def train(self, x, y, alpha, epochs):
    """
    Entrena el perceptrón usando descenso de gradiente.

    La regla de actualización es:
    delta = (y_true - y_pred) * f_activation_derivative(z)
    weight_nuevo = weight_viejo + alpha * delta * x_i
    bias_nuevo = bias_viejo + alpha * delta
    """
    x_np = np.asarray(x) # Asegurar que x es numpy array
    y_np = np.asarray(y) # Asegurar que y es numpy array
    
    n_samples = x_np.shape[0]

    for epoch in range(epochs):
      epoch_errors = [] # Guardará los errores (y_true - y_pred) de la época
      
      # Iterar sobre cada muestra de entrenamiento
      for i in range(n_samples):
        xi = x_np[i]
        yi = y_np[i]

        # --- Forward pass (Predicción) ---
        # Llama a predict, que internamente guarda xi en self.last_input
        # y calcula z = w*x+b guardándolo en self.last_z
        y_pred = self.predict(xi) 
        
        # --- Backward pass (Cálculo del gradiente y actualización) ---
        error = yi - y_pred

        # Calcular la derivada de la activación en el punto z
        # self.last_z contiene el resultado de la combinación lineal calculado en predict()
        f_prime_z = self.f_activation.derivative(self.last_z)

        # Calcular el término delta (gradiente local)
        delta = error * f_prime_z

        # --- Actualización de pesos y bias ---
        self.weights += alpha * delta * xi # xi ya está guardado en self.last_input si se necesitara
        self.bias += alpha * delta
        # --- Fin Actualización ---

        epoch_errors.append(error) # Guardar el error simple para la historia

        # --- Middleware (si está configurado) ---
        if self._analyzer_middleware:
          data_middleware = {
            "weights": self.weights.copy(),
            "bias": self.bias.copy(),
            "error": error,           # Error simple (y_true - y_pred)
            "delta": delta,           # El término delta calculado
            "f_prime_z": f_prime_z,   # Valor de la derivada de la activación
            "y_pred": y_pred,
            "y_true": yi,
            "xi": xi.copy(),
            "yi": yi,
            "ssr": epoch_errors.copy(), # Historial de errores *dentro* de la época actual
            "last_z": self.last_z,      # z = w*x + b
            "last_input": self.last_input.copy() # x
          }
          self._compile_in_training(data_middleware)
        # --- Fin Middleware ---

      # Guardar el historial de errores de la época completa
      self.error_history.append(epoch_errors)
      
      # Calcular y mostrar el error medio absoluto de la época
      mean_abs_error_epoch = np.mean(np.abs(epoch_errors))
      self._verbose_train(epoch, mean_abs_error_epoch, epochs)

    print ("\n" if self.verbose else "") # Nueva línea al final si verbose

  def linear_combination(self, x):
    # Asegúrate que x sea un array de numpy
    x_np = np.asarray(x) 
    return np.dot(x_np, self.weights) + self.bias

  def predict(self, x):
    # Guardamos la entrada y el resultado de la combinación lineal (antes de activar)
    self.last_input = np.asarray(x) # Asegurar que x es numpy array
    self.last_z = self.linear_combination(self.last_input) 
    # Aplicamos la función de activación
    return self.f_activation(self.last_z)

  # Métodos in_training, _compile_in_training son iguales a PerceptronBase
  def in_training(self, analyzer:Analyzer=None, options:list[MiddlewareTrainingMonolayer]=[]):
    if analyzer:
      if not options:
        raise ValueError("¡Define las operaciones que serán ejecutadas durante el entrenamiento!")
      
    self._analyzer_middleware = analyzer
    self._options_middleware = options

  def _compile_in_training(self, data_middleware:dict):
    for option in self._options_middleware:
      target_key = option.value["target"]
      if target_key not in self._analyzer_middleware.middleware_results:
        self._analyzer_middleware.middleware_results[target_key] = []
      
      if target_key in data_middleware:
          q_data = data_middleware[target_key]
          self._analyzer_middleware.middleware_results[target_key].append(q_data)
      # else:
          # print(f"Warning: Middleware target '{target_key}' not found in data_middleware")

  def __repr__(self):
    f_name = self.f_activation.__name__ if hasattr(self.f_activation, '__name__') else str(self.f_activation)
    return (f"PerceptronBaseGradient(f_activation={f_name}, "
            f"input_units={self.units}, weights={self.weights}, bias={self.bias})")
