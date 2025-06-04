"""
--- ANÁLISIS SEMI AUTOMÁTICO DEL PERCEPTRÓN MULTICAPA Y SU ENTRENAMIENTO ---
"""

import sys
sys.path.append("../perceptron")

from termpyx import Console
from termpyx.src.enums.color import Color
import numpy as np

from pyceptron import PerceptronBase
from pyceptron.utils.generators import Parabola1Generator
from pyceptron import ReLU

console = Console()
generator = Parabola1Generator()
x_train, y_train = generator.generate()

# -- Zona de perceptrones
perceptron1 = PerceptronBase(
  f_activation=ReLU(),
  input_units=1,
  init_random_hiperparameters=True
)

perceptron2 = PerceptronBase(
  f_activation=ReLU(),
  input_units=1,
  init_random_hiperparameters=True
)

perceptron3 = PerceptronBase(
  f_activation=ReLU(),
  input_units=1,
  init_random_hiperparameters=True
)

perceptron4 = PerceptronBase(
  ReLU(),
  input_units=2,
  init_random_hiperparameters=True
)

def predict(x_input):
  resultado1 = perceptron1.predict(x_input) # Perceptrones de entrada
  resultado2 = perceptron2.predict(resultado1) # Capa densa 1
  resultado3 = perceptron3.predict(resultado1) # Capa densa 1
  resultado4 = perceptron4.predict(np.array([resultado2, resultado3])) # Capa de salida
  return resultado4

def train(epochs=10):
  alpha = 0.001
  relu = ReLU()
  
  for i in range(epochs):
    x_input = generator.expected(2)
    resultado4 = predict(x_input)
    error = x_input - resultado4

    console.separator("Perceptron 4", separator="_", length=5, color=Color.MAGENTA)
    d_resultado4 = error * relu.derivative(resultado4)
    dw4 = d_resultado4 * np.array([perceptron2.last_z[0], perceptron3.last_z[0]])
    db4 = d_resultado4

    for i in [perceptron4.weights, alpha, dw4, dw4]:
      console.log(i)
    perceptron4.weights -= alpha * dw4
    perceptron4.bias -= alpha * db4

    console.info(f"Perceptron 4: {perceptron4}")

    console.separator("Perceptron 3", separator="_", length=5, color=Color.MAGENTA)
    d_resultado3 = error * relu.derivative(perceptron3.last_z[0])
    dw3 = d_resultado3 * np.array(perceptron2.last_z[0])
    db3 = d_resultado3

    perceptron3.weights -= alpha * dw3
    perceptron3.bias -= alpha * db3

    console.info(f"Perceptron 3: {perceptron3}")

    console.separator("Perceptron 2", separator="_", length=5, color=Color.MAGENTA)
    d_resultado2 = error * relu.derivative(perceptron2.last_z[0])
    print (perceptron1.last_z)
    dw2 = d_resultado2 * np.array(perceptron1.last_z)
    db2 = d_resultado2

    perceptron2.weights -= alpha * dw2
    perceptron2.bias -= alpha * db2

    console.info(f"Perceptron 2: {perceptron2}")

    console.separator("Perceptron 1", separator="_", length=5, color=Color.MAGENTA)
    d_resultado1 = error * relu.derivative(perceptron1.last_z)
    dw1 = d_resultado2 * np.array(x_input)
    db1 = d_resultado1

    perceptron1.weights -= alpha * dw1
    perceptron1.bias -= alpha * db1

    console.info(f"Perceptron 1: {perceptron1}")

train(10000)

x_input = generator.expected(2)
r = predict(2)

print (r, x_input)