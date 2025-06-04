"""
--- ANÁLISIS MANUAL DEL PERCEPTRON MULTICAPA Y ENTRENAMIENTO
"""

import sys
sys.path.append("../perceptron")

from termpyx import Console
from termpyx.src.enums.color import Color

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

# -- interconectividad

import numpy as np

x_input = generator.expected(2)
resultado1 = perceptron1.predict(x_input) # Perceptrones de entrada
resultado2 = perceptron2.predict(resultado1) # Capa densa 1
resultado3 = perceptron3.predict(resultado1) # Capa densa 1
resultado4 = perceptron4.predict(np.array([resultado2, resultado3])) # Capa de salida

# Analizando los resultados

print (f"Resultado: {resultado4} para la entrada {x_input}")
error = x_input-resultado4
print (f"Error: {error}")

# Backpropagation
console.separator("Backpropagation")

console.info(f"Perceptron 4 antes de la actualización de pesos: \n{perceptron4}")
console.info(f"Perceptron 3 antes de la actualización de pesos: \n{perceptron3}")
console.info(f"Perceptron 2 antes de la actualización de pesos: \n{perceptron2}")
console.info(f"Perceptron 1 antes de la actualización de pesos: \n{perceptron1}")

# -- Backward
console.separator("Backward iniciado", separator="-", length=6, color=Color.GREEN)
alpha = 0.002
relu = ReLU()

console.separator("Perceptron 4", separator="_", length=5, color=Color.MAGENTA)
d_resultado4 = error * relu.derivative(resultado4)
dw4 = d_resultado4 * np.array([resultado2, resultado3])
db4 = d_resultado4

for i in [perceptron4.weights, alpha, dw4, dw4]:
  console.log(i)
perceptron4.weights += alpha * dw4
perceptron4.bias += alpha * db4

console.info(f"Perceptron 4: {perceptron4}")

console.separator("Perceptron 3", separator="_", length=5, color=Color.MAGENTA)
d_resultado3 = error * relu.derivative(resultado3)
dw3 = d_resultado3 * np.array(resultado2)
db3 = d_resultado3

perceptron3.weights += alpha * dw3
perceptron3.bias += alpha * db3

console.info(f"Perceptron 3: {perceptron3}")

console.separator("Perceptron 2", separator="_", length=5, color=Color.MAGENTA)
d_resultado2 = error * relu.derivative(resultado2)
dw2 = d_resultado2 * np.array(resultado1)
db2 = d_resultado2

perceptron2.weights += alpha * dw2
perceptron2.bias += alpha * db2

console.info(f"Perceptron 2: {perceptron2}")

console.separator("Perceptron 1", separator="_", length=5, color=Color.MAGENTA)
d_resultado1 = error * relu.derivative(resultado1)
dw1 = d_resultado2 * np.array(x_input)
db1 = d_resultado1

perceptron1.weights += alpha * dw1
perceptron1.bias += alpha * db1

console.info(f"Perceptron 1: {perceptron1}")

# -- Prueba
console.separator("Volviendo a probar el modelo", separator="-", length=6, color=Color.GREEN)

x_input = generator.expected(2)
resultado1 = perceptron1.predict(x_input) # Perceptrones de entrada
resultado2 = perceptron2.predict(resultado1) # Capa densa 1
resultado3 = perceptron3.predict(resultado1) # Capa densa 1
resultado4 = perceptron4.predict(np.array([resultado2, resultado3])) # Capa de salida

# Analizando los resultados

print (f"Resultado: {resultado4} para la entrada {x_input}")
error1 = x_input-resultado4
print (f"Error: {error1}")


print (f"Diferencial: {error-error1}")