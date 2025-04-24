import sys
sys.path.append("../perceptron")

from perceptron import PerceptronBase
from perceptron.utils.generators import Parabola1Generator
from perceptron import relu

generator = Parabola1Generator()
x_train, y_train = generator.generate()

# -- Zona de perceptrones
perceptron1 = PerceptronBase(
  relu,
  1,
  True
)

perceptron2 = PerceptronBase(
  relu,
  1,
  True
)

perceptron3 = PerceptronBase(
  relu,
  1,
  True
)

perceptron4 = PerceptronBase(
  relu,
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