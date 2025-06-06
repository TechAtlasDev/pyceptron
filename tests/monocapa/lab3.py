"""
--- Probando el uso de la función de activación ReLU en un perceptrón ---
"""

import sys
sys.path.append("../perceptron")

from pyceptron import PerceptronClassic, ReLU

perceptron = PerceptronClassic(
  f_activation=ReLU(),
  input_units=1,
  init_random_hiperparameters=True
)

xi = perceptron.predict(100)

print (f"Perceptron: {xi}")