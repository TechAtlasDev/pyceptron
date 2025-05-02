"""
--- Probando el uso de la función de activación ReLU en un perceptrón ---
"""

import sys
sys.path.append("../perceptron")

from perceptron import Perceptron, ReLU

perceptron = Perceptron(
  f_activation=ReLU(),
  input_units=1,
  init_random_hiperparameters=True
)

xi = perceptron.predict(100)

print (f"Perceptron: {xi}")