"""
--- Probando el uso de la función de activación ReLU en un perceptrón ---
"""

import sys
sys.path.append("../perceptron")

from perceptron import PerceptronBase, relu

perceptron = PerceptronBase(
  f_activation=relu,
  input_units=1
)

xi = perceptron.predict(100)

print (f"Perceptron: {xi}")