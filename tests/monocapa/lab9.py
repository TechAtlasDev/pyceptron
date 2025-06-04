"""
Inyección de hiperparámetros en el perceptron
"""

import sys
sys.path.append("../perceptron")

from pyceptron import PerceptronClassic
from pyceptron import ELU

weights = [1.8]

perceptron = PerceptronClassic(f_activation=ELU(), input_units=len(weights), verbose=False)
perceptron.weights = weights
perceptron.bias = 32

r = perceptron.predict(15)
print (r)