"""
Cargar un perceptrón
"""

import sys
sys.path.append("../perceptron")

from pyceptron.utils import load_perceptron
from pyceptron.objects.perceptrons.models.perceptronABC import PerceptronABC

route = "perceptrones/perceptron.json"

perceptron:PerceptronABC = load_perceptron(route)
print (perceptron.predict(0))