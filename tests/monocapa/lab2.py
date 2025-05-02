"""
--- ECUACIÓN 1 ---
Problema de regresión lineal

y = y*251-220

El objetivo es predecir el valor de y para un valor de x dado.
"""

import sys
sys.path.append("../perceptron")

from perceptron import Perceptron, Linear
from perceptron.utils.generators import Equation1Generator

generator = Equation1Generator()
x_train, y_train = generator.generate(quantity=100, range_start=-20, range_end=20, shuffle=True)

perceptron = Perceptron(
  f_activation=Linear(),
  input_units=1,
  init_random_hiperparameters=True,
  verbose=True
)

perceptron.train(
  x=x_train,
  y=y_train,
  epochs=20,
  alpha=0.01,
)

# Graficando rendimiento
from perceptron.utils.analyzer import Analyzer

analizer = Analyzer(perceptron)
analizer.error()
analizer.mse()
analizer.debug()

yi = float(generator.expected(-100021))
yj = float(perceptron.predict(-100021)[0])

print (f"Prediction: {yj} -> Esperado: {yi} | Error: {int(round(yi - yj))}")