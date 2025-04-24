"""
--- CELSIUS A FAHRENHEIT ---

Problema clásico de regresión lineal
El problema de la conversión de Celsius a Fahrenheit es un problema clásico de regresión lineal."""

import sys
sys.path.append("../perceptron")

from perceptron import PerceptronBase, linear
from perceptron.utils.generators import Celsius2FahrenheitGenerator

generator = Celsius2FahrenheitGenerator()
x_train, y_train = generator.generate(quantity=100, range_start=-20, range_end=20, shuffle=True)

perceptron = PerceptronBase(
  f_activation=linear,
  input_units=1,
  init_random_hiperparameters=True,
  verbose=True
)

perceptron.train(
  x=x_train,
  y=y_train,
  epochs=500,
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

print (f"Prediction: {yj} -> Esperado: {yi} | Error: {int(yi - yj)}")