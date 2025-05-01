"""
--- CELSIUS A FAHRENHEIT ---

Problema clásico de regresión lineal
El problema de la conversión de Celsius a Fahrenheit es un problema clásico de regresión lineal."""

import sys
sys.path.append("../perceptron")

# Importando las librerías
from perceptron import PerceptronBase, Linear
from perceptron.utils.generators import Celsius2FahrenheitGenerator
from perceptron.utils.analyzer import Analyzer
from perceptron import MiddlewareTrainingMonolayer

# Creando el set de datos
generator = Celsius2FahrenheitGenerator()
x_train, y_train = generator.generate(quantity=50, range_start=-20, range_end=20, shuffle=True)

# Creando el perceptrón
perceptron = PerceptronBase(
  f_activation=Linear(),
  input_units=1,
  init_random_hiperparameters=True,
  verbose=True
)

# Creando el analyzer
analyzer = Analyzer(perceptron)

# Antes de entrenar el modelo, se va a analizar estos datos:
datos = [
  MiddlewareTrainingMonolayer.HISTORY_WEIGHTS,
  MiddlewareTrainingMonolayer.HISTORY_BIASES,
]

perceptron.in_training(analyzer=analyzer, options=datos)

# Entrenando al perceptrón
epocas = 25
perceptron.train(
  x=x_train,
  y=y_train,
  epochs=epocas,
  alpha=0.01,
)

# Graficando rendimiento
analyzer.error()
analyzer.mse()
analyzer.debug()

yi = float(generator.expected(-100021))
yj = float(perceptron.predict(-100021)[0])

error = abs(yi - yj)
print(f"Prediction: {yj:.4f} -> Esperado: {yi:.4f} | Error absoluto: {error:.4f}")

# Analizando los pesos
analyzer.history_weights()

# Analizando los bias
analyzer.history_bias()

generator.graph_x()
generator.graph_y()

analyzer.compare_graph(generator)