"""
--- DETECTANDO ANOMALÍAS EN EL DATASET - CELSIUS A FAHRENHEIT ---

Problema clásico de regresión lineal
El problema de la conversión de Celsius a Fahrenheit es un problema clásico de regresión lineal.

ESTE LABORATORIO SE ENFOCA EN ANALIZAR EL COMPORTAMIENTO DE PERCEPTRONES CUYO APRENDIZAJE ES NULO Y EMPEORA POR CADA ENTRENAMIENTO

HIPÓTESIS: El perceptrón tiene la capacidad de poder aprender, este laboratorio se enfoca en el análisis de diferentes perceptrones inicializados con datasets diferentes.
  |- Si se encuentra un problema durante el entrenamiento o predicción, se debe al dataset.
"""

#epocas = int(input("Inserta la cantidad de épocas: "))

import sys
sys.path.append("../perceptron")

# Importando las librerías
from perceptron import Perceptron, Linear
from perceptron.utils.generators import Celsius2FahrenheitGenerator
from perceptron.utils.analyzer import Analyzer
from perceptron import MiddlewareTrainingMonolayer

iteraciones = 10

for i in range(iteraciones):

  # Creando el set de datos
  generator = Celsius2FahrenheitGenerator()
  x_train, y_train = generator.generate(quantity=100, range_start=-20, range_end=20, shuffle=True)

  # Creando el perceptrón
  perceptron = Perceptron(
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
  epocas = 15
  perceptron.train(
    x=x_train,
    y=y_train,
    epochs=epocas,
    alpha=0.01,
  )

  yi = float(generator.expected(-100021))
  yj = float(perceptron.predict(-100021)[0])

  error = abs(yi - yj)
  if error > 10:

    print ("Anomalía detectada, error: {}".format(error))
    # Graficando rendimiento
    analyzer.error()
    analyzer.mse()
    analyzer.debug()

    print(f"Prediction: {yj:.4f} -> Esperado: {yi:.4f} | Error absoluto: {error:.4f}")

    # Analizando los pesos
    analyzer.history_weights()

    # Analizando los bias
    analyzer.history_bias()

    print (f"Longitud del dataset: {len(x_train)}")

  else:
    print (f"El perceptrón aprendió correctamente. -> Error: {error}")
    generator.graph()
    print (f"Longitud del dataset: {len(x_train)}")
    analyzer.debug()
    print(f"Prediction: {yj:.4f} -> Esperado: {yi:.4f} | Error absoluto: {error:.4f}")
    print (f"Longitud del dataset: {len(x_train)}")

  input("Presiona enter para continuar...")