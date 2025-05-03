"""
--- CELSIUS A FAHRENHEIT ---

Laboratorio que compara el rendimiento entre el perceptrón de entrenamiento clásico y el perceptrón de entrenamiento a base de gradientes.
"""

import sys
sys.path.append("../perceptron")

# Importando las librerías
from perceptron import *
from perceptron.utils.generators import Celsius2FahrenheitGenerator
from perceptron.utils.analyzer import Analyzer
from perceptron import MiddlewareTrainingMonolayer

# Creando el set de datos
generator = Celsius2FahrenheitGenerator()
x_train, y_train = generator.generate(quantity=50, range_start=-20, range_end=20, shuffle=True)

data_options = [
  MiddlewareTrainingMonolayer.HISTORY_WEIGHTS,
  MiddlewareTrainingMonolayer.HISTORY_BIASES,
]
epocas = 100

# -- Preparando perceptrón clásico
perceptron_clasico = PerceptronClassic(
  f_activation=Linear(),
  input_units=1,
  init_random_hiperparameters=True,
)
analyzer_classic = Analyzer(perceptron_clasico)
perceptron_clasico.in_training(analyzer=analyzer_classic, options=data_options)

# -- Preparando perceptrón de gradiente
perceptron_gradiente = PerceptronGradient(
  f_activation=ELU(1.5),
  input_units=1,
  init_random_hiperparameters=True,
)
analyzer_gradient = Analyzer(perceptron_gradiente)
perceptron_gradiente.in_training(analyzer=analyzer_gradient, options=data_options)

analyzer_gradient.console.separator("¡Perceptrones creados!")
analyzer_gradient.console.separator("Entrenando a los perceptrones", length=15, separator="+")

# Entrenando al perceptrón clásico
analyzer_classic.console.separator(" +++ Entrenando al perceptrón clásico +++ ")
perceptron_clasico.train(
  x=x_train,
  y=y_train,
  epochs=epocas,
  alpha=0.01,
)
analyzer_classic.debug()

# Entrenando al perceptrón de gradiente
analyzer_gradient.console.separator(" +++ Entrenando al perceptrón de gradiente +++ ")
perceptron_gradiente.train(
  x=x_train,
  y=y_train,
  epochs=epocas,
  alpha=0.01,
)
analyzer_gradient.debug()

# Comparando resultados
from matplotlib import pyplot as plt
error_history_classic = [sum(ssr) / len(ssr) for ssr in perceptron_clasico.error_history]
error_history_gradient = [sum(ssr) / len(ssr) for ssr in perceptron_gradiente.error_history]

plt.plot(error_history_classic, label="Perceptron clásico") # Graficando el error del perceptrón clásico
plt.plot(error_history_gradient, label="Perceptron de gradiente") # Graficando el error del perceptrón clásico

plt.grid()
plt.title("Comparación de rendimiento")
plt.xlabel("Época")
plt.ylabel("Error")
plt.legend()
#plt.annotate("Estabilización", xy=(10, error_history_gradient[10]), xytext=(12, error_history_gradient[10]+0.5),
#             arrowprops=dict(facecolor='green', arrowstyle="->"))
plt.show()