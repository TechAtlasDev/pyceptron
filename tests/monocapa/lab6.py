import sys
sys.path.append("../perceptron")

# Importando las librerías
from perceptron import PerceptronClassic, Linear
from perceptron.utils.generators import Celsius2FahrenheitGenerator
from perceptron.utils.analyzer import Analyzer
from perceptron import MiddlewareTrainingMonolayer


iteraciones = 100
a = 0

for i in range(iteraciones):

  # Creando el set de datos
  generator = Celsius2FahrenheitGenerator()
  x_train, y_train = generator.generate(quantity=100, range_start=-20, range_end=20, shuffle=True)

  # Creando el perceptrón
  perceptron = PerceptronClassic(
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
  epocas = 10
  perceptron.train(
    x=x_train,
    y=y_train,
    epochs=epocas,
    alpha=0.01,
  )

  yi = float(generator.expected(-100021))
  yj = float(perceptron.predict(-100021)[0])

  error = abs(yi - yj)
  if error > 1:
    print ("Anomalía detectada, error: {}".format(error))

  else:
    print (f"El perceptrón aprendió correctamente. -> Error: {error}")
    #generator.graph()
    analyzer.debug()
    print(f"Prediction: {yj:.4f} -> Esperado: {yi:.4f} | Error absoluto: {error:.4f}")
    a += 1

print ("El número de perceptrones que aprendieron correctamente es: {}".format(a))