import sys
sys.path.append("../perceptron")

from perceptron import ReLU, Layer, Sequential, Linear
from termpyx import Console
import numpy as np

console = Console()

modelo = Sequential()
capa1 = Layer(units=1, activation=ReLU(), input_units=1)
capa2 = Layer(units=2, activation=ReLU())
capa3 = Layer(units=1, activation=ReLU())

layers = [capa1, capa2, capa3]
modelo.add_layers(layers)

console.separator("Modelo")
console.info(modelo.layers)
console.info(modelo)

console.separator("Capas")
console.log(capa1)
console.log(capa2)
console.log(capa3)

console.separator("Predicción")
import matplotlib.pyplot as plt

value_test = np.array([1])
console.log(f"Dato de prueba: {value_test}")
value_predicted = modelo.predict(value_test)
console.success(f"Dato de retorno: {value_predicted}")

x = list(range(10))
y = [modelo.predict(z) for z in x]

plt.plot(x, y)
plt.show()