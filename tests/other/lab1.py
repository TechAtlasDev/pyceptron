"""
Verificar que las funciones de activación funcionan correctamente.
"""

import sys
sys.path.append("../perceptron")

# Importando las librerías
from pyceptron import Linear, Sigmoid, ReLU, Tanh, LeakyReLU, ELU

# Definiendo los valores de prueba
values = [
  -210,
  -1,
  0,
  1,
  210
]

# Definiendo las funciones de activación
activations = [
    Linear(),
    Sigmoid(),
    ReLU(),
    Tanh(),
    LeakyReLU(),
    ELU()
]

# Definiendo los nombres de las funciones de activación
activation_names = [
    "Linear",
    "Sigmoid",
    "ReLU",
    "Tanh",
    "LeakyReLU",
    "ELU"
]

# Iterando sobre las funciones de activación
for activation, name in zip(activations, activation_names):
    print(f"Función de activación: {name}")
    for value in values:
        result = activation(value)
        print(f"  Valor de entrada: {value} -> Valor de salida: {result}")
    print()
