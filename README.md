# 🧠 Pyceptron

<iframe src="https://app.warp.dev/block/embed/UE8Y0j28TomxzFn59Jfjk5" title="embedded warp block" style="width: 922px; height: 435px; border:0; overflow:hidden;" allow="clipboard-read; clipboard-write"></iframe>

**Pyceptron** es una librería open-source desarrollada con el objetivo de desarrollar, analizar y demostrar muchos conceptos que un perceptrón nos ofrece, a base de una arquitectura minimalista y muy fácil de desarrollar.

## 🖥️ Instalación

Puedes instalar la librería usando el comando

```bash
pip install pyceptron
```

O puedes usar poetry con

```bash
poetry add pyceptron
```

## 🎯 Uso básico

La librería fue diseñada para que puedas implementar de manera fácil y rápida tus soluciones!

```python
# Importar las librerías
from pyceptron import PerceptronClassic
from pyceptron import Linear

from ... import dataset # Importa tu dataset

# Creando el perceptron
perceptron = PerceptronClassic(
  f_activation=Linear(),
  input_units=1
)

perceptron.train(
  x=dataset.x, y=dataset.y, alpha=0.01, epochs=20
)

# -- Evaluando perceptron --
x_test = 100
y_pred = perceptron.predict(x_test)

print (f"Predicción de {x_test} -> Predicho: {y_pred}")
```

## ⚒️ Funciones extra

**Pyceptron** va más allá de un simple `train()` y `predict()`. Incluye herramientas potentes para que entiendas a fondo el proceso de aprendizaje:

### Generadores de Datasets `(pyceptron.utils.generators)`

No necesitas preparar tus datos manualmente. **Pyceptron** te permite crear datasets personalizados con relaciones matemáticas predefinidas, perfectos para experimentar y validar el comportamiento de los perceptrones.

La clase base `DatasetGenerator` ofrece una implementación robusta para la creación de conjuntos de datos. Permite definir la **cantidad** de puntos, el **rango** de los valores de entrada (`x`), y opcionalmente **barajar** los datos. Es fundamental destacar que, al barajar (`shuffle=True`), la librería asegura que los **pares (X, Y) se mantengan intactos**, preservando la relación subyacente del dataset. Esto evita los problemas de divergencia catastrófica que ocurren cuando `X` y `Y` son aleatorizados independientemente.

Puedes definir tus propios generadores heredando de `DatasetGenerator` y simplemente proporcionando la función matemática deseada:

```Python
from pyceptron.utils.generators.base import DatasetGenerator

# --- Ejemplos de Generadores Incluidos ---

# Generador de datos para la conversión Celsius a Fahrenheit (relación lineal)
class Celsius2FahrenheitGenerator(DatasetGenerator):
  def __init__(self, function=lambda x : x * 9/5 + 32):
    super().__init__(function=function) # Llama al constructor de la base

# Generador de datos para una relación lineal diferente
class Equation1Generator(DatasetGenerator):
  def __init__(self, function=lambda x : x * 251 - 220):
    super().__init__(function=function)

# Generador de datos para una relación parabólica (no lineal)
class Parabola1Generator(DatasetGenerator):
  def __init__(self, function=lambda x : x**2 + x*3 + 5):
    super().__init__(function=function)
```

**Ejemplos de Uso:**

```Python
from pyceptron.utils.generators import Celsius2FahrenheitGenerator, Parabola1Generator

# Generar datos para la conversión Celsius a Fahrenheit
celsius_generator = Celsius2FahrenheitGenerator()
x_celsius, y_celsius = celsius_generator.generate(quantity=100, range_start=-50, range_end=50, shuffle=True)

# Generar datos para una relación parabólica
parabola_generator = Parabola1Generator()
x_parabola, y_parabola = parabola_generator.generate(quantity=100, range_start=-10, range_end=10)

# Puedes obtener el valor esperado para cualquier entrada, manteniendo la coherencia
expected_f = celsius_generator.expected(25) # Devuelve 77.0

# Los generadores también incluyen métodos para visualizar los datos:
celsius_generator.graph() # Grafica la relación X vs Y
celsius_generator.graph_x() # Grafica la distribución de X
celsius_generator.graph_y() # Grafica la distribución de Y
```

Estos generadores son cruciales para probar cómo tus perceptrones se adaptan a diferentes tipos de problemas *(lineales vs. no lineales)* y para investigar el impacto del orden de los datos.

### Analizador de Rendimiento `(pyceptron.utils.analyzer)`
El `Analyzer` es tu laboratorio de depuración y visualización. Te permite una introspección profunda del proceso de entrenamiento del perceptrón.

- **Monitoreo del Error:** Visualiza el historial del error promedio y el Error Cuadrático Medio (MSE) a lo largo de las épocas, dándote una idea clara de la convergencia (o divergencia) del modelo.

- **Historial de Hiperparámetros:** Observa cómo evolucionan los pesos y el bias del perceptrón durante el entrenamiento. Esto es invaluable para entender el proceso de optimización.

- **Comparación de Datos:** Gráfica las predicciones del perceptrón frente a la "línea de frontera" real de tus datos, permitiendo un diagnóstico visual inmediato del rendimiento del modelo.

- **Resumen Detallado:** Obtén un desglose en consola de las métricas clave y los hiperparámetros finales del perceptrón.

```Python
from pyceptron import PerceptronClassic, Linear
from pyceptron.utils.generators import Celsius2FahrenheitGenerator
from pyceptron.utils.analyzer import Analyzer
from pyceptron.enums.middleware_training_monolayer import MiddlewareTrainingMonolayer

# ... (código de creación del dataset y perceptrón, como en el "Uso Básico") ...

# Instancia el analizador con tu perceptrón
analyzer = Analyzer(perceptron)

# Configura el perceptrón para que el analizador recolecte datos durante el entrenamiento
perceptron.in_training(
  analyzer=analyzer,
  options=[
    MiddlewareTrainingMonolayer.HISTORY_WEIGHTS, # Recolectar historial de pesos
    MiddlewareTrainingMonolayer.HISTORY_BIASES,  # Recolectar historial de bias
  ]
)

# ... (entrenamiento del perceptrón) ...

# Usa las funciones del analizador para visualizar y depurar:
analyzer.mse()
analyzer.error()
analyzer.history_weights()
analyzer.history_bias()
analyzer.compare_graph(generator)
analyzer.debug()
```

Esta herramienta es vital para comprender la estabilidad, la convergencia y las posibles anomalías en el aprendizaje, como se demostró en nuestros [laboratorios de comparación entre perceptrones](https://github.com/TechAtlasDev/pyceptron/tree/main/tests/monocapa).

### Persistencia de Modelos

Guarda y carga tus perceptrones entrenados fácilmente para evitar reentrenamientos y para usar tus modelos en aplicaciones:

```Python
from pyceptron.utils import load_perceptron

route = "perceptrones/perceptron.json"

perceptron = load_perceptron(route)
print (perceptron.predict(0))
```

## 📐 Arquitectura del proyecto

Pyceptron está diseñado con la modularidad como pilar central, permitiendo una fácil comprensión y extensibilidad.

- `pyceptron/factivations`: Contiene implementaciones de diversas funciones de activación (lineales y no lineales).

- `pyceptron/objects/perceptrons`: Aquí residen las definiciones de los modelos de perceptrones.

  - `models/`: Define las clases base abstractas para garantizar una interfaz coherente.
  - `pieces/`: Componentes reutilizables como la lógica central y el exportador/importador de modelos. Dentro de `trainers/` encontrarás las implementaciones de los algoritmos de entrenamiento específicos (clásico, gradiente).
  - `variants/`: Las implementaciones concretas de los perceptrones (`PerceptronClassic`, `PerceptronGradient`) que combinan la base del perceptrón con los distintos entrenadores usando herencia múltiple.
- `pyceptron/objects/layers`: Aunque actualmente enfocado en perceptrones monocapa, la presencia de `layer.py` y `sequential.py` anticipa futuras extensiones para redes neuronales multicapa.
- `pyceptron/utils`: Utilidades esenciales para el ecosistema:
  - `analyzer/:` El motor detrás de las herramientas de análisis y visualización.
  - `generators/`: Herramientas para crear datasets de prueba con diferentes relaciones matemáticas.
  - `loaders/`: Funcionalidad para cargar modelos guardados.
- `pyceptron/enums`: Enumeraciones para gestionar opciones internas de manera clara, como las opciones de recolección de datos del middleware durante el entrenamiento.

Esta estructura promueve la **separación de responsabilidades**, haciendo que cada parte del código sea más manejable y el proyecto sea altamente escalable para futuras características.

## 🤝 Contribuciones
¡Las contribuciones son bienvenidas! Si deseas contribuir, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una rama (`git checkout -b feature/nueva-feature`).
3. Realiza tus cambios y haz un commit (`git commit -am 'Añadir nueva-feature'`).
4. Haz un push a la rama (`git push origin feature/nueva-feature`).
5. Abre un Pull Request.

## 📝 Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## 📬 Contacto
Para cualquier pregunta o sugerencia, por favor abre un issue en el repositorio o contacta a gjimenezdeza@gmail.com.

¡Gracias por usar Pyceptron! 🚀