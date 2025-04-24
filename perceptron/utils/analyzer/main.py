from termpyx import Console
from matplotlib import pyplot as plt

class Analyzer:
  def __init__(self, perceptron):
    self.perceptron = perceptron
    self.console = Console(in_debug=True)

  def mse(self):
    error_mse = []
    for ssr in self.perceptron.error_history:
      error_mse.append(
        self.mse_calc(ssr)
      )

    plt.grid()
    plt.plot(error_mse)
    plt.title("MSE history")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()
    return error_mse

  def mse_calc(self, data:list):
    return (sum([(error ** 2) for error in data]) / len(data))

  def error(self):
    history_ssr = []
    for ssr in self.perceptron.error_history:
      history_ssr.append(
        sum(ssr) / len(ssr)
      )

    #plt.grid()
    plt.plot(history_ssr)
    plt.title("Error history")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

  def error_calc(self, data:list):
    return sum(data)/len(data)

  def ssr(self):
    unit_ssr = []
    for ssr in self.perceptron.error_history:
      for d in ssr:
        unit_ssr.append(d[0])

    plt.grid()
    plt.plot(unit_ssr)
    plt.title("SSR history")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

  def debug(self):
    error_history = self.perceptron.error_history
    self.console.info(f"Longitud de épocas: {len(error_history)}")
    self.console.info(f"Cantidad de intentos por cada época: {len(error_history[0])}")
    self.console.info(f"Error de la última época: {self.error_calc(error_history[-1])}")
    self.console.info(f"Error cuadrático medio de la primera y última eṕoca: {self.mse_calc(error_history[0])} & {self.mse_calc(error_history[-1])}")

    # Analizando el perceptrón:
    self.console.separator("Hiperparámetros del perceptrón")
    self.console.info(f"Función de activación: {self.perceptron.f_activation.__name__}")
    self.console.info(f"Umbral: {self.perceptron.bias}")
    self.console.info(f"Pesos: {self.perceptron.weights}")

  def graph_error_list(self, error_list:list):
    plt.grid()
    plt.plot(error_list)
    plt.title("Error history")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()