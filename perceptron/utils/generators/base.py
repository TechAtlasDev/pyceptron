from .abs import Generator
import numpy

class DatasetGenerator(Generator):
  def __init__(self, function=lambda x: x):
    self.function = function
    self.x = []
    self.y = []

  def generate(self, quantity:int=10, range_start:int=-10, range_end:int=10, shuffle=False):
    if not shuffle:
      self.x = [x for x in range(quantity)]
      self.y = [self.function(x) for x in self.x]

      return self.x, self.y

    self.x = numpy.linspace(range_start, range_end, quantity)
    self.y = [self.function(x) for x in self.x]

    return self.x, self.y
  
  def graph(self):
    import matplotlib.pyplot as plt
    plt.plot(self.x, self.y)
    plt.title("Dataset")
    plt.show()

  def expected(self, x):
    return self.function(x)