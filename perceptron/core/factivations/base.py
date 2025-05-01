class ActFunctionBase:
  def __init__(self):
    self.function:callable = None 
    self.derivative:callable = None

  def __call__(self, x):
    return self.function(x)

  def derivative(self, x):
    return self.derivative(x)
  
  def __repr__(self):
    return f"ActFunctionBase({self.function.__name__}, {self.derivative.__name__})"