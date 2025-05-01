from .base import ActFunctionBase

class Linear(ActFunctionBase):
  def __init__(self):
    self.function = lambda x: x
    self.derivative = lambda x: 1

  def __repr__(self):
    return f"Linear()"