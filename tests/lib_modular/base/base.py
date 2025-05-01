from .perceptron_train import PerceptronTrainBase
from .perceptron_predict import PerceptronPredictBase

class PerceptronBase(PerceptronTrainBase, PerceptronPredictBase):
  def __init__(self):
    super().__init__()