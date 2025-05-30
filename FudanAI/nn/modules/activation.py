from FudanAI import Tensor
import FudanAI.autograd.functional as F
from .module import Module


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return 1 / (1 + F.exp(-input))


class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)
