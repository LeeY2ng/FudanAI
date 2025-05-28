import numpy as np

from FudanAI import Tensor


class Parameter(Tensor):
    """A kind of Tensor that is to be considered a module parameter."""

    def __init__(self, *shape) -> None:
        data = np.random.randn(*shape)
        super().__init__(data=data, requires_grad=True)
