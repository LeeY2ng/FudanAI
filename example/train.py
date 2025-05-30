import numpy as np

from FudanAI import Tensor
import FudanAI.nn as nn
import FudanAI.optim as optim


class Model(nn.Module):
    def __init__(self, in_features=3):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 5, bias=True)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(5, 1, bias=True)

    def forward(self, input):
        output = self.linear1(input)
        output = self.relu1(output)
        output = self.linear2(output)
        return output


def train(model, x, y, epoch=30):  # TODO
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    mse_loss = nn.MSELoss()
    for i in range(1, epoch + 1):
        model.zero_grad()
        output = model(x)
        loss = mse_loss(output, y)
        print(f"train: epoch {i}, loss {loss}")
        loss.backward()
        optimizer.step()


def test(model, x, y):
    output = model(x)
    mse_loss = nn.MSELoss()
    loss = mse_loss(output, y)
    print(f"test: loss {loss}")


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=10, std=0.01)


def main():
    coef = Tensor(np.array([1, 3, 2]))
    x_train = Tensor(np.random.rand(100, 3))
    y_train = x_train @ coef + 5
    x_test = Tensor(np.random.rand(20, 3))
    y_test = x_test @ coef + 5
    model = Model()
    model.apply(init_weights)
    for name, module in model.named_modules(prefix="model"):
        if isinstance(module, nn.Linear):
            print(f"{name}.weights: {module.weight}")
    train(model, x_train, y_train)
    test(model, x_test, y_test)


if __name__ == "__main__":
    main()
