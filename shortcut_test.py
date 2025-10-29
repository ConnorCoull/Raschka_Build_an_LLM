import torch
import torch.nn as nn
from utils.shortcut_utils import ExampleDeepNeuralNetwork

torch.manual_seed(123)

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)

def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


print_gradients(model_with_shortcut, sample_input)
# ^ not nearly as significant a difference as should be - perhaps shortcut isn't implemented properly?
# Fixed when shortcut model declaration is commented out? why? Also works well when order is with shortcut, without shortcut
print("-" * 20)
print_gradients(model_without_shortcut, sample_input)
#print("-" * 20)