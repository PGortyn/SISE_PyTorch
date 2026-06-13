import torch
from som import SOM
from data import features

print("PyTorch version: ", torch.__version__)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f'Using {device} device')

if __name__ == "__main__":
    som = SOM(10, 10, 11)
    loader, tensor, target = som.GetWineDataloader()
    sample = tensor[0]
    row, col = som.FindBMU(sample)
    som.UpdateWeights(sample, row, col, 0.5, 5)
