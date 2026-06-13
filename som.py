import torch
from torch import nn, optim
from data import features, targets
from torch.utils.data import TensorDataset, DataLoader

class SOM:

    def __init__(self, width, height, dim, lr = 0.5, sigma = None):
        self.width = width
        self.height = height
        self.inputDim = dim
        self.lr = lr

        if sigma is None:
            sigma = max(width, height) / 2

        self.sigma = sigma

        self.weights = torch.rand(height, width, dim)
        self.gridX, self.gridY = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    # Best Matching Unit
    def FindBMU(self, sample):
        distances = torch.norm(self.weights - sample, dim = 2)
        bmuIndex = torch.argmin(distances)

        row = bmuIndex // self.width
        col = bmuIndex % self.width

        return row, col

    def Neighbourhood(self, row, col, sigma):
        distSq = (self.gridX - row) ** 2 + (self.gridY - col) ** 2
        return torch.exp(-distSq / (2 * sigma ** 2))

    def UpdateWeights(self, sample, row, col, lr, sigma):
        influence = self.Neighbourhood(row, col, sigma)
        influence = influence.unsqueeze(-1)
        self.weights += lr * influence * (sample - self.weights)

    @staticmethod
    def GetWineDataloader(batchSize=64):
        feature_tensor = torch.tensor(features, dtype=torch.float32)

        dataset = TensorDataset(feature_tensor)

        dataLoader = DataLoader(dataset, batchSize, shuffle=True)

        return dataLoader, feature_tensor, targets

