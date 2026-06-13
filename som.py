import torch

class SOM:

    def __init__(self, width, height, dim, lr = 0.5, sigma = None, seed = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.width = width
        self.height = height
        self.inputDim = dim
        self.startLR = lr
        if sigma is None:
            sigma = max(width, height) / 2
        self.startSigma = sigma
        self.weights = (torch.rand(height, width, dim) * 2) - 1
        self.gridX, self.gridY = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    def Train(self, data, maxEpochs, minError = 0.5):
        errors = []
        for epoch in range(maxEpochs):
            lr = self.DecayLearningRate(epoch, maxEpochs)
            sigma = self.DecaySigma(epoch, maxEpochs)
            indices = torch.randperm(len(data))
            shuffled = data[indices]
            for sample in shuffled:
                row, col = self.FindBMU(sample)
                self.UpdateWeights(sample, row, col, lr, sigma)
            qe = self.QuantizationError(data)
            errors.append(qe)
            print(f'Epoch {epoch + 1} / {maxEpochs}\n  Quantization error: {qe:.4f}')
            if qe <= minError:
                break
        return errors

    # Best Matching Unit
    def FindBMU(self, sample):
        distances = torch.norm(self.weights - sample, dim = 2)
        bmuIndex = torch.argmin(distances)

        row = bmuIndex // self.width
        col = bmuIndex % self.width

        return row, col

    def UpdateWeights(self, sample, row, col, lr, sigma):
        influence = self.Neighbourhood(row, col, sigma)
        influence = influence.unsqueeze(-1)
        self.weights += lr * influence * (sample - self.weights)

    def Neighbourhood(self, row, col, sigma):
        distSq = (self.gridX - row) ** 2 + (self.gridY - col) ** 2
        return torch.exp(-distSq / (2 * (sigma ** 2)))

    def DecayLearningRate(self, epoch, maxEpochs):
        return self.startLR * (1 - (epoch / maxEpochs))

    def DecaySigma(self, epoch, maxEpochs):
        s = self.startSigma * (1 - (epoch / maxEpochs))
        return max(0.5, s)

    def QuantizationError(self, data):
        error = 0.0
        for sample in data:
            row, col = self.FindBMU(sample)
            bmuWeight = self.weights[row, col]
            error += torch.norm(sample - bmuWeight).item()
        return error / len(data)

    def CreateHitMap(self, data):
        hits = torch.zeros(self.height, self.width)
        for sample in data:
            row, col = self.FindBMU(sample)
            hits[row, col] += 1

        return hits

    def CreateQualityMap(self, data, labels):
        qualitySum = torch.zeros(self.height, self.width)
        counts = torch.zeros(self.height, self.width)
        labels = labels.squeeze()
        for sample, quality in zip(data, labels):
            row, col = self.FindBMU(sample)
            qualitySum[row, col] += float(quality)
            counts[row, col] += 1
        qualityMap = torch.where(counts > 0, qualitySum / counts, torch.tensor(float("nan")))

        return qualityMap

    def CreateUMatrix(self):
        uMatrix = torch.zeros(self.height, self.width)

        for row in range(self.height):
            for col in range(self.width):
                neighbours = []
                current = self.weights[row, col]
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

                for dr, dc in directions:
                    nr = row + dr
                    nc = col + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        neighbour = self.weights[nr, nc]
                        dist = torch.norm(current - neighbour)
                        neighbours.append(dist.item())
                if neighbours:
                    uMatrix[row, col] = sum(neighbours) / len(neighbours)
        return uMatrix