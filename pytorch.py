import torch
from som import SOM
from data import LoadQualityData, LoadOriginData, WHITE, RED, ALL, ORIGIN, QUALITY
from plotter import PlotLearning
import argparse
import ast
import sys

print("PyTorch version: ", torch.__version__)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f'Using {device} device')

def HandleArguments():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-X", type=ast.literal_eval)
    argParser.add_argument("-Y", type=ast.literal_eval)
    argParser.add_argument("-wine_type", type=str)
    argParser.add_argument("-lr", type=ast.literal_eval, help="LEARNING RATE")
    argParser.add_argument("-sigma", type=ast.literal_eval)
    argParser.add_argument("-epochs", type=ast.literal_eval)
    argParser.add_argument("-min_error", type=ast.literal_eval)
    argParser.add_argument("-seed", type=ast.literal_eval)
    argParser.add_argument("-data", type=str)
    args = argParser.parse_args()
    sizeX = args.X
    sizeY = args.Y
    if sizeX is None:
        sizeX = 10
    if sizeY is None:
        sizeY = sizeX
    wineType = args.wine_type
    if wineType != ALL and wineType != WHITE and wineType != RED:
        wineType = ALL
    print(f'Wine type: {wineType}')
    lr = args.lr
    if lr is None:
        lr = 0.5
    sigma = args.sigma
    epochs = args.epochs
    if epochs is None:
        epochs = 50
    err = args.min_error
    if err is None:
        err = 0.5
    seed = args.seed
    dataType = args.data
    if dataType is None or (dataType != QUALITY and dataType != ORIGIN):
        dataType = ORIGIN
    if dataType == QUALITY:
        tensor, targets = LoadQualityData(wineType)
    else:
        tensor, targets = LoadOriginData()
    som = SOM(sizeX, sizeY, tensor.shape[1], lr, sigma, seed)
    errors = som.Train(tensor, epochs, err)
    hits = som.CreateHitMap(tensor)
    qMap = som.CreateQualityMap(tensor, targets)
    uMatrix = som.CreateUMatrix()
    PlotLearning(errors, hits, qMap, uMatrix, sizeX, sizeY, dataType)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        HandleArguments()
    else:
        # tensor, targets = LoadQualityData(RED)
        tensor, targets = LoadOriginData()
        som = SOM(10, 10, tensor.shape[1])
        errors = som.Train(tensor, 50)
        hits = som.CreateHitMap(tensor)
        qMap = som.CreateQualityMap(tensor, targets)
        uMatrix = som.CreateUMatrix()
        PlotLearning(errors, hits, qMap, uMatrix, dataType=ORIGIN)
