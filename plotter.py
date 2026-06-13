import os
from matplotlib import pyplot as plt
from data import ORIGIN, QUALITY

GRAPH_PATH = "graphs/"

def MakeEssentialDirs():
    os.makedirs(GRAPH_PATH, exist_ok=True)

def PlotLearning(errors, hits, qualityMap, uMat, x = 8, y = 8, dataType = QUALITY):
    MakeEssentialDirs()
    PlotErrors(errors, dataType)
    PlotHits(hits, x, y, dataType)
    PlotTargets(qualityMap, x, y, dataType)
    PlotUMatrix(uMat, x, y, dataType)
    # plt.show()

def PlotErrors(errors, dataType):
    plt.plot(errors)
    plt.xlabel("Epoch")
    plt.ylabel("Quantization Error")
    plt.ylim(bottom=0)
    plt.title("SOM learning curve")
    plt.grid(True)
    plt.savefig(f'{GRAPH_PATH}{dataType}_quantization_error_graph')

def PlotHits(hits, x, y, dataType):
    plt.figure(figsize=(x,y))
    plt.imshow(hits.numpy(), cmap="Blues")
    plt.colorbar(label="Number of wines")
    plt.title("SOM hit map")
    plt.savefig(f'{GRAPH_PATH}SOM_{dataType}_hit_map')

def PlotTargets(map, x, y, dataType):
    plt.figure(figsize=(x,y))
    plt.imshow(map.numpy(), cmap="viridis")
    if dataType == QUALITY:
        plt.colorbar(label="Average Wine Quality")
        plt.title("Wine Quality SOM Map")
    else:
        plt.colorbar(label="Wine Origin")
        plt.title("Wine Origin SOM Map")
    plt.savefig(f'{GRAPH_PATH}SOM_{dataType}_map')

def PlotUMatrix(uMat, x, y, dataType):
    plt.figure(figsize=(x,y))
    plt.imshow(uMat.numpy(), cmap="hot")
    plt.colorbar(label="Neighbour distance")
    plt.title("SOM U-matrix")
    plt.savefig(f'{GRAPH_PATH}SOM_{dataType}_U_Matrix')