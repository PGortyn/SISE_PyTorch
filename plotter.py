import os
from matplotlib import pyplot as plt

GRAPH_PATH = "graphs/"

def MakeEssentialDirs():
    os.makedirs(GRAPH_PATH, exist_ok=True)

def PlotLearning(errors, hits, qualityMap, uMat):
    PlotErrors(errors)
    PlotHits(hits)
    PlotQuality(qualityMap)
    PlotUMatrix(uMat)

def PlotErrors(errors):
    MakeEssentialDirs()
    plt.plot(errors)
    plt.xlabel("Epoch")
    plt.ylabel("Quantization Error")
    plt.ylim(bottom=0)
    plt.title("SOM learning curve")
    plt.grid(True)
    plt.savefig(f'{GRAPH_PATH}quantization_error_graph')
    # plt.show()

def PlotHits(hits):
    plt.figure(figsize=(8,8))
    plt.imshow(hits.numpy(), cmap="Blues")
    plt.colorbar(label="Number of wines")
    plt.title("SOM hit map")
    plt.savefig(f'{GRAPH_PATH}SOM_hit_map')
    # plt.show()

def PlotQuality(map):
    plt.figure(figsize=(8,8))
    plt.imshow(map.numpy(), cmap="viridis")
    plt.colorbar(label="Average Wine Quality")
    plt.title("Wine Quality SOM Map")
    plt.savefig(f'{GRAPH_PATH}SOM_quality_map')
    # plt.show()

def PlotUMatrix(uMat):
    plt.figure(figsize=(8, 8))
    plt.imshow(uMat.numpy(), cmap="hot")
    plt.colorbar(label="Neighbour distance")
    plt.title("SOM U-matrix")
    plt.savefig(f'{GRAPH_PATH}SOM_U_Matrix')
    plt.show()