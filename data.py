from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd

RED = "red"
WHITE = "white"
ALL = "all"

ORIGIN = "origin"
QUALITY = "quality"

DATA_DIR = "data/"

COLOR_COL = "color"
QUALITY_COL = "quality"

def LoadQualityData(wineType):
    # make sure proper type is passed
    if wineType != ALL and wineType != RED and wineType != WHITE:
        wineType = ALL

    red = pd.read_csv(f'{DATA_DIR}winequality-red.csv', sep=";")
    white = pd.read_csv(f'{DATA_DIR}winequality-white.csv', sep=";")

    if wineType == RED:
        d = red
    elif wineType == WHITE:
        d = white
    else:
        red[COLOR_COL] = 0
        white[COLOR_COL] = 1
        d = pd.concat([red, white], ignore_index=True)

    # data (as pandas dataframes)
    features = d.drop(columns=[QUALITY_COL])
    targets = d[QUALITY_COL]

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    # targets = scaler.fit_transform(targets)

    feature_tensor = torch.tensor(features, dtype=torch.float32)

    return feature_tensor, targets

def LoadOriginData():
    # fetch dataset
    wine = fetch_ucirepo(id=109)

    # data (as pandas dataframes)
    features = wine.data.features
    targets = wine.data.targets

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features_tensor = torch.tensor(features, dtype=torch.float32)

    return features_tensor, targets


