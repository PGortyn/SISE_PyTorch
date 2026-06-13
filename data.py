from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import torch

RED = "red"
WHITE = "white"
ALL = "all"

class Data:
    def LoadData(self, wineType):
        # make sure proper type is passed
        if wineType != ALL and wineType != RED and wineType != WHITE:
            wineType = ALL

        # fetch dataset
        wine_quality = fetch_ucirepo(id=186)

        # data (as pandas dataframes)
        features = wine_quality.data.features.copy()
        targets = wine_quality.data.targets.copy()

        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        # targets = scaler.fit_transform(targets)

        if wineType != ALL:
            mask = features["color"] == wineType
            features = features[mask]
            targets = targets[mask]
            features.drop(columns=["color"])

        feature_tensor = torch.tensor(features, dtype=torch.float32)

        return feature_tensor, targets
