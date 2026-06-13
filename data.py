from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
features = wine_quality.data.features
targets = wine_quality.data.targets

scaler = StandardScaler()

features = scaler.fit_transform(features)
# targets = scaler.fit_transform(targets)