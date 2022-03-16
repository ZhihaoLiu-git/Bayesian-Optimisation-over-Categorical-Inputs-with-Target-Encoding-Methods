from sktools import QuantileEncoder
import pandas as pd
from sklearn.datasets import load_boston




bunch = load_boston()
y = bunch.target
X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
enc = QuantileEncoder(cols=['CHAS', 'RAD']).fit(X, y)
numeric_dataset = enc.transform(X)
print(numeric_dataset.info())