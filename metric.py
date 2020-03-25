import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.manifold import TSNE


class Metric:
    def __init__(self, fname, days):
        self.file = pd.read_csv(fname)
        self.data = self.file.values[:, -1*days:]
        self.data_embedded = TSNE(n_components=3).fit_transform(self.data)

    def get_pearson(self, stock1: int, stock2: int):
        return pearsonr(self.data[stock1], self.data[stock2])

    def get_spearman(self, stock1: int, stock2: int):
        return spearmanr(self.data[stock1], self.data[stock2])

    def get_tsne(self, stock: int):
        return self.data_embedded[stock]


metric = Metric('residuals.csv', 50)
print(metric.get_pearson(2, 5))
print(metric.get_spearman(2, 11))
print(metric.get_tsne(51))