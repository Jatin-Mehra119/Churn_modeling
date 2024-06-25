import pandas as pd

class Loader:
    @staticmethod
    def load(path):
        return pd.read_csv(path, index_col='CustomerId')
