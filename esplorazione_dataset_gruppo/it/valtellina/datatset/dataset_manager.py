
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from grafici import Grafici

gf = Grafici()

class Ds_manager:
    def __init__(self):
        self.data = self.carica_data()

    def carica_data(self):
    # fetch dataset 
        auto_mpg = fetch_ucirepo(id=9) 
  
    # data (as pandas dataframes) 
        X = auto_mpg.data.features 
        y = auto_mpg.data.targets 

        df = pd.concat([X, y], axis=1)
        return df


    def mostra_cont(self, righe):
        return self.data.head(righe) 
    
    def struttura(self):
        return
        {
            "shape" : self.data.shape,
            "info"  : self.data.info()
        }

    def summary(self):
        return self.data.describe()
    
    def valori_stringhe(self):
        cat_cols = self.data.select_dtypes(include="string").columns

        for col in cat_cols:
            print(f"\n{col} ({self.data[col].nunique()} valori unici):")
            print(self.data[col].unique())


    def missing_values(self):
        return self.data.isnull().sum().sort_values(ascending=False)
    
    def missing_percentage(self):
        return (self.data.isnull().mean()).sort_values(ascending=False)

    def correlation_matrix(self):
        return self.data.corr(numeric_only=True)

    def plot_correlation(self):
        return Grafici.plot_correlation()

    def plot_hist(self, col):
        return Grafici.plot_hist(col)
    
    def plot_distribution(self, col):
        return Grafici.plot_distribution(col)
    
    def plot_scatter(self, x, y):
        return Grafici.plot_scatter(x,y)

    def plot_box(self, col):
        return Grafici.plot_box(col)
    
    def plot_counts(self, col):
        return Grafici.plot_counts(col)


    def outliers_iqr_per_col(self):
        result = {}

        numeric_cols = self.data.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers_count = self.data[(self.data[col] < lower) | (self.data[col] > upper)].shape[0]
            result[col] = outliers_count

        return result


    def outliers_zscore_per_col(self, threshold=3):
        result = {}

        numeric_cols = self.data.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            col_data = self.data[col].dropna()

            if col_data.std() == 0:
                result[col] = 0
                continue

            z_scores = zscore(col_data)

            outliers_count = np.sum(np.abs(z_scores) > threshold)
            result[col] = int(outliers_count)

        return result






