
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore


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
        print("Shape:", self.data.shape)
        self.data.info()

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
        corr = self.data.corr(numeric_only=True)

        plt.figure(figsize=(10,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice di correlazione")
        plt.show()

    def plot_hist(self, col):
        plt.figure(figsize=(6,4))
        plt.hist(self.data[col].dropna(), bins=30)
        plt.title(f"Distribuzione di {col}")
        plt.show()


    def plot_distribution(self, col):
        plt.figure(figsize=(6,4))
        sns.histplot(self.data[col], kde=True)
        plt.title(f"Distribuzione di {col}")
        plt.show()

    def plot_scatter(self, x, y):
        plt.figure(figsize=(6,4))
        sns.scatterplot(data=self.data, x=x, y=y)
        plt.title(f"{x} vs {y}")
        plt.show()

    
    def plot_box(self, col):
        plt.figure(figsize=(6,4))
        sns.boxplot(y=self.data[col])
        plt.title(f"Boxplot di {col}")
        plt.show()

    def plot_counts(self, col):
        plt.figure(figsize=(6,4))
        sns.countplot(y=self.data[col])
        plt.title(f"Distribuzione di {col}")
        plt.show()


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


# Analisi descrittiva sull'intero dataset
dati1 = Persone(df)
print(dati1.mostra_cont(5))
print(dati1.struttura()) #
print(dati1.summary()) # ovviamente solo per variabili numeriche
print("Numero di missing value per colonna:", dati1.missing_values())
print("Missing value in %:",dati1.missing_percentage())
print("Matrice di correlazione",dati1.correlation_matrix())
print(dati1.plot_correlation())


# Analisi su singole variabili
#dati1.plot_hist("age")
#dati1.plot_hist("hours-per-week")
#dati1.plot_scatter("age","hours-per-week")
#dati1.plot_counts("workclass")

# Individuazione outlier:
print("Numero outlier secondo IQR:" , dati1.outliers_iqr_per_col())
print("Numero outlier secondo Z score:" , dati1.outliers_zscore_per_col())
# lo zeta score standarizza i dati, assumendo una distribuzione normale, e considera outliers
# tutti quei valori in modulo superiori a 3, perchè in teoria in una norm. stand. il 99,7%  rientra in questo range
# non funziona bene con i redditi, distr. non normale


print(dati1.valori_stringhe())


