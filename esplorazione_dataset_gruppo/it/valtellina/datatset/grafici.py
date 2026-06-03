import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class Grafici:

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